from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from src.research.v10b4_xauusd_intraday_context_audit import (
    CONTEXT_FAMILIES,
    PREDICTION_TARGET,
    rank_context_candidates,
    run_audit,
)


@dataclass
class _Symbol:
    name: str
    description: str = ""
    path: str = ""
    currency_base: str = ""
    currency_profit: str = ""
    currency_margin: str = "USD"
    digits: int = 2
    point: float = 0.01
    trade_tick_size: float = 0.01
    trade_tick_value: float = 0.1
    trade_contract_size: float = 100.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    spread: int = 20
    trade_mode: int = 4


@dataclass
class _Tick:
    time: int
    time_msc: int
    bid: float = 100.0
    ask: float = 100.1


class _FakeMt5:
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15

    def __init__(self, symbols: list[_Symbol], *, future_symbol: str | None = None, history: bool = True) -> None:
        self._symbols = symbols
        self._by_name = {item.name: item for item in symbols}
        self.future_symbol = future_symbol
        self.history = history
        self.rate_calls: list[tuple[str, int, int, int]] = []

    def symbols_get(self):
        return tuple(self._symbols)

    def symbol_info(self, name: str):
        return self._by_name.get(name)

    def symbol_select(self, name: str, enabled: bool):
        return bool(enabled and name in self._by_name)

    def symbol_info_tick(self, name: str):
        base = int(datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc).timestamp())
        if name == self.future_symbol:
            base += 120
        return _Tick(time=base, time_msc=base * 1000)

    def copy_rates_from_pos(self, name: str, timeframe: int, start_pos: int, count: int):
        self.rate_calls.append((name, timeframe, start_pos, count))
        if not self.history:
            return []
        step = 300 if timeframe == self.TIMEFRAME_M5 else 900
        rows = 1_200 if timeframe == self.TIMEFRAME_M5 else 500
        end = int(datetime(2026, 1, 10, 11, 55, tzinfo=timezone.utc).timestamp())
        first = end - (rows - 1) * step
        return [{"time": first + index * step, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0} for index in range(rows)]

    def last_error(self):
        return (0, "OK")


def _gold() -> _Symbol:
    return _Symbol("XAUUSD", "Gold vs US Dollar", currency_base="XAU", currency_profit="USD")


def _silver(name: str = "XAGUSD") -> _Symbol:
    return _Symbol(name, "Silver vs US Dollar", currency_base="XAG", currency_profit="USD")


def _usdjpy(name: str = "USDJPY") -> _Symbol:
    return _Symbol(name, "US Dollar vs Japanese Yen", currency_base="USD", currency_profit="JPY", digits=3, point=0.001)


class V10B4XAUUSDIntradayContextAuditTests(unittest.TestCase):
    def test_prediction_target_is_xauusd_only(self) -> None:
        self.assertEqual(PREDICTION_TARGET, "XAUUSD")
        with tempfile.TemporaryDirectory() as tmp:
            mt5 = _FakeMt5([_gold(), _silver(), _usdjpy()])
            report = run_audit(
                mt5,
                root=Path(tmp),
                persist=False,
                captured_at=datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc),
            )
        self.assertEqual(report["prediction_target"], "XAUUSD")
        self.assertTrue(report["context_read_only"])
        self.assertEqual(report["model_promotion"], "NONE")
        self.assertEqual(report["automatic_execution"], "DISABLED")
        self.assertFalse(report["production_integration"])

    def test_suffix_symbol_discovery_prefers_real_xag_pair(self) -> None:
        family = next(item for item in CONTEXT_FAMILIES if item.key == "SILVER")
        wrong = _Symbol("GOLDMINER", "Silver mining equity", currency_base="", currency_profit="USD")
        suffixed = _silver("XAGUSD.a")
        ranked = rank_context_candidates([wrong, suffixed], family)
        self.assertEqual(ranked[0].name, "XAGUSD.a")

    def test_two_core_contexts_make_audit_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mt5 = _FakeMt5([_gold(), _silver(), _usdjpy()])
            report = run_audit(
                mt5,
                root=Path(tmp),
                persist=True,
                captured_at=datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc),
            )
            output = Path(tmp) / "reports" / "XAUUSD" / "v10b4_intraday_context_audit.json"
            self.assertTrue(output.exists())
        self.assertEqual(report["decision"], "INTRADAY_CONTEXT_READY_FOR_RESEARCH")
        self.assertGreaterEqual(report["core_ready_count"], 2)

    def test_future_tick_age_is_not_clamped_and_family_is_invalid(self) -> None:
        mt5 = _FakeMt5([_gold(), _silver(), _usdjpy()], future_symbol="XAGUSD")
        report = run_audit(
            mt5,
            persist=False,
            captured_at=datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc),
        )
        silver = next(item for item in report["families"] if item["family"] == "SILVER")
        self.assertEqual(silver["status"], "INVALID")
        self.assertEqual(silver["reason"], "FUTURE_TIMESTAMP_GUARD")
        self.assertLess(silver["tick"]["age_seconds"], 0.0)

    def test_no_context_fails_closed(self) -> None:
        report = run_audit(
            _FakeMt5([_gold()], history=False),
            persist=False,
            captured_at=datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(report["decision"], "NO_VERIFIED_INTRADAY_CONTEXT")
        self.assertEqual(report["ready_family_count"], 0)

    def test_history_probe_uses_completed_bar_offset(self) -> None:
        mt5 = _FakeMt5([_gold(), _silver(), _usdjpy()])
        run_audit(
            mt5,
            persist=False,
            captured_at=datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc),
        )
        self.assertTrue(mt5.rate_calls)
        self.assertTrue(all(start_pos == 1 for _, _, start_pos, _ in mt5.rate_calls))


if __name__ == "__main__":
    unittest.main()
