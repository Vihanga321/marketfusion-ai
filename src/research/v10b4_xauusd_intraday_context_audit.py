"""V1.0B.4 read-only MT5 intraday intermarket availability audit.

XAUUSD remains the only prediction target. Other instruments are inspected only
as possible causal context sensors for later research. This module never opens,
changes, or closes trades and performs no model promotion.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.assets.contracts import ROOT, asset_paths, discover_broker_symbol

CONTRACT_VERSION = "v1.0b4-xauusd-intraday-context-audit-v2"
PREDICTION_TARGET = "XAUUSD"
M5_AUDIT_ROWS = 5_000
M15_AUDIT_ROWS = 2_000
MIN_READY_M5_ROWS = 1_000
MIN_READY_M15_ROWS = 400
FUTURE_TICK_TOLERANCE_SECONDS = 5.0
MAX_BROKER_CLOCK_OFFSET_SECONDS = 14 * 60 * 60
BROKER_CLOCK_QUANTUM_SECONDS = 15 * 60
BROKER_CLOCK_RESIDUAL_TOLERANCE_SECONDS = 120.0


@dataclass(frozen=True)
class ContextFamily:
    key: str
    label: str
    tier: str
    exact_names: tuple[str, ...] = ()
    token_groups: tuple[tuple[str, ...], ...] = ()
    currency_base: str | None = None
    currency_profit: str | None = None
    excluded_tokens: tuple[str, ...] = ()


CONTEXT_FAMILIES: tuple[ContextFamily, ...] = (
    ContextFamily(
        "SILVER", "Silver / U.S. Dollar", "CORE",
        exact_names=("XAGUSD",), token_groups=(("XAG", "USD"), ("SILVER",)),
        currency_base="XAG", currency_profit="USD",
    ),
    ContextFamily(
        "USD_INDEX", "U.S. Dollar Index", "CORE",
        exact_names=("DXY", "USDX", "USDINDEX", "DOLLARINDEX"),
        token_groups=(("DXY",), ("USDX",), ("DOLLAR", "INDEX"), ("USD", "INDEX")),
    ),
    ContextFamily(
        "USDJPY", "U.S. Dollar / Japanese Yen", "CORE",
        exact_names=("USDJPY",), token_groups=(("USD", "JPY"),),
        currency_base="USD", currency_profit="JPY",
    ),
    ContextFamily(
        "EURUSD", "Euro / U.S. Dollar", "CORE",
        exact_names=("EURUSD",), token_groups=(("EUR", "USD"),),
        currency_base="EUR", currency_profit="USD",
    ),
    ContextFamily(
        "US500", "U.S. 500 equity-risk proxy", "RISK",
        exact_names=("US500", "SPX500", "SP500", "USA500"),
        token_groups=(("S&P", "500"), ("SPX", "500"), ("US", "500")),
    ),
    ContextFamily(
        "US100", "U.S. 100 technology-risk proxy", "RISK",
        exact_names=("US100", "NAS100", "USTEC", "NASDAQ100"),
        token_groups=(("NASDAQ", "100"), ("NAS", "100"), ("US", "100")),
    ),
    ContextFamily(
        "WTI", "WTI crude-oil proxy", "SECONDARY",
        exact_names=("XTIUSD", "USOIL", "WTI", "WTIUSD"),
        token_groups=(("WTI",), ("US", "OIL")), excluded_tokens=("BRENT",),
    ),
    ContextFamily(
        "US10Y", "U.S. 10-year Treasury market proxy", "SECONDARY",
        exact_names=("US10Y", "US10YR", "UST10Y", "TNOTE10"),
        token_groups=(("10", "YEAR", "TREASURY"), ("10Y", "TREASURY"), ("10", "YEAR", "NOTE")),
    ),
)


def _symbol_text(symbol: Any) -> str:
    fields = (
        getattr(symbol, "name", ""), getattr(symbol, "description", ""),
        getattr(symbol, "path", ""), getattr(symbol, "currency_base", ""),
        getattr(symbol, "currency_profit", ""),
    )
    return " ".join(str(value) for value in fields).upper()


def _normalized_name(value: str) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _identity_guard(symbol: Any, family: ContextFamily) -> bool:
    """Reject ticker collisions and unrelated securities before ranking.

    MT5 catalogs can contain stocks/ETFs whose ticker happens to be USDX, WTI,
    IEF, etc. Exact ticker text alone is therefore not sufficient identity
    evidence for non-FX context families.
    """
    text = _symbol_text(symbol)
    description = str(getattr(symbol, "description", "")).upper()
    path = str(getattr(symbol, "path", "")).upper()
    base = str(getattr(symbol, "currency_base", "")).upper()
    profit = str(getattr(symbol, "currency_profit", "")).upper()

    if family.currency_base and family.currency_profit:
        return (base, profit) == (family.currency_base, family.currency_profit)

    if family.key == "USD_INDEX":
        identity = (
            "DOLLAR INDEX" in text
            or "USD INDEX" in text
            or "US DOLLAR INDEX" in text
            or "DXY INDEX" in text
        )
        security_collision = any(token in f"{description} {path}" for token in (" ETF", "EQUITY", "STOCK", "SHARE"))
        return identity and not security_collision

    if family.key == "US500":
        identity = "500" in text and any(token in text for token in ("SPX", "S&P", "US 500", "US500"))
        return identity and ("INDEX" in text or "INDICES" in path)

    if family.key == "US100":
        identity = "100" in text and any(token in text for token in ("NASDAQ", "NAS100", "USTEC", "US TECH", "US100"))
        return identity and ("INDEX" in text or "INDICES" in path)

    if family.key == "WTI":
        oil_identity = any(token in text for token in ("CRUDE OIL", "WTI OIL", "WEST TEXAS", "US OIL", "USOIL", "XTIUSD"))
        security_collision = any(token in f"{description} {path}" for token in (" INC", " CORP", " LTD", " PLC", " ETF", "EQUITY", "STOCK", "SHARE"))
        return oil_identity and not security_collision

    if family.key == "US10Y":
        direct_name = _normalized_name(getattr(symbol, "name", "")) in {_normalized_name(item) for item in family.exact_names}
        treasury_identity = any(token in text for token in ("10 YEAR TREASURY", "10-YEAR TREASURY", "10Y TREASURY", "10 YEAR NOTE", "10-YEAR NOTE"))
        security_collision = any(token in f"{description} {path}" for token in (" ETF", "ISHARES", "PROSHARES", "EQUITY", "STOCK", "SHARE"))
        return (direct_name or treasury_identity) and not security_collision

    return True


def _candidate_score(symbol: Any, family: ContextFamily) -> int:
    name = str(getattr(symbol, "name", ""))
    normalized = _normalized_name(name)
    text = _symbol_text(symbol)
    if any(token.upper() in text for token in family.excluded_tokens):
        return -1
    if not _identity_guard(symbol, family):
        return -1

    score = 0
    exact = {_normalized_name(item) for item in family.exact_names}
    if normalized in exact:
        score += 100
    elif any(normalized.startswith(item) or item in normalized for item in exact):
        score += 55

    base = str(getattr(symbol, "currency_base", "")).upper()
    profit = str(getattr(symbol, "currency_profit", "")).upper()
    if family.currency_base and family.currency_profit:
        if (base, profit) == (family.currency_base, family.currency_profit):
            score += 90
        else:
            return -1

    for group in family.token_groups:
        if all(token.upper() in text for token in group):
            score += 25 + 5 * len(group)
            break
    return score if score > 0 else -1


def rank_context_candidates(symbols: Iterable[Any], family: ContextFamily) -> list[Any]:
    scored = [(item, _candidate_score(item, family)) for item in symbols]
    scored = [(item, score) for item, score in scored if score >= 0]
    return [
        item for item, _ in sorted(
            scored,
            key=lambda pair: (-pair[1], len(str(getattr(pair[0], "name", ""))), str(getattr(pair[0], "name", ""))),
        )
    ]


def _tick_epoch_seconds(tick: Any) -> float | None:
    if tick is None:
        return None
    raw_time = getattr(tick, "time", None)
    raw_msc = getattr(tick, "time_msc", None)
    try:
        if raw_msc not in (None, 0):
            value = float(raw_msc) / 1000.0
        elif raw_time not in (None, 0):
            value = float(raw_time)
        else:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value > 0 else None


def _calibrate_broker_clock(tick: Any, captured_at: datetime) -> dict[str, object]:
    """Measure a fixed MT5 server clock offset from the XAUUSD live tick.

    The offset is measured, never assumed. We accept only timezone-like offsets
    close to a 15-minute quantum and preserve the observed residual for audit.
    """
    epoch_seconds = _tick_epoch_seconds(tick)
    if epoch_seconds is None:
        return {
            "status": "UNRESOLVED",
            "reason": "XAUUSD_LIVE_TICK_TIMESTAMP_UNAVAILABLE",
            "observed_offset_seconds": None,
            "normalized_offset_seconds": None,
            "residual_seconds": None,
        }
    observed = epoch_seconds - captured_at.timestamp()
    normalized = round(observed / BROKER_CLOCK_QUANTUM_SECONDS) * BROKER_CLOCK_QUANTUM_SECONDS
    residual = observed - normalized
    if abs(normalized) > MAX_BROKER_CLOCK_OFFSET_SECONDS or abs(residual) > BROKER_CLOCK_RESIDUAL_TOLERANCE_SECONDS:
        return {
            "status": "UNRESOLVED",
            "reason": "XAUUSD_CLOCK_OFFSET_NOT_TIMEZONE_LIKE",
            "observed_offset_seconds": float(observed),
            "normalized_offset_seconds": None,
            "residual_seconds": float(residual),
        }
    return {
        "status": "PASS",
        "reason": "MEASURED_FROM_XAUUSD_LIVE_TICK",
        "observed_offset_seconds": float(observed),
        "normalized_offset_seconds": int(normalized),
        "residual_seconds": float(residual),
        "raw_tick_utc": datetime.fromtimestamp(epoch_seconds, timezone.utc).isoformat(),
        "normalized_tick_utc": datetime.fromtimestamp(epoch_seconds - normalized, timezone.utc).isoformat(),
    }


def _rates_summary(rates: Any, captured_at: datetime, clock_offset_seconds: int) -> dict[str, object]:
    if rates is None:
        return {"rows": 0, "first_bar_utc": None, "last_bar_utc": None, "duplicates": 0, "future_bars": 0}
    frame = pd.DataFrame(rates)
    if frame.empty or "time" not in frame.columns:
        return {"rows": 0, "first_bar_utc": None, "last_bar_utc": None, "duplicates": 0, "future_bars": 0}
    raw_timestamps = pd.to_datetime(pd.to_numeric(frame["time"], errors="coerce"), unit="s", utc=True, errors="coerce").dropna()
    if raw_timestamps.empty:
        return {"rows": 0, "first_bar_utc": None, "last_bar_utc": None, "duplicates": 0, "future_bars": 0}
    timestamps = raw_timestamps - pd.Timedelta(seconds=clock_offset_seconds)
    captured = pd.Timestamp(captured_at)
    if captured.tzinfo is None:
        captured = captured.tz_localize("UTC")
    else:
        captured = captured.tz_convert("UTC")
    return {
        "rows": int(len(timestamps)),
        "raw_first_bar_utc": pd.Timestamp(raw_timestamps.min()).isoformat(),
        "raw_last_bar_utc": pd.Timestamp(raw_timestamps.max()).isoformat(),
        "first_bar_utc": pd.Timestamp(timestamps.min()).isoformat(),
        "last_bar_utc": pd.Timestamp(timestamps.max()).isoformat(),
        "clock_offset_seconds_applied": int(clock_offset_seconds),
        "duplicates": int(timestamps.duplicated().sum()),
        "future_bars": int((timestamps > captured + pd.Timedelta(seconds=FUTURE_TICK_TOLERANCE_SECONDS)).sum()),
    }


def _tick_summary(tick: Any, captured_at: datetime, clock_offset_seconds: int) -> dict[str, object]:
    if tick is None:
        return {"status": "UNAVAILABLE", "bid": None, "ask": None, "raw_time": None, "raw_time_msc": None, "tick_utc": None, "age_seconds": None}
    raw_time = getattr(tick, "time", None)
    raw_msc = getattr(tick, "time_msc", None)
    epoch_seconds = _tick_epoch_seconds(tick)
    if epoch_seconds is None:
        return {
            "status": "UNAVAILABLE",
            "bid": float(getattr(tick, "bid", 0.0)), "ask": float(getattr(tick, "ask", 0.0)),
            "raw_time": raw_time, "raw_time_msc": raw_msc, "tick_utc": None, "age_seconds": None,
        }
    try:
        raw_tick_utc = datetime.fromtimestamp(epoch_seconds, timezone.utc)
        tick_utc = datetime.fromtimestamp(epoch_seconds - clock_offset_seconds, timezone.utc)
        raw_age = (captured_at - raw_tick_utc).total_seconds()
        age = (captured_at - tick_utc).total_seconds()
    except (TypeError, ValueError, OSError, OverflowError):
        return {"status": "INVALID_TIMESTAMP", "bid": getattr(tick, "bid", None), "ask": getattr(tick, "ask", None), "raw_time": raw_time, "raw_time_msc": raw_msc, "tick_utc": None, "age_seconds": None}
    status = "FUTURE_TIMESTAMP" if age < -FUTURE_TICK_TOLERANCE_SECONDS else "AVAILABLE"
    return {
        "status": status,
        "bid": float(getattr(tick, "bid", 0.0)), "ask": float(getattr(tick, "ask", 0.0)),
        "raw_time": raw_time, "raw_time_msc": raw_msc,
        "raw_tick_utc": raw_tick_utc.isoformat(), "raw_age_seconds": float(raw_age),
        "tick_utc": tick_utc.isoformat(), "age_seconds": float(age),
        "clock_offset_seconds_applied": int(clock_offset_seconds),
    }


def _audit_family(
    mt5: Any,
    symbols: list[Any],
    family: ContextFamily,
    captured_at: datetime,
    clock_offset_seconds: int,
) -> dict[str, object]:
    candidates = rank_context_candidates(symbols, family)
    candidate_names = [str(getattr(item, "name", "")) for item in candidates[:5]]
    if not candidates:
        return {"family": family.key, "label": family.label, "tier": family.tier, "status": "UNAVAILABLE", "reason": "NO_VERIFIED_BROKER_SYMBOL_CANDIDATE", "candidate_names": []}

    selected_name: str | None = None
    select_errors: list[str] = []
    for candidate in candidates[:5]:
        name = str(getattr(candidate, "name", ""))
        try:
            if bool(mt5.symbol_select(name, True)):
                selected_name = name
                break
        except Exception as exc:
            select_errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if selected_name is None:
        return {"family": family.key, "label": family.label, "tier": family.tier, "status": "UNAVAILABLE", "reason": "SYMBOL_SELECT_FAILED", "candidate_names": candidate_names, "errors": select_errors}

    info = mt5.symbol_info(selected_name)
    tick = _tick_summary(mt5.symbol_info_tick(selected_name), captured_at, clock_offset_seconds)
    m5 = _rates_summary(mt5.copy_rates_from_pos(selected_name, getattr(mt5, "TIMEFRAME_M5"), 1, M5_AUDIT_ROWS), captured_at, clock_offset_seconds)
    m15 = _rates_summary(mt5.copy_rates_from_pos(selected_name, getattr(mt5, "TIMEFRAME_M15"), 1, M15_AUDIT_ROWS), captured_at, clock_offset_seconds)

    invalid_future = tick.get("status") == "FUTURE_TIMESTAMP" or int(m5["future_bars"]) > 0 or int(m15["future_bars"]) > 0
    ready_history = int(m5["rows"]) >= MIN_READY_M5_ROWS and int(m15["rows"]) >= MIN_READY_M15_ROWS
    if invalid_future:
        status, reason = "INVALID", "FUTURE_TIMESTAMP_GUARD"
    elif ready_history:
        status, reason = "READY", "BOUNDED_COMPLETED_HISTORY_AVAILABLE"
    elif int(m5["rows"]) > 0 or int(m15["rows"]) > 0:
        status, reason = "LIMITED", "INSUFFICIENT_BOUNDED_HISTORY"
    else:
        status, reason = "UNAVAILABLE", "NO_COMPLETED_HISTORY"

    return {
        "family": family.key, "label": family.label, "tier": family.tier,
        "status": status, "reason": reason, "broker_symbol": selected_name,
        "candidate_names": candidate_names,
        "description": str(getattr(info, "description", "")) if info is not None else "",
        "path": str(getattr(info, "path", "")) if info is not None else "",
        "currency_base": str(getattr(info, "currency_base", "")) if info is not None else "",
        "currency_profit": str(getattr(info, "currency_profit", "")) if info is not None else "",
        "tick": tick, "M5": m5, "M15": m15,
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    temporary.replace(path)


def run_audit(mt5: Any, *, root: Path = ROOT, persist: bool = True, captured_at: datetime | None = None) -> dict[str, object]:
    captured = captured_at or datetime.now(timezone.utc)
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=timezone.utc)
    else:
        captured = captured.astimezone(timezone.utc)

    gold = discover_broker_symbol(mt5, PREDICTION_TARGET)
    if not mt5.symbol_select(gold.broker_symbol, True):
        raise RuntimeError(f"MT5 symbol_select({gold.broker_symbol!r}) failed for broker-clock calibration")
    broker_clock = _calibrate_broker_clock(mt5.symbol_info_tick(gold.broker_symbol), captured)
    if broker_clock["status"] != "PASS":
        payload: dict[str, object] = {
            "contract_version": CONTRACT_VERSION,
            "captured_at_utc": captured.isoformat(),
            "prediction_target": PREDICTION_TARGET,
            "prediction_broker_symbol": gold.broker_symbol,
            "context_read_only": True,
            "decision": "NO_VERIFIED_INTRADAY_CONTEXT",
            "next_phase": "RESOLVE_MT5_BROKER_CLOCK_BEFORE_CONTEXT_RESEARCH",
            "broker_clock_calibration": broker_clock,
            "core_ready_count": 0,
            "ready_family_count": 0,
            "invalid_family_count": 0,
            "families": [],
            "model_promotion": "NONE",
            "production_integration": False,
            "automatic_execution": "DISABLED",
            "runtime": "SHADOW_ADVISORY_ONLY",
            "manual_confirmation": "REQUIRED",
        }
        if persist:
            report = asset_paths(PREDICTION_TARGET, root).reports / "v10b4_intraday_context_audit.json"
            _atomic_json(report, payload)
        return payload

    clock_offset_seconds = int(broker_clock["normalized_offset_seconds"])
    symbols_raw = mt5.symbols_get()
    if symbols_raw is None:
        raise RuntimeError(f"MT5 symbols_get failed: {mt5.last_error()}")
    symbols = list(symbols_raw)
    families = [_audit_family(mt5, symbols, family, captured, clock_offset_seconds) for family in CONTEXT_FAMILIES]

    core_ready = [item for item in families if item["tier"] == "CORE" and item["status"] == "READY"]
    all_ready = [item for item in families if item["status"] == "READY"]
    invalid = [item for item in families if item["status"] == "INVALID"]
    if len(core_ready) >= 2 and not invalid:
        decision = "INTRADAY_CONTEXT_READY_FOR_RESEARCH"
        next_phase = "COLLECT_AND_EVALUATE_VERIFIED_INTRADAY_CONTEXT"
    elif all_ready:
        decision = "LIMITED_INTRADAY_CONTEXT_AVAILABLE"
        next_phase = "REVIEW_AVAILABLE_CONTEXT_BEFORE_FEATURE_RESEARCH"
    else:
        decision = "NO_VERIFIED_INTRADAY_CONTEXT"
        next_phase = "ADD_EXTERNAL_VERIFIED_INTRADAY_PROVIDER"

    payload = {
        "contract_version": CONTRACT_VERSION,
        "captured_at_utc": captured.isoformat(),
        "prediction_target": PREDICTION_TARGET,
        "prediction_broker_symbol": gold.broker_symbol,
        "context_read_only": True,
        "broker_clock_calibration": broker_clock,
        "decision": decision,
        "next_phase": next_phase,
        "core_ready_count": len(core_ready),
        "ready_family_count": len(all_ready),
        "invalid_family_count": len(invalid),
        "families": families,
        "model_promotion": "NONE",
        "production_integration": False,
        "automatic_execution": "DISABLED",
        "runtime": "SHADOW_ADVISORY_ONLY",
        "manual_confirmation": "REQUIRED",
    }
    if persist:
        report = asset_paths(PREDICTION_TARGET, root).reports / "v10b4_intraday_context_audit.json"
        _atomic_json(report, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"MetaTrader5 unavailable: {exc}")
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize() failed: {mt5.last_error()}")
    try:
        report = run_audit(mt5, persist=not args.no_persist)
        print(json.dumps(report, indent=2, default=str))
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
