from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest

import pandas as pd

from src.assets.contracts import BrokerSymbolSpec, asset_paths, discover_broker_symbol, get_asset
from src.assets.registry import approved_champions, empty_promotion_state, load_asset_registry, model_id
from src.evaluation.v08_observations import observation_id
from src.research.v10a_targets import FINAL_BAND, build_target_research, neutral_band_return
from src.runtime.v10a_asset_state import build_asset_state


def info(name: str, base: str = "XAU", profit: str = "USD", description: str = "Gold vs US Dollar") -> SimpleNamespace:
    return SimpleNamespace(
        name=name, description=description, digits=2, point=.01,
        trade_tick_size=.01, trade_tick_value=.1, trade_contract_size=100,
        volume_min=.01, volume_max=100, volume_step=.01, spread=45,
        trade_mode=4, currency_base=base, currency_profit=profit,
        currency_margin="USD",
    )


class FakeMt5:
    def __init__(self) -> None:
        self.items = [info("GOLD", "USD", "USD", "Barrick Gold Corporation"), info("XAUUSDm")]

    def symbols_get(self):
        return self.items

    def symbol_info(self, name):
        return next(item for item in self.items if item.name == name)

    def last_error(self):
        return (1, "Success")


class XauusdAssetTests(unittest.TestCase):
    def test_asset_is_precious_metal_not_forex(self):
        asset = get_asset("gold")
        self.assertEqual((asset.asset_id, asset.asset_class, asset.display_symbol), ("XAUUSD", "PRECIOUS_METAL", "XAU/USD"))

    def test_discovery_rejects_gold_equity_and_accepts_broker_suffix(self):
        spec = discover_broker_symbol(FakeMt5(), "XAUUSD")
        self.assertEqual(spec.broker_symbol, "XAUUSDm")

    def test_point_tick_and_atr_normalization_use_broker_contract(self):
        spec = BrokerSymbolSpec.from_mt5("XAUUSD", info("XAUUSD"))
        self.assertAlmostEqual(spec.price_to_points(.47), 47)
        self.assertAlmostEqual(spec.price_to_ticks(.47), 47)
        self.assertAlmostEqual(spec.points_to_price(47), .47)
        self.assertAlmostEqual(spec.atr_fraction(2, 20), .1)

    def test_paths_and_registry_are_asset_isolated(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            eur, gold = asset_paths("EURUSD", root), asset_paths("XAUUSD", root)
            self.assertNotEqual(eur.market, gold.market)
            self.assertNotEqual(eur.model_registry, gold.model_registry)
            gold.model_registry.parent.mkdir(parents=True)
            gold.model_registry.write_text(json.dumps([{"asset_id": "EURUSD", "horizon_minutes": 15}]), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_asset_registry("XAUUSD", gold.model_registry)

    def test_model_identity_and_empty_state_are_symbol_aware(self):
        identity = model_id("XAUUSD", "xgboost", 15, "2026-08-31T05:00:00Z", "abcdef1234567890")
        self.assertIn("xgboost_xauusd_15m_20260831050000_abcdef123456", identity)
        self.assertEqual(empty_promotion_state("XAUUSD")["champions"]["15"]["model_status"], "NO_APPROVED_MODEL")
        self.assertEqual(approved_champions("XAUUSD"), {})

    def test_observation_identity_cannot_collide_across_symbols(self):
        eur = observation_id("EURUSD", "2026-08-31T05:00:00Z", 15, "model")
        gold = observation_id("XAUUSD", "2026-08-31T05:00:00Z", 15, "model")
        self.assertNotEqual(eur, gold)

    def test_gold_neutral_band_is_causal_and_cost_aware(self):
        spec = BrokerSymbolSpec.from_mt5("XAUUSD", info("XAUUSD"))
        close = pd.Series([4000.0, 4100.0])
        spread = pd.Series([40.0, 500.0])
        atr = pd.Series([20.0, 200.0])
        first = neutral_band_return(close, spread, atr, spec, FINAL_BAND).iloc[0]
        changed_future = neutral_band_return(close, spread.where(spread.index == 0, 99999), atr.where(atr.index == 0, 99999), spec, FINAL_BAND).iloc[0]
        self.assertEqual(first, changed_future)
        self.assertGreater(first, 0)

    def test_target_requires_exact_future_completed_bar(self):
        spec = BrokerSymbolSpec.from_mt5("XAUUSD", info("XAUUSD"))
        times = pd.date_range("2026-01-01", periods=80, freq="5min", tz="UTC")
        times = times.delete(20)
        frame = pd.DataFrame({"bar_close_utc": times, "open": 4000.0, "high": 4002.0, "low": 3998.0, "close": pd.Series(range(len(times)), dtype=float) + 4000, "spread_points": 40})
        targets, report = build_target_research(frame, spec)
        decision = pd.Timestamp("2026-01-01T01:25:00Z")
        row = targets.loc[pd.to_datetime(targets["bar_close_utc"], utc=True).eq(decision)].iloc[0]
        self.assertTrue(pd.isna(row["outcome_future_return_15m"]))
        self.assertEqual(report["target_leakage"], "PASS")

    def test_xau_runtime_never_loads_eurusd_champion(self):
        state = build_asset_state("XAUUSD", "2026-08-31T05:45:00Z")
        self.assertEqual(state["system"]["symbol"], "XAUUSD")
        self.assertTrue(all(item["model_status"] == "NO_APPROVED_MODEL" for item in state["predictions"]["horizons"].values()))
        self.assertIsNone(state["predictions"]["horizons"]["15"]["prob_up"])
        self.assertFalse(state["decision"]["trading_enabled"])


if __name__ == "__main__":
    unittest.main()
