from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.fusion.v06b_engine import build_advisory, fuse_probabilities, trade_window
from src.fusion.v06b_risk import classify_event_risk, classify_market_regime, classify_session, classify_spread, freshness


NOW = pd.Timestamp("2026-08-24T12:00:00Z")


def market_frame(count: int = 30, complete: bool = True) -> pd.DataFrame:
    times = pd.date_range(end=NOW - pd.Timedelta(minutes=5), periods=count, freq="5min")
    values = np.linspace(0.00005, 0.00025, count)
    return pd.DataFrame({
        "decision_timestamp_utc": times, "m5_close": 1.17,
        "m5_volatility_60m": values, "m5_return_15m": 0.0002,
        "m5_return_60m": 0.0003, "m5_return_240m": 0.0004,
        "m5_spread_points": 1.0, "m5_spread_median_60m": 1.0,
        "feature_complete": complete, "is_asia_session": 0,
        "is_london_session": 1, "is_new_york_session": 1,
        "is_london_ny_overlap": 1,
    })


def events(minutes: float = 300, code: str = "nonfarm-payrolls") -> pd.DataFrame:
    return pd.DataFrame([{
        "event_id": "840030016", "event_timestamp_utc": NOW + pd.Timedelta(minutes=minutes),
        "event_name": "Nonfarm Payrolls" if code == "nonfarm-payrolls" else "Gross Domestic Product", "event_code": code, "captured_at_gmt": NOW,
    }])


def horizon(prob=(0.15, 0.20, 0.65), approved=True):
    return {"model_status": "APPROVED_CHAMPION" if approved else "NO_APPROVED_MODEL", "model_id": "m", "prob_down": prob[0], "prob_neutral": prob[1], "prob_up": prob[2]}


def shadow(items=None):
    items = items or {"15": horizon(approved=False), "60": horizon(approved=False), "240": horizon(approved=False)}
    return {"status": "PASS_SHADOW_RUNNING", "captured_at_utc": NOW.isoformat(), "horizons": items}


class V06BFusionTests(unittest.TestCase):
    def test_01_session_overlap_precedence(self):
        self.assertEqual(classify_session(market_frame().iloc[-1]), "OVERLAP")

    def test_02_session_asia(self):
        self.assertEqual(classify_session({"is_asia_session": 1}), "ASIA")

    def test_03_session_london(self):
        self.assertEqual(classify_session({"is_london_session": 1}), "LONDON")

    def test_04_session_new_york(self):
        self.assertEqual(classify_session({"is_new_york_session": 1}), "NEW_YORK")

    def test_05_session_off(self):
        self.assertEqual(classify_session({}), "OFF")

    def test_06_regime_uses_only_prior_rows(self):
        frame = market_frame()
        result = classify_market_regime(frame, frame.iloc[-1])
        self.assertEqual(result["history_rows"], len(frame) - 1)

    def test_07_regime_insufficient_history(self):
        frame = market_frame(10)
        self.assertEqual(classify_market_regime(frame, frame.iloc[-1])["volatility_regime"], "UNAVAILABLE")

    def test_08_regime_extreme(self):
        frame = market_frame(); frame.loc[frame.index[-1], "m5_volatility_60m"] = 1.0
        self.assertEqual(classify_market_regime(frame, frame.iloc[-1])["volatility_regime"], "EXTREME")

    def test_09_spread_normal(self):
        self.assertEqual(classify_spread(pd.Series({"m5_spread_points": 1, "m5_spread_median_60m": 1}))["status"], "NORMAL")

    def test_10_spread_wide(self):
        self.assertEqual(classify_spread(pd.Series({"m5_spread_points": 2, "m5_spread_median_60m": 1}))["status"], "WIDE")

    def test_11_spread_extreme(self):
        self.assertEqual(classify_spread(pd.Series({"m5_spread_points": 3, "m5_spread_median_60m": 1}))["status"], "EXTREME")

    def test_12_spread_invalid(self):
        self.assertEqual(classify_spread(pd.Series({"m5_spread_points": np.nan, "m5_spread_median_60m": 1}))["status"], "UNAVAILABLE")

    def test_13_event_none_far(self):
        self.assertEqual(classify_event_risk(events(300), NOW)["status"], "NONE")

    def test_14_event_watch(self):
        self.assertEqual(classify_event_risk(events(120), NOW)["status"], "WATCH")

    def test_15_event_elevated(self):
        self.assertEqual(classify_event_risk(events(30), NOW)["status"], "ELEVATED")

    def test_16_event_block_pre_release(self):
        self.assertEqual(classify_event_risk(events(10), NOW)["status"], "BLOCK")

    def test_17_event_block_post_release(self):
        self.assertEqual(classify_event_risk(events(-10), NOW)["status"], "BLOCK")

    def test_18_event_post_release_volatility(self):
        self.assertEqual(classify_event_risk(events(-30), NOW)["status"], "POST_RELEASE_VOLATILITY")

    def test_19_unknown_event_not_inferred(self):
        self.assertEqual(classify_event_risk(events(10, "gdp"), NOW)["data_status"], "DEGRADED")

    def test_20_missing_event_data_cautious(self):
        self.assertTrue(classify_event_risk(None, NOW)["blocks_direction"])

    def test_21_freshness_fresh(self):
        self.assertEqual(freshness(NOW - pd.Timedelta(minutes=1), NOW, 5)["status"], "FRESH")

    def test_22_freshness_stale(self):
        self.assertEqual(freshness(NOW - pd.Timedelta(minutes=6), NOW, 5)["status"], "STALE")

    def test_23_freshness_unavailable(self):
        self.assertEqual(freshness(None, NOW, 5)["status"], "UNAVAILABLE")

    def test_24_no_model_is_very_low_wait(self):
        result = fuse_probabilities({"15": horizon(approved=False), "60": horizon(approved=False), "240": horizon(approved=False)})
        self.assertEqual((result["direction"], result["confidence"]), ("WAIT", "VERY_LOW"))

    def test_25_two_horizon_buy_consensus(self):
        result = fuse_probabilities({"15": horizon(), "60": horizon(), "240": horizon(approved=False)})
        self.assertEqual(result["direction"], "UP")

    def test_26_horizon_weights_are_normalized(self):
        result = fuse_probabilities({"15": horizon(), "60": horizon(), "240": horizon(approved=False)})
        self.assertAlmostEqual(sum(result["normalized_weights"].values()), 1.0)

    def test_27_one_horizon_is_insufficient(self):
        result = fuse_probabilities({"15": horizon(), "60": horizon(approved=False), "240": horizon(approved=False)})
        self.assertEqual(result["status"], "INSUFFICIENT_APPROVED_HORIZONS")

    def test_28_disagreement_waits(self):
        result = fuse_probabilities({"15": horizon(), "60": horizon((0.65, 0.2, 0.15)), "240": horizon(approved=False)})
        self.assertEqual(result["status"], "HORIZON_DISAGREEMENT")

    def test_29_neutral_waits(self):
        result = fuse_probabilities({"15": horizon((0.1, .8, .1)), "60": horizon((.1, .8, .1)), "240": horizon(approved=False)})
        self.assertEqual(result["direction"], "WAIT")

    def test_30_malformed_probabilities_fail_closed(self):
        bad = horizon((np.nan, .2, .8))
        result = fuse_probabilities({"15": bad, "60": horizon(), "240": horizon(approved=False)})
        self.assertIn(15, result["malformed_horizons"])

    def test_31_no_model_advisory_exact_gate(self):
        frame = market_frame()
        payload = build_advisory(frame, shadow(), events(), {}, {"captured_at_utc": NOW.isoformat()}, NOW)
        self.assertEqual((payload["action"], payload["confidence"], payload["decision_gate"]), ("WAIT", "VERY_LOW", "WAIT_NO_MODEL"))

    def test_32_intelligence_never_overrides_direction(self):
        frame = market_frame()
        payload = build_advisory(frame, shadow({"15": horizon(), "60": horizon(), "240": horizon()}), events(), {"news_15m_count": 99}, {"captured_at_utc": NOW.isoformat()}, NOW)
        self.assertFalse(payload["intelligence"]["directional_override"])

    def test_33_wait_has_no_trade_window(self):
        self.assertEqual(trade_window(NOW, "WAIT", [], classify_event_risk(events(), NOW))["status"], "NOT_APPLICABLE_WAIT")

    def test_34_trade_window_starts_next_m5(self):
        window = trade_window(NOW + pd.Timedelta(minutes=2), "UP", [15, 60], classify_event_risk(events(), NOW))
        self.assertEqual(pd.Timestamp(window["start_utc"]), pd.Timestamp("2026-08-24T12:05:00Z"))

    def test_35_event_truncates_trade_window(self):
        event = classify_event_risk(events(25), NOW)
        window = trade_window(NOW, "UP", [60], event)
        self.assertEqual(window["status"], "TOO_SHORT_WAIT")

    def test_36_reassessment_always_present(self):
        payload = build_advisory(market_frame(), shadow(), events(), {}, {"captured_at_utc": NOW.isoformat()}, NOW)
        self.assertIsNotNone(payload["next_reassessment_utc"])

    def test_37_reasons_are_priority_sorted(self):
        payload = build_advisory(market_frame(30, False), shadow(), None, {}, {}, NOW)
        priorities = [item["priority"] for item in payload["reasons"]]
        self.assertEqual(priorities, sorted(priorities, reverse=True))

    def test_38_manual_confirmation_and_no_trading(self):
        payload = build_advisory(market_frame(), shadow(), events(), {}, {"captured_at_utc": NOW.isoformat()}, NOW)
        self.assertTrue(payload["manual_confirmation_required"]); self.assertFalse(payload["trading_enabled"])

    def test_39_event_boundary_240_is_watch(self):
        self.assertEqual(classify_event_risk(events(240), NOW)["status"], "WATCH")

    def test_40_event_boundary_60_is_elevated(self):
        self.assertEqual(classify_event_risk(events(60), NOW)["status"], "ELEVATED")

    def test_41_event_boundary_minus_60_is_post_release(self):
        self.assertEqual(classify_event_risk(events(-60), NOW)["status"], "POST_RELEASE_VOLATILITY")

    def test_42_elevated_event_blocks_direction(self):
        self.assertTrue(classify_event_risk(events(30), NOW)["blocks_direction"])

    def test_43_stale_inference_is_a_blocker(self):
        item = shadow(); item["captured_at_utc"] = (NOW - pd.Timedelta(minutes=10)).isoformat()
        payload = build_advisory(market_frame(), item, events(), {}, {"captured_at_utc": NOW.isoformat()}, NOW)
        self.assertIn("STALE_MODEL_DATA", payload["blocking_reason_codes"])

    def test_44_stale_model_artifact_is_a_blocker(self):
        items = {"15": horizon(), "60": horizon(), "240": horizon()}
        for item in items.values():
            item["model_created_at_utc"] = (NOW - pd.Timedelta(days=46)).isoformat()
        payload = build_advisory(market_frame(), shadow(items), events(), {}, {"captured_at_utc": NOW.isoformat()}, NOW)
        self.assertIn("STALE_MODEL_DATA", payload["blocking_reason_codes"])

    def test_45_stale_causal_intelligence_is_a_blocker(self):
        context = {"decision_timestamp_utc": NOW - pd.Timedelta(minutes=31)}
        payload = build_advisory(market_frame(), shadow(), events(), context, {"captured_at_utc": NOW.isoformat()}, NOW)
        self.assertIn("INTELLIGENCE_STALE", payload["blocking_reason_codes"])

    def test_46_action_and_confidence_meanings_are_explicit(self):
        payload = build_advisory(market_frame(), shadow(), events(), {}, {"captured_at_utc": NOW.isoformat()}, NOW)
        self.assertIn("No directional advisory", payload["action_meaning"])
        self.assertIn("No approved model", payload["confidence_meaning"])


if __name__ == "__main__":
    unittest.main()
