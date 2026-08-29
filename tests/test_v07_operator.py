from __future__ import annotations

import unittest

from src.dashboard.v07_operator import build_operator_status, classify_data_freshness, forex_session_state, market_calendar


def state(gate: str = "WAIT_STALE_MARKET_DATA") -> dict[str, object]:
    return {
        "system": {"status": "FAIL_CLOSED", "mode": "SHADOW_ADVISORY_ONLY"},
        "market": {"freshness": {"observed_at_utc": "2026-08-28T21:00:00Z"}},
        "decision": {
            "action": "WAIT", "gate": gate, "trading_enabled": False,
            "manual_confirmation_required": True,
            "next_reassessment": {"utc": "2026-08-29T12:05:00Z"},
        },
        "trade_window": {"status": "NOT_APPLICABLE_WAIT", "start_utc": None, "end_utc": None},
        "reasons": [{"message": "Latest completed market feature row is stale."}],
    }


def v05a(feature_time: str = "2026-08-28T21:00:00Z") -> dict[str, object]:
    return {
        "status": "PASS_RUNNING", "last_dataset_decision_utc": feature_time,
        "timeframes": {
            "M5": {"last_bar_close_utc": feature_time},
            "M15": {"last_bar_close_utc": feature_time},
        },
    }


class MarketCalendarTests(unittest.TestCase):
    def test_weekend_is_closed_with_real_next_open(self):
        value = market_calendar("2026-08-29T12:00:00Z")
        self.assertEqual(value["status"], "CLOSED_WEEKEND")
        self.assertFalse(value["market_open"])
        self.assertEqual(value["next_market_open_utc"], "2026-08-30T21:00:00+00:00")

    def test_weekday_market_is_open(self):
        value = market_calendar("2026-08-31T12:00:00Z")
        self.assertEqual(value["status"], "OPEN")
        self.assertTrue(value["market_open"])

    def test_opening_soon_uses_new_york_dst(self):
        value = market_calendar("2026-08-30T20:30:00Z")
        self.assertEqual(value["status"], "OPENING_SOON")
        self.assertEqual(value["next_market_open_utc"], "2026-08-30T21:00:00+00:00")


class SessionTests(unittest.TestCase):
    def test_summer_london_new_york_overlap(self):
        value = forex_session_state("2026-07-01T13:00:00Z")
        self.assertEqual(value["active_sessions"], ["LONDON", "NEW_YORK"])
        self.assertTrue(value["overlap"])

    def test_winter_london_new_york_overlap_respects_dst(self):
        value = forex_session_state("2026-01-15T14:00:00Z")
        self.assertEqual(value["active_sessions"], ["LONDON", "NEW_YORK"])

    def test_weekend_session_has_next_forex_open(self):
        value = forex_session_state("2026-08-29T12:00:00Z")
        self.assertEqual(value["current_session"], "WEEKEND")
        self.assertEqual(value["next_session"], "SYDNEY")


class OperatorStatusTests(unittest.TestCase):
    def test_freshness_categories_and_missing_timestamp(self):
        self.assertEqual(classify_data_freshness(30), "LIVE")
        self.assertEqual(classify_data_freshness(300), "FRESH")
        self.assertEqual(classify_data_freshness(600), "DELAYED")
        self.assertEqual(classify_data_freshness(901), "STALE")
        self.assertEqual(classify_data_freshness(None), "UNAVAILABLE")

    def test_weekend_stale_data_is_context_not_system_failure(self):
        value = build_operator_status(state(), v05a(), "2026-08-29T12:00:00Z", "2026-08-28T20:59:58Z")
        self.assertEqual(value["system_health"]["status"], "PASS")
        self.assertEqual(value["trading_state"], "MARKET_CLOSED")
        self.assertEqual(value["data_freshness"]["status"], "STALE")
        self.assertEqual(value["data_freshness"]["reason"], "STALE_MARKET_CLOSED")
        self.assertFalse(value["trading_window"]["trading_enabled"])

    def test_open_market_without_model_waits_for_model(self):
        value = build_operator_status(state("WAIT_NO_MODEL"), v05a("2026-08-31T11:58:00Z"), "2026-08-31T12:00:00Z", "2026-08-31T11:59:58Z")
        self.assertEqual(value["market"]["status"], "OPEN")
        self.assertEqual(value["trading_window"]["status"], "WAITING_FOR_MODEL")
        self.assertEqual(value["trading_state"], "WAIT")

    def test_missing_timestamps_remain_unavailable(self):
        runtime_state = state()
        runtime_state["market"]["freshness"].pop("observed_at_utc")
        value = build_operator_status(runtime_state, {"status": "PASS_RUNNING", "timeframes": {}}, "2026-08-31T12:00:00Z", None)
        self.assertEqual(value["data_freshness"]["status"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
