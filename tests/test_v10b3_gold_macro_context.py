from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.intelligence.gold_macro_context import (
    CONTEXT_SERIES,
    CONSERVATIVE_AVAILABILITY_LAG_DAYS,
    _parse_fred_csv,
    context_feature_names,
    context_store,
    join_context_asof,
    refresh_gold_macro_context,
)
from src.research.v10b3_xauusd_macro_intermarket import FEATURE_SETS, evidence_gate


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _Session:
    def get(self, url: str, **_: object) -> _Response:
        series_id = url.rsplit("=", 1)[-1]
        return _Response(
            f"DATE,{series_id}\n"
            "2026-01-02,100.0\n"
            "2026-01-05,101.0\n"
            "2026-01-06,.\n"
            "2026-01-07,102.0\n"
            "2026-01-08,103.0\n"
            "2026-01-09,104.0\n"
        )


class V10B3GoldMacroContextTests(unittest.TestCase):
    def test_fred_csv_uses_conservative_availability(self) -> None:
        spec = CONTEXT_SERIES[0]
        frame = _parse_fred_csv(
            f"DATE,{spec.series_id}\n2026-01-02,100.0\n2026-01-05,101.0\n",
            spec,
        )
        expected = pd.Timestamp("2026-01-06T23:59:00Z")
        self.assertEqual(frame.loc[0, "available_at_utc"], expected)
        self.assertEqual(CONSERVATIVE_AVAILABILITY_LAG_DAYS, 4)
        self.assertFalse(bool(frame["vintage_safe"].any()))

    def test_refresh_persists_verified_series_without_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, manifest = refresh_gold_macro_context(root, session=_Session())
            self.assertEqual(set(frame["series_id"]), {item.series_id for item in CONTEXT_SERIES})
            self.assertTrue(context_store(root).exists())
            self.assertFalse(manifest["vintage_safe"])
            self.assertFalse(manifest["production_model_eligible"])

    def test_asof_join_never_uses_future_context(self) -> None:
        rows = []
        for spec in CONTEXT_SERIES:
            rows.extend([
                {
                    "series_key": spec.key, "series_id": spec.series_id,
                    "observation_date_utc": pd.Timestamp("2026-01-01T00:00:00Z"),
                    "available_at_utc": pd.Timestamp("2026-01-05T00:00:00Z"),
                    "value": 10.0, "change_1": 0.1, "change_5": 0.5, "vintage_safe": False,
                },
                {
                    "series_key": spec.key, "series_id": spec.series_id,
                    "observation_date_utc": pd.Timestamp("2026-01-02T00:00:00Z"),
                    "available_at_utc": pd.Timestamp("2026-01-06T00:00:00Z"),
                    "value": 20.0, "change_1": 0.2, "change_5": 0.6, "vintage_safe": False,
                },
            ])
        context = pd.DataFrame(rows)
        decisions = pd.DataFrame({
            "decision_timestamp_utc": [
                pd.Timestamp("2026-01-05T12:00:00Z"),
                pd.Timestamp("2026-01-06T12:00:00Z"),
            ]
        })
        joined = join_context_asof(decisions, context)
        self.assertEqual(joined.loc[0, "ctx_yield_10y_value"], 10.0)
        self.assertEqual(joined.loc[1, "ctx_yield_10y_value"], 20.0)
        self.assertGreaterEqual(joined.loc[0, "ctx_yield_10y_age_hours"], 0.0)

    def test_macro_feature_sets_have_no_target_or_future_fields(self) -> None:
        expected = set(context_feature_names())
        self.assertTrue(expected.issubset(set(FEATURE_SETS["PRICE_CORE_PLUS_MACRO"])))
        for names in FEATURE_SETS.values():
            for name in names:
                lower = name.lower()
                self.assertFalse(lower.startswith(("target_", "outcome_", "future_")))
                self.assertNotIn("future", lower)

    def test_evidence_gate_requires_real_delta_over_price_only(self) -> None:
        baseline = {"balanced_accuracy_mean": 0.510, "cost_aware_mean": -0.001}
        weak = {
            "status": "EVALUATED", "balanced_accuracy_mean": 0.516,
            "recent_fold_balanced_accuracy": 0.515, "folds_beating_class_prior": 4,
            "log_loss_mean": 0.690, "class_prior_log_loss_mean": 0.691,
            "cost_aware_mean": -0.001,
        }
        status, failed = evidence_gate(weak, baseline)
        self.assertEqual(status, "NO_MACRO_EVIDENCE")
        self.assertIn("log_loss", failed)


if __name__ == "__main__":
    unittest.main()
