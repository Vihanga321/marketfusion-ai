from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import requests

from src.intelligence.gold_macro_context import (
    CONTEXT_SERIES,
    CONSERVATIVE_AVAILABILITY_LAG_DAYS,
    FRED_MAX_ATTEMPTS,
    _parse_fred_csv,
    context_feature_names,
    context_store,
    fetch_fred_series,
    join_context_asof,
    refresh_gold_macro_context,
    series_cache_path,
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


class _FlakySession(_Session):
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls += 1
        if self.calls <= self.failures:
            raise requests.exceptions.ReadTimeout("simulated FRED timeout")
        return super().get(url, **kwargs)


class _SelectiveFailureSession(_Session):
    def __init__(self, fail_series: set[str]) -> None:
        self.fail_series = fail_series

    def get(self, url: str, **kwargs: object) -> _Response:
        series_id = url.rsplit("=", 1)[-1]
        if series_id in self.fail_series:
            raise requests.exceptions.ReadTimeout(f"simulated timeout {series_id}")
        return super().get(url, **kwargs)


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

    def test_fetch_retries_transient_timeout(self) -> None:
        session = _FlakySession(failures=1)
        with patch("src.intelligence.gold_macro_context.time.sleep", return_value=None):
            frame, metadata = fetch_fred_series(CONTEXT_SERIES[0], session=session)
        self.assertFalse(frame.empty)
        self.assertEqual(session.calls, 2)
        self.assertEqual(metadata["download_attempts"], 2)
        self.assertEqual(metadata["source_mode"], "NETWORK_REFRESH")

    def test_fetch_fails_after_bounded_attempts(self) -> None:
        session = _FlakySession(failures=100)
        with patch("src.intelligence.gold_macro_context.time.sleep", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "failed after"):
                fetch_fred_series(CONTEXT_SERIES[0], session=session)
        self.assertEqual(session.calls, FRED_MAX_ATTEMPTS)

    def test_refresh_persists_verified_series_without_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame, manifest = refresh_gold_macro_context(root, session=_Session())
            self.assertEqual(set(frame["series_id"]), {item.series_id for item in CONTEXT_SERIES})
            self.assertTrue(context_store(root).exists())
            self.assertTrue(all(series_cache_path(item, root).exists() for item in CONTEXT_SERIES))
            self.assertFalse(manifest["vintage_safe"])
            self.assertFalse(manifest["production_model_eligible"])
            self.assertEqual(manifest["network_refresh_count"], len(CONTEXT_SERIES))
            self.assertEqual(manifest["cache_fallback_count"], 0)

    def test_refresh_uses_last_good_series_cache_on_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refresh_gold_macro_context(root, session=_Session())
            failing = {CONTEXT_SERIES[1].series_id}
            with patch("src.intelligence.gold_macro_context.time.sleep", return_value=None):
                frame, manifest = refresh_gold_macro_context(
                    root, session=_SelectiveFailureSession(failing)
                )
            self.assertEqual(set(frame["series_id"]), {item.series_id for item in CONTEXT_SERIES})
            self.assertEqual(manifest["cache_fallback_count"], 1)
            cached = next(item for item in manifest["series"] if item["series_id"] in failing)
            self.assertEqual(cached["source_mode"], "CACHE_FALLBACK")
            self.assertIn("timeout", cached["last_refresh_error"].lower())

    def test_incomplete_first_refresh_preserves_successful_series_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed_id = CONTEXT_SERIES[-1].series_id
            with patch("src.intelligence.gold_macro_context.time.sleep", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "Successful series were cached for resume"):
                    refresh_gold_macro_context(
                        root, session=_SelectiveFailureSession({failed_id})
                    )
            self.assertFalse(context_store(root).exists())
            for spec in CONTEXT_SERIES[:-1]:
                self.assertTrue(series_cache_path(spec, root).exists())
            self.assertFalse(series_cache_path(CONTEXT_SERIES[-1], root).exists())

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
