from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.events.consensus_measure_mapping import mapping_for
from src.events.consensus_source_registry import ApprovalStatus, source_for
from src.events.v04d_consensus import (
    ConsensusAuditError,
    add_prior_only_robust_normalization,
    add_surprise_fields,
    freeze_or_verify_fingerprints,
    map_trading_economics_row,
    parse_provider_forecast,
    provider_approval_status,
    select_latest_verified_pre_release_snapshot,
)


EVENT_TIME = pd.Timestamp("2024-01-05T13:30:00Z")
REFERENCE = pd.Timestamp("2023-12-01T00:00:00Z")


def provider_row(**overrides):
    row = {
        "CalendarId": "123",
        "Date": "2024-01-05T13:30:00",
        "Country": "United States",
        "Category": "Non Farm Payrolls",
        "Event": "Non Farm Payrolls",
        "Reference": "Dec/23",
        "ReferenceDate": "2023-12-01T00:00:00",
        "Forecast": "175K",
        "ForecastValue": 175000,
        "TEForecast": "180K",
        "TEForecastValue": 180000,
        "LastUpdate": "2024-01-05T13:31:00",
        "Unit": "K",
        "DateSpan": 0,
        "Ticker": "NFP",
        "Symbol": "NFP",
    }
    row.update(overrides)
    return row


def map_row(**kwargs):
    return map_trading_economics_row(
        provider_row(**kwargs.pop("provider_overrides", {})),
        event_id="e1",
        event_type="us_employment_situation",
        event_timestamp_utc=EVENT_TIME,
        reference_period=REFERENCE,
        measure_id="nonfarm_payroll_change",
        canonical_actual_value=216.0,
        mapping=mapping_for("nonfarm_payroll_change"),
        **kwargs,
    )


class V04DConsensusTests(unittest.TestCase):
    def test_trading_economics_not_preapproved(self):
        source = source_for("Trading Economics")
        self.assertFalse(source.point_in_time_safe)
        self.assertEqual(source.approval_status, ApprovalStatus.CONDITIONALLY_APPROVED)

    def test_teforecast_is_not_consensus_input(self):
        mapping = mapping_for("nonfarm_payroll_change")
        value = parse_provider_forecast("175K", 175000, mapping)
        self.assertEqual(value, 175.0)
        row = map_row(provider_overrides={"TEForecast": "999K", "TEForecastValue": 999000})
        self.assertEqual(row["consensus_value"], 175.0)

    def test_forecastvalue_payroll_scale(self):
        mapping = mapping_for("nonfarm_payroll_change")
        self.assertEqual(parse_provider_forecast("", 175000, mapping), 175.0)

    def test_consensus_without_snapshot_timestamp_is_unverified(self):
        row = map_row()
        self.assertEqual(row["consensus_status"], "POINT_IN_TIME_UNVERIFIED")
        self.assertFalse(row["point_in_time_verified"])
        self.assertTrue(pd.isna(row["consensus_available_from_utc"]))

    def test_lastupdate_does_not_prove_availability(self):
        row = map_row(provider_overrides={"LastUpdate": "2024-01-05T12:00:00"})
        self.assertEqual(row["consensus_status"], "POINT_IN_TIME_UNVERIFIED")
        self.assertFalse(row["point_in_time_verified"])

    def test_verified_snapshot_before_t_is_accepted(self):
        row = map_row(
            consensus_snapshot_timestamp_utc="2024-01-05T12:30:00Z",
            snapshot_timestamp_verified=True,
        )
        self.assertEqual(row["consensus_status"], "COMPLETE")
        self.assertTrue(row["point_in_time_verified"])

    def test_snapshot_exactly_at_t_rejected(self):
        row = map_row(
            consensus_snapshot_timestamp_utc=EVENT_TIME,
            snapshot_timestamp_verified=True,
        )
        self.assertEqual(row["consensus_status"], "POST_RELEASE_ONLY")
        self.assertFalse(row["point_in_time_verified"])

    def test_snapshot_after_t_rejected(self):
        row = map_row(
            consensus_snapshot_timestamp_utc="2024-01-05T13:31:00Z",
            snapshot_timestamp_verified=True,
        )
        self.assertEqual(row["consensus_status"], "POST_RELEASE_ONLY")

    def test_event_timestamp_mismatch_rejected(self):
        row = map_row(provider_overrides={"Date": "2024-01-05T13:31:00"})
        self.assertEqual(row["consensus_status"], "EVENT_MAPPING_AMBIGUOUS")

    def test_unit_mismatch_rejected(self):
        row = map_row(provider_overrides={"Unit": "%"})
        self.assertEqual(row["consensus_status"], "UNIT_MISMATCH")

    def test_reference_period_mismatch_rejected(self):
        row = map_row(provider_overrides={"ReferenceDate": "2023-11-01T00:00:00"})
        self.assertEqual(row["consensus_status"], "REFERENCE_PERIOD_MISMATCH")

    def test_missing_consensus_stays_null(self):
        row = map_row(provider_overrides={"Forecast": "", "ForecastValue": None})
        self.assertEqual(row["consensus_status"], "UNAVAILABLE")
        self.assertIsNone(row["consensus_value"])

    def test_latest_verified_snapshot_is_deterministic(self):
        base = map_row(
            consensus_snapshot_timestamp_utc="2024-01-05T10:00:00Z",
            snapshot_timestamp_verified=True,
        )
        newer = dict(base)
        newer["consensus_available_from_utc"] = pd.Timestamp("2024-01-05T12:00:00Z")
        newer["consensus_snapshot_timestamp_utc"] = newer["consensus_available_from_utc"]
        newer["consensus_value"] = 180.0
        selected = select_latest_verified_pre_release_snapshot(pd.DataFrame([newer, base]))
        self.assertEqual(len(selected), 1)
        self.assertEqual(float(selected.iloc[0]["consensus_value"]), 180.0)

    def test_unverified_snapshot_never_selected(self):
        row = map_row()
        selected = select_latest_verified_pre_release_snapshot(pd.DataFrame([row]))
        self.assertTrue(selected.empty)

    def test_raw_surprise_arithmetic(self):
        row = map_row(
            consensus_snapshot_timestamp_utc="2024-01-05T12:30:00Z",
            snapshot_timestamp_verified=True,
        )
        result = add_surprise_fields(pd.DataFrame([row]))
        self.assertAlmostEqual(float(result.iloc[0]["surprise_signed_raw"]), 41.0)

    def test_unverified_consensus_cannot_create_surprise(self):
        result = add_surprise_fields(pd.DataFrame([map_row()]))
        self.assertTrue(np.isnan(result.iloc[0]["surprise_signed_raw"]))

    def test_prior_only_normalization_ignores_future(self):
        rows = []
        for i in range(12):
            rows.append({
                "event_id": f"e{i}",
                "event_timestamp_utc": pd.Timestamp("2020-01-01T00:00:00Z") + pd.DateOffset(months=i),
                "measure_id": "x",
                "surprise_signed_raw": float(i),
            })
        frame = pd.DataFrame(rows)
        first = add_prior_only_robust_normalization(frame, minimum_history=3)
        changed = frame.copy()
        changed.loc[11, "surprise_signed_raw"] = 9999.0
        second = add_prior_only_robust_normalization(changed, minimum_history=3)
        pd.testing.assert_series_equal(
            first.loc[:10, "surprise_z_robust"],
            second.loc[:10, "surprise_z_robust"],
            check_names=False,
        )

    def test_fingerprint_mutation_hard_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            manifest = root / "fingerprint.json"
            source.write_bytes(b"abc")
            freeze_or_verify_fingerprints([source], manifest, root=root, initialize=True)
            freeze_or_verify_fingerprints([source], manifest, root=root)
            source.write_bytes(b"abcd")
            with self.assertRaises(ConsensusAuditError):
                freeze_or_verify_fingerprints([source], manifest, root=root)

    def test_missing_fingerprint_requires_explicit_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            source.write_bytes(b"abc")
            with self.assertRaises(ConsensusAuditError):
                freeze_or_verify_fingerprints([source], root / "missing.json", root=root)

    def test_provider_approval_requires_verified_rows(self):
        unverified = pd.DataFrame([map_row()])
        status, _ = provider_approval_status(unverified)
        self.assertEqual(status, "INSUFFICIENT_METADATA")
        verified = pd.DataFrame([map_row(
            consensus_snapshot_timestamp_utc="2024-01-05T12:30:00Z",
            snapshot_timestamp_verified=True,
        )])
        status, _ = provider_approval_status(verified)
        self.assertEqual(status, "APPROVED_POINT_IN_TIME")


if __name__ == "__main__":
    unittest.main()
