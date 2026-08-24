from __future__ import annotations

import unittest

import pandas as pd

from src.events.mt5_calendar_compatibility_audit import audit


class V04DMt5CompatibilityTests(unittest.TestCase):
    def test_cpi_rounding_compatibility_does_not_make_surprise_eligible(self):
        frame = pd.DataFrame([
            {"measure_id": "headline_cpi_yoy_sa", "reference_month": "2016-02", "mt5_actual_value": 1.0, "v04c_actual_value": 0.9735998130960155, "actual_difference": 0.02640018690398449},
            {"measure_id": "core_cpi_yoy_sa", "reference_month": "2016-02", "mt5_actual_value": 2.3, "v04c_actual_value": 2.269, "actual_difference": 0.031},
            {"measure_id": "unemployment_rate", "reference_month": "2016-02", "mt5_actual_value": 4.9, "v04c_actual_value": 4.9, "actual_difference": 0.0},
            {"measure_id": "nonfarm_payroll_change", "reference_month": "2016-02", "mt5_actual_value": 242.0, "v04c_actual_value": 242.0, "actual_difference": 0.0},
        ])
        summary, mismatches = audit(frame)
        cpi = summary.loc[summary["measure_id"].eq("headline_cpi_yoy_sa")].iloc[0]
        self.assertEqual(cpi["compatibility_status"], "ROUNDING_COMPATIBLE_DIAGNOSTIC")
        self.assertFalse(bool(cpi["cross_source_historical_surprise_eligible"]))
        self.assertFalse(bool(cpi["historical_mt5_forecast_pit_verified"]))
        self.assertEqual(len(mismatches), 0)

    def test_nfp_mismatch_is_quarantined(self):
        frame = pd.DataFrame([
            {"measure_id": "headline_cpi_yoy_sa", "reference_month": "2015-01", "mt5_actual_value": 1.0, "v04c_actual_value": 1.02, "actual_difference": -0.02},
            {"measure_id": "core_cpi_yoy_sa", "reference_month": "2015-01", "mt5_actual_value": 1.6, "v04c_actual_value": 1.61, "actual_difference": -0.01},
            {"measure_id": "unemployment_rate", "reference_month": "2015-01", "mt5_actual_value": 5.7, "v04c_actual_value": 5.7, "actual_difference": 0.0},
            {"measure_id": "nonfarm_payroll_change", "reference_month": "2015-01", "mt5_actual_value": 201.0, "v04c_actual_value": 257.0, "actual_difference": -56.0},
        ])
        summary, mismatches = audit(frame)
        nfp = summary.loc[summary["measure_id"].eq("nonfarm_payroll_change")].iloc[0]
        self.assertEqual(nfp["compatibility_status"], "COMPATIBILITY_REVIEW_REQUIRED")
        self.assertEqual(len(mismatches.loc[mismatches["measure_id"].eq("nonfarm_payroll_change")]), 1)


if __name__ == "__main__":
    unittest.main()
