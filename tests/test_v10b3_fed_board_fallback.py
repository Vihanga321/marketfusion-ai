from __future__ import annotations

import unittest

import pandas as pd

from src.intelligence.gold_macro_context import CONTEXT_SERIES
from src.intelligence.gold_macro_fed_board_fallback import parse_fed_board_payload


class V10B3FedBoardFallbackTests(unittest.TestCase):
    def test_h10_summary_parser(self) -> None:
        html = """
        <table>
          <tr><th>Date</th><th>Rate</th></tr>
          <tr><td>2-JAN-26</td><td>118.1000</td></tr>
          <tr><td>5-JAN-26</td><td>ND</td></tr>
          <tr><td>6-JAN-26</td><td>118.3000</td></tr>
        </table>
        """
        frame = parse_fed_board_payload(
            html, CONTEXT_SERIES[0], "h10_summary", "JRXWTFB_N.B"
        )
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[0]["value"], 118.1)
        self.assertEqual(
            frame.iloc[0]["observation_date_utc"],
            pd.Timestamp("2026-01-02T00:00:00Z"),
        )
        self.assertFalse(bool(frame["vintage_safe"].any()))

    def test_h15_full_csv_parser_contract(self) -> None:
        csv_text = """H.15 Selected Interest Rates
Series Description,10-year Treasury
Unit,Percent
Multiplier,1
Currency,NA
Time Period,RIFLGFCY10_N.B
2026-08-27,4.67
2026-08-28,4.73
"""
        frame = parse_fed_board_payload(
            csv_text, CONTEXT_SERIES[2], "h15_ddp_csv", "RIFLGFCY10_N.B"
        )
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[-1]["value"], 4.73)
        self.assertGreater(
            frame.iloc[-1]["available_at_utc"],
            frame.iloc[-1]["observation_date_utc"],
        )


if __name__ == "__main__":
    unittest.main()
