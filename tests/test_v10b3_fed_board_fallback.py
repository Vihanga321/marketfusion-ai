from __future__ import annotations

import unittest

import pandas as pd

from src.intelligence.gold_macro_context import CONTEXT_SERIES
from src.intelligence.gold_macro_fed_board_fallback import parse_fed_board_html


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
        frame = parse_fed_board_html(html, CONTEXT_SERIES[0], "h10_summary")
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[0]["value"], 118.1)
        self.assertEqual(frame.iloc[0]["observation_date_utc"], pd.Timestamp("2026-01-02T00:00:00Z"))
        self.assertFalse(bool(frame["vintage_safe"].any()))

    def test_h15_preview_parser(self) -> None:
        html = """
        <table>
          <tr><td>H15/H15/RIFLGFCY10_N.B</td><td>Description</td><td>10 year</td></tr>
          <tr><th>Unique ID</th><th>Time Period</th><th>Value</th></tr>
          <tr><td>H15/H15/RIFLGFCY10_N.B</td><td>2026-08-27</td><td>4.67</td></tr>
          <tr><td>H15/H15/RIFLGFCY10_N.B</td><td>2026-08-28</td><td>4.73</td></tr>
        </table>
        """
        frame = parse_fed_board_html(html, CONTEXT_SERIES[2], "h15_preview")
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[-1]["value"], 4.73)
        self.assertGreater(frame.iloc[-1]["available_at_utc"], frame.iloc[-1]["observation_date_utc"])


if __name__ == "__main__":
    unittest.main()
