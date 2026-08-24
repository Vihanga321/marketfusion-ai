from __future__ import annotations

import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

import pandas as pd

from src.intelligence.features import build_context_snapshot, model_feature_view
from src.intelligence.macro_collectors import fetch_fred_graph, parse_ecb_csv, parse_fred_graph_csv
from src.intelligence.news_collectors import classify_text, fetch_rss, parse_gdelt_json, parse_rss_xml
from src.intelligence.quality import build_quality_report
from src.intelligence.store import merge_macro, merge_news
from src.intelligence.v05b_runner import main


CAPTURE = pd.Timestamp("2026-08-24T10:00:00Z")


class V05BContinuousIntelligenceTests(unittest.TestCase):
    def test_rss_first_observed_is_capture_not_provider_publish(self):
        xml = """<?xml version='1.0'?><rss><channel><item><guid>x1</guid><title>Federal Reserve policy update</title><link>https://example.test/a</link><pubDate>Sun, 23 Aug 2026 10:00:00 GMT</pubDate><description>Interest rates and inflation.</description></item></channel></rss>"""
        frame = parse_rss_xml(xml, "FED_MONETARY", CAPTURE)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "first_observed_utc"], CAPTURE)
        self.assertEqual(frame.loc[0, "available_from_utc"], CAPTURE)
        self.assertLess(frame.loc[0, "provider_published_utc"], CAPTURE)

    def test_gdelt_metadata_is_conservative(self):
        payload = {"articles": [{"url": "https://news.test/a", "title": "ECB discusses inflation", "seendate": "20260824T095500Z", "domain": "news.test"}]}
        frame = parse_gdelt_json(payload, CAPTURE)
        self.assertEqual(frame.loc[0, "available_from_utc"], CAPTURE)
        self.assertEqual(frame.loc[0, "topic_inflation"], 1)
        self.assertEqual(frame.loc[0, "eur_relevant"], 1)

    def test_classifier_is_deterministic_metadata_not_trade_direction(self):
        flags = classify_text("US payroll jobs and unemployment", "Federal Reserve watches labor market", "USD")
        self.assertEqual(flags["topic_labor"], 1)
        self.assertEqual(flags["usd_relevant"], 1)
        self.assertNotIn("buy", flags)
        self.assertNotIn("sell", flags)

    def test_news_store_keeps_first_seen_row(self):
        xml = """<rss><channel><item><guid>x1</guid><title>CPI release</title><link>https://example.test/a</link></item></channel></rss>"""
        first = parse_rss_xml(xml, "BLS_CPI", CAPTURE)
        later = parse_rss_xml(xml, "BLS_CPI", CAPTURE + pd.Timedelta(minutes=5))
        merged, new_rows = merge_news(first, later)
        self.assertEqual(new_rows, 0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(pd.Timestamp(merged.loc[0, "first_observed_utc"]), CAPTURE)

    def test_fred_current_endpoint_is_not_backdated(self):
        csv = "DATE,DGS2\n2026-08-21,4.00\n2026-08-24,4.10\n"
        frame = parse_fred_graph_csv(csv, "us_2y_yield", CAPTURE)
        self.assertEqual(frame.loc[0, "observation_period"], pd.Timestamp("2026-08-24T00:00:00Z"))
        self.assertEqual(frame.loc[0, "available_from_utc"], CAPTURE)

    def test_ecb_current_endpoint_is_not_backdated(self):
        csv = "TIME_PERIOD,OBS_VALUE,VALID_FROM\n2026-08-23,2.25,2026-08-23T00:00:00Z\n2026-08-24,2.25,2026-08-24T00:00:00Z\n"
        frame = parse_ecb_csv(csv, "ecb_main_refinancing_rate", CAPTURE)
        self.assertEqual(frame.loc[0, "available_from_utc"], CAPTURE)

    def test_collectors_reject_html_or_challenge_payloads(self):
        response = Mock()
        response.headers = {"Content-Type": "text/html"}
        response.content = b"<html><body>challenge</body></html>"
        response.text = response.content.decode()
        response.status_code = 200
        response.raise_for_status.return_value = None
        session = Mock()
        session.get.return_value = response
        with self.assertRaisesRegex(ValueError, "unexpected Content-Type"):
            fetch_rss("FED_MONETARY", CAPTURE, session)
        with self.assertRaisesRegex(ValueError, "unexpected Content-Type"):
            fetch_fred_graph("us_2y_yield", CAPTURE, session)

    def test_macro_revisions_are_append_only(self):
        first = parse_fred_graph_csv("DATE,DGS10\n2026-08-24,4.20\n", "us_10y_yield", CAPTURE)
        revised = parse_fred_graph_csv("DATE,DGS10\n2026-08-24,4.21\n", "us_10y_yield", CAPTURE + pd.Timedelta(minutes=5))
        merged, new_rows = merge_macro(first, revised)
        self.assertEqual(new_rows, 1)
        self.assertEqual(len(merged), 2)

    def _news(self):
        xml = """<rss><channel><item><guid>x1</guid><title>Federal Reserve inflation policy</title><link>https://example.test/a</link></item></channel></rss>"""
        return parse_rss_xml(xml, "FED_MONETARY", CAPTURE - pd.Timedelta(minutes=10))

    def _macro(self):
        frames = [
            parse_fred_graph_csv("DATE,DFF\n2026-08-23,5.00\n", "us_effective_fed_funds", CAPTURE - pd.Timedelta(minutes=30)),
            parse_fred_graph_csv("DATE,DGS2\n2026-08-23,4.00\n", "us_2y_yield", CAPTURE - pd.Timedelta(minutes=30)),
            parse_fred_graph_csv("DATE,DGS10\n2026-08-23,4.20\n", "us_10y_yield", CAPTURE - pd.Timedelta(minutes=30)),
            parse_fred_graph_csv("DATE,T10Y2Y\n2026-08-23,0.20\n", "us_10y_minus_2y", CAPTURE - pd.Timedelta(minutes=30)),
            parse_ecb_csv("TIME_PERIOD,OBS_VALUE\n2026-08-23,2.25\n", "ecb_main_refinancing_rate", CAPTURE - pd.Timedelta(minutes=30)),
        ]
        return pd.concat(frames, ignore_index=True)

    def test_context_uses_only_first_observed_before_decision(self):
        news = self._news()
        future = news.copy()
        future["news_id"] = "future"
        future["first_observed_utc"] = CAPTURE + pd.Timedelta(minutes=1)
        future["available_from_utc"] = future["first_observed_utc"]
        context = build_context_snapshot(pd.concat([news, future], ignore_index=True), self._macro(), CAPTURE)
        self.assertEqual(int(context.loc[0, "news_15m_count"]), 1)
        self.assertAlmostEqual(float(context.loc[0, "macro_policy_rate_spread_us_minus_ecb_pctpt"]), 2.75)

    def test_model_view_rejects_outcomes(self):
        context = build_context_snapshot(self._news(), self._macro(), CAPTURE)
        self.assertFalse(any(c.startswith("outcome_") for c in model_feature_view(context).columns))
        context["outcome_future"] = 1
        with self.assertRaises(ValueError):
            model_feature_view(context)

    def test_quality_passes_causal_synthetic_data(self):
        news = self._news()
        macro = self._macro()
        context = build_context_snapshot(news, macro, CAPTURE)
        result = build_quality_report(news, macro, context, CAPTURE)
        self.assertTrue(result.passed, result.report)

    def test_continuous_runner_ctrl_c_exits_cleanly(self):
        output = StringIO()
        with (
            patch("src.intelligence.v05b_runner.arguments", return_value=Namespace(once=False, interval=60)),
            patch("src.intelligence.v05b_runner.run_cycle", side_effect=KeyboardInterrupt),
            redirect_stdout(output),
        ):
            main()
        self.assertIn("V0.5B stopped by user. Stored history remains intact.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
