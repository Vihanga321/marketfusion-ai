from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.engines.contract import ENGINE_CONTRACT_VERSION
from src.engines.engine_layer import engine_status
from src.engines.liquidity import liquidity_analysis
from src.engines.patterns import PATTERN_TYPES, detect_patterns
from src.engines.price_action import breakout_events, candle_events
from src.engines.research import ABLATION_GROUPS, evaluate_pattern_records
from src.engines.structure import classify_swings, structure_events, structure_state, support_resistance
from src.engines.swings import confirmed_swings
from src.marketdata.v05a_contract import BAR_COLUMNS, TIMEFRAMES


def bars_from_ohlc(values: list[tuple[float, float, float, float]], start: str = "2026-08-24T00:00:00Z", minutes: int = 5) -> pd.DataFrame:
    opens = pd.date_range(start, periods=len(values), freq=f"{minutes}min")
    frame = pd.DataFrame(values, columns=["open", "high", "low", "close"])
    frame["bar_open_utc"] = opens
    frame["bar_close_utc"] = opens + pd.Timedelta(minutes=minutes)
    frame["tick_volume"] = 100
    frame["spread_points"] = 1
    frame["real_volume"] = 0
    frame["first_observed_utc"] = frame["bar_close_utc"]
    frame["source"] = "TEST"
    return frame[BAR_COLUMNS]


def flat_bars(count: int = 80) -> pd.DataFrame:
    return bars_from_ohlc([(100.0, 100.1, 99.9, 100.0)] * count)


def swings(prices: list[tuple[str, float]], start_index: int = 2) -> list[dict[str, object]]:
    times = pd.date_range("2026-08-24T00:00:00Z", periods=100, freq="5min")
    result = []
    for offset, (kind, price) in enumerate(prices):
        index = start_index + offset * 3
        result.append({"swing_id": f"s{offset}-{kind}", "timeframe": "M5", "kind": kind, "pivot_time_utc": times[index].isoformat(), "confirmed_at_utc": times[index + 2].isoformat(), "price": price, "left_bars": 2, "right_bars": 2})
    return result


class SwingCausalityTests(unittest.TestCase):
    def test_pivot_is_unavailable_until_right_side_close(self):
        frame = bars_from_ohlc([(10, 11, 9, 10), (10, 12, 9.5, 11), (11, 15, 10, 12), (12, 13, 10.5, 11), (11, 12, 9.8, 10)])
        before = confirmed_swings(frame, "M5", as_of=frame.iloc[3]["bar_close_utc"], left_bars=2, right_bars=2)
        after = confirmed_swings(frame, "M5", as_of=frame.iloc[4]["bar_close_utc"], left_bars=2, right_bars=2)
        self.assertFalse(any(item["price"] == 15 for item in before))
        pivot = next(item for item in after if item["price"] == 15)
        self.assertEqual(pivot["confirmed_at_utc"], pd.Timestamp(frame.iloc[4]["bar_close_utc"]).isoformat())

    def test_future_bars_do_not_change_earlier_swing_set(self):
        frame = bars_from_ohlc([(10, 11, 9, 10), (10, 12, 9.5, 11), (11, 15, 10, 12), (12, 13, 10.5, 11), (11, 12, 9.8, 10), (10, 30, 8, 20)])
        as_of = frame.iloc[4]["bar_close_utc"]
        self.assertEqual(confirmed_swings(frame.iloc[:5], "M5", as_of=as_of), confirmed_swings(frame, "M5", as_of=as_of))


class ChartPatternTests(unittest.TestCase):
    def assert_pattern(self, expected: str, points: list[tuple[str, float]]) -> None:
        found = {item["pattern_type"]: item for item in detect_patterns(flat_bars(), "M5", swings(points))}
        self.assertIn(expected, found)
        item = found[expected]
        self.assertEqual(item["timeframe"], "M5")
        self.assertIn(item["status"], {"FORMING", "CONFIRMED", "BROKEN", "INVALIDATED"})
        self.assertIn("STRUCTURAL_GEOMETRY_FIT", item["evidence"]["score_semantics"])
        self.assertGreaterEqual(item["score"], 0)
        self.assertLessEqual(item["score"], 1)

    def test_reversal_and_range_pattern_geometries(self):
        cases = {
            "DOUBLE_TOP": [("HIGH",103),("LOW",100),("HIGH",103.02)],
            "DOUBLE_BOTTOM": [("LOW",97),("HIGH",100),("LOW",97.02)],
            "TRIPLE_TOP": [("HIGH",103),("LOW",100),("HIGH",103.01),("LOW",100.2),("HIGH",102.99)],
            "TRIPLE_BOTTOM": [("LOW",97),("HIGH",100),("LOW",97.01),("HIGH",99.8),("LOW",96.99)],
            "HEAD_AND_SHOULDERS": [("HIGH",103),("LOW",100),("HIGH",104),("LOW",100.2),("HIGH",103.02)],
            "INVERSE_HEAD_AND_SHOULDERS": [("LOW",97),("HIGH",100),("LOW",96),("HIGH",99.8),("LOW",96.98)],
            "ASCENDING_TRIANGLE": [("HIGH",103),("LOW",99),("HIGH",103.01),("LOW",99.3),("HIGH",102.99),("LOW",99.6)],
            "DESCENDING_TRIANGLE": [("HIGH",103),("LOW",99),("HIGH",102.7),("LOW",99.01),("HIGH",102.4),("LOW",98.99)],
            "SYMMETRICAL_TRIANGLE": [("HIGH",103),("LOW",99),("HIGH",102.7),("LOW",99.3),("HIGH",102.4),("LOW",99.6)],
            "EXPANDING_TRIANGLE": [("HIGH",102),("LOW",100),("HIGH",102.3),("LOW",99.7),("HIGH",102.6),("LOW",99.4)],
            "RISING_WEDGE": [("HIGH",102),("LOW",99),("HIGH",102.15),("LOW",99.3),("HIGH",102.3),("LOW",99.6)],
            "FALLING_WEDGE": [("HIGH",103),("LOW",100),("HIGH",102.7),("LOW",99.55),("HIGH",102.4),("LOW",99.1)],
            "RECTANGLE_RANGE": [("HIGH",103),("LOW",99),("HIGH",103.01),("LOW",99.01),("HIGH",102.99),("LOW",98.99)],
        }
        for name, points in cases.items():
            with self.subTest(name=name):
                self.assert_pattern(name, points)

    def continuation_bars(self, direction: int, pennant: bool) -> pd.DataFrame:
        impulse = np.linspace(100, 105 if direction > 0 else 95, 8)
        values = []
        for index, close in enumerate(impulse):
            prior = impulse[max(0, index - 1)]
            values.append((float(prior), float(max(prior, close) + .08), float(min(prior, close) - .08), float(close)))
        for index in range(16):
            if pennant:
                upper, lower = 105 - index * .06, 103 + index * .06
            elif direction > 0:
                upper, lower = 105 - index * .025, 104.5 - index * .025
            else:
                upper, lower = 95.5 + index * .025, 95 + index * .025
            close = (upper + lower) / 2
            values.append((close, upper, lower, close))
        return bars_from_ohlc(values)

    def test_continuation_pattern_geometries(self):
        points = swings([("HIGH",102),("LOW",100),("HIGH",103),("LOW",101),("HIGH",104),("LOW",102)])
        cases = [("BULL_FLAG",1,False),("BEAR_FLAG",-1,False),("BULL_PENNANT",1,True),("BEAR_PENNANT",-1,True)]
        for name, direction, pennant in cases:
            with self.subTest(name=name):
                found = {item["pattern_type"] for item in detect_patterns(self.continuation_bars(direction, pennant), "M5", points)}
                self.assertIn(name, found)

    def test_negative_geometry_does_not_force_any_named_pattern(self):
        detected = {item["pattern_type"] for item in detect_patterns(flat_bars(), "M5", swings([("HIGH",103),("LOW",99),("HIGH",105)]))}
        for name in PATTERN_TYPES:
            with self.subTest(name=name):
                self.assertNotIn(name, detected)


class PriceActionTests(unittest.TestCase):
    def names(self, values: list[tuple[float,float,float,float]]) -> set[str]:
        return {item["event_type"] for item in candle_events(bars_from_ohlc(values), "M5")}

    def test_single_and_two_candle_geometry(self):
        self.assertIn("DOJI", self.names([(10,11,9,10.05)]))
        self.assertIn("BULLISH_ENGULFING", self.names([(10.8,11,9.8,10),(9.9,11.1,9.7,11)]))
        self.assertIn("BEARISH_ENGULFING", self.names([(10,11.2,9.8,11),(11.1,11.3,9.7,9.9)]))
        self.assertIn("INSIDE_BAR", self.names([(10,12,8,11),(10.5,11.5,9,10.7)]))
        self.assertIn("OUTSIDE_BAR", self.names([(10,11,9,10.5),(10.4,12,8,11.5)]))
        self.assertIn("BULLISH_PIN_BAR", self.names([(10,10.3,8,10.2)]))
        self.assertIn("BEARISH_PIN_BAR", self.names([(10,12,9.8,9.9)]))

    def test_context_and_three_candle_geometry(self):
        down = [(13,13.2,12.7,12.8),(12.8,13,12.2,12.3),(12.3,12.5,11.7,11.8),(11.8,12,11.2,11.3),(11.3,11.5,9,11.2)]
        self.assertIn("HAMMER", self.names(down))
        inverted = down[:-1] + [(11.3,13.5,11.1,11.2)]
        self.assertIn("INVERTED_HAMMER", self.names(inverted))
        up = [(9,9.4,8.9,9.3),(9.3,9.8,9.2,9.7),(9.7,10.2,9.6,10.1),(10.1,10.7,10,10.6),(10.6,12.8,10.5,10.7)]
        self.assertIn("SHOOTING_STAR", self.names(up))
        hanging = up[:-1] + [(10.7,10.9,8.5,10.8)]
        self.assertIn("HANGING_MAN", self.names(hanging))
        self.assertIn("MORNING_STAR", self.names([(11,11.1,9.8,10),(10,10.2,9.8,10.05),(10,11,9.9,10.8)]))
        self.assertIn("EVENING_STAR", self.names([(10,11.2,9.9,11),(11,11.2,10.9,11.05),(11,11.1,9.9,10.1)]))
        self.assertIn("THREE_WHITE_SOLDIERS", self.names([(10,11,9.9,10.9),(10.8,11.8,10.7,11.7),(11.6,12.6,11.5,12.5)]))
        self.assertIn("THREE_BLACK_CROWS", self.names([(12.5,12.6,11.5,11.6),(11.7,11.8,10.7,10.8),(10.9,11,9.9,10)]))


class StructureLiquidityResearchTests(unittest.TestCase):
    def test_structure_labels_state_bos_and_choch(self):
        points = swings([("HIGH",102),("LOW",99),("HIGH",103),("LOW",100),("HIGH",104),("LOW",98)])
        labels = [item["structure_label"] for item in classify_swings(points)]
        self.assertIn("HH", labels); self.assertIn("HL", labels); self.assertIn("LL", labels)
        self.assertEqual(structure_state(classify_swings(points[:5])), "TREND_UP")
        frame = bars_from_ohlc([(100,101,99,100),(100,102,99.5,101),(101,102.5,100,102.2),(102.2,104,101,103.5),(103.5,104,97,97.5)])
        available = swings([("HIGH",102),("LOW",99)], start_index=-2)
        available[0]["confirmed_at_utc"] = frame.iloc[1]["bar_close_utc"].isoformat(); available[1]["confirmed_at_utc"] = frame.iloc[1]["bar_close_utc"].isoformat()
        events = structure_events(frame, "M5", available)
        self.assertIn("BOS", {item["event_type"] for item in events})
        self.assertIn("CHOCH", {item["event_type"] for item in events})

    def test_support_resistance_liquidity_sweep_and_fvg_fill(self):
        frame = bars_from_ohlc([(100,101,99,100),(100,103,99.8,102),(102,102.2,100,101),(101,103.02,100.5,102),(102,103.5,101,102.5),(102.5,102.8,99,100.5),(103.7,104,103.6,103.8),(103.5,104,100.8,101)])
        points = swings([("HIGH",103),("LOW",100),("HIGH",103.02)])
        for item in points: item["confirmed_at_utc"] = frame.iloc[3]["bar_close_utc"].isoformat()
        levels = support_resistance(frame, "M5", points)
        self.assertIsNotNone(levels["nearest_support"])
        self.assertTrue(levels["dynamic"] == [] or levels["dynamic"][0]["name"].startswith("EMA"))
        liquidity = liquidity_analysis(frame, "M5", points)
        self.assertTrue(liquidity["equal_levels"])
        self.assertTrue(liquidity["sweeps"])
        self.assertTrue(liquidity["fair_value_gaps"])
        gap = liquidity["fair_value_gaps"][-1]
        self.assertIn(gap["status"], {"OPEN", "PARTIALLY_FILLED", "FILLED"})
        before = liquidity_analysis(frame, "M5", points, as_of=frame.iloc[6]["bar_close_utc"])
        self.assertTrue(all(item["detected_at_utc"] <= frame.iloc[6]["bar_close_utc"].isoformat() for item in before["fair_value_gaps"]))
        self.assertTrue(any(item["status"] == "OPEN" for item in before["fair_value_gaps"]))
        self.assertTrue(any(item["status"] == "FILLED" for item in liquidity["fair_value_gaps"]))

    def test_breakout_and_retest_are_emitted_only_after_completed_closes(self):
        frame = bars_from_ohlc([(100,101,99,100),(100,101.5,99.8,101),(101,103,100.8,102.5),(102.5,102.7,101.9,102.2),(102.2,102.4,101.8,102.1)])
        level = {"level_id":"r1","kind":"RESISTANCE","price":102.0,"available_at_utc":frame.iloc[1]["bar_close_utc"].isoformat()}
        before = breakout_events(frame, "M5", [level], as_of=frame.iloc[1]["bar_close_utc"])
        after = breakout_events(frame, "M5", [level], as_of=frame.iloc[-1]["bar_close_utc"])
        self.assertEqual(before, [])
        self.assertIn("RESISTANCE_BREAKOUT", {item["event_type"] for item in after})
        self.assertIn("BREAKOUT_RETEST_CANDIDATE", {item["event_type"] for item in after})

    def test_research_is_sample_gated_and_ablation_groups_are_separate(self):
        frame = flat_bars(70)
        pattern = {"pattern_type":"DOUBLE_TOP","timeframe":"M5","detected_at_utc":frame.iloc[10]["bar_close_utc"].isoformat(),"direction":"BEARISH"}
        report = evaluate_pattern_records([pattern], frame, min_samples=30)
        self.assertTrue(report["sample_status"].eq("INSUFFICIENT_DATA").all())
        self.assertIn("CHART_PATTERNS", ABLATION_GROUPS)
        self.assertNotIn("production_model", sum((list(value) for value in ABLATION_GROUPS.values()), []))

    def test_engine_uses_only_completed_h1_and_is_observational(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            feature_path = root / "features.parquet"
            decision = pd.Timestamp("2026-08-24T01:00:00Z")
            pd.DataFrame({"decision_timestamp_utc":[decision],"m5_return_5m":[.001],"m5_return_15m":[.001],"m5_return_60m":[.001],"m5_return_240m":[.001],"m5_volatility_60m":[.0002],"m5_spread_points":[1]}).to_parquet(feature_path)
            for timeframe, minutes in (("M5",5),("M15",15),("H1",60)):
                frame = flat_bars(20)
                frame["bar_open_utc"] = pd.date_range("2026-08-23T05:00:00Z", periods=20, freq=f"{minutes}min")
                frame["bar_close_utc"] = frame["bar_open_utc"] + pd.Timedelta(minutes=minutes)
                frame["first_observed_utc"] = frame["bar_close_utc"]
                future = frame.iloc[-1:].copy(); future["bar_open_utc"] = decision; future["bar_close_utc"] = decision + pd.Timedelta(minutes=minutes); future["first_observed_utc"] = future["bar_close_utc"]
                pd.concat([frame, future], ignore_index=True).to_parquet(root / TIMEFRAMES[timeframe].filename)
            result = engine_status(decision, feature_path, root)
        self.assertEqual(result["contract_version"], ENGINE_CONTRACT_VERSION)
        self.assertLessEqual(pd.Timestamp(result["timeframes"]["H1"]["latest_completed_bar_utc"]), decision)
        self.assertTrue(result["observational_only"]); self.assertFalse(result["v06_integration"]); self.assertFalse(result["trading_enabled"])


if __name__ == "__main__":
    unittest.main()
