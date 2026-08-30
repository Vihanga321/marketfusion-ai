from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.learning.v05c_dataset import build_horizon_dataset
from src.research.v09a2_contract import ABLATION_SEQUENCE, MIN_EVENT_SAMPLES, grade_evidence
from src.research.v09a2_evaluation import ablation_features, evaluate_events
from src.research.v09a2_replay import audit_bar_history, build_replay, replay_timeframe


def bars(count: int = 40, minutes: int = 5) -> pd.DataFrame:
    close=1.1000+np.sin(np.arange(count)/3)*.001
    frame=pd.DataFrame({"bar_close_utc":pd.date_range("2026-01-05T00:05:00Z",periods=count,freq=f"{minutes}min")})
    frame["open"]=np.r_[close[0],close[:-1]]; frame["close"]=close
    frame["high"]=np.maximum(frame.open,frame.close)+.0002; frame["low"]=np.minimum(frame.open,frame.close)-.0002
    frame["tick_volume"]=100; frame["spread_points"]=1
    return frame


class ReplayCausalityTests(unittest.TestCase):
    def test_future_bars_do_not_change_earlier_features_or_events(self):
        source=bars(45); cutoff=source.iloc[34].bar_close_utc
        left,left_events=replay_timeframe(source.iloc[:35],"M5")
        full,full_events=replay_timeframe(source,"M5")
        pd.testing.assert_frame_equal(left,full.loc[full.available_at_utc.le(cutoff)].reset_index(drop=True))
        earlier=full_events.loc[pd.to_datetime(full_events.detected_at_utc,utc=True).le(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(left_events.reset_index(drop=True),earlier)

    def test_confirmed_pivot_is_delayed_two_completed_bars(self):
        source=bars(12); source.loc[4,"high"]=1.2
        features,_=replay_timeframe(source,"M5")
        column="structure_m5_confirmed_swing"
        self.assertEqual(float(features.loc[4,column]),0.0)
        self.assertNotEqual(float(features.loc[6,column]),0.0)

    def test_fvg_fill_is_not_visible_at_detection(self):
        source=bars(8); source.loc[0,["high","low"]]=[1.0010,1.0000]
        source.loc[2,["open","high","low","close"]]=[1.0022,1.0030,1.0020,1.0025]
        source.loc[3,["open","high","low","close"]]=[1.0025,1.0028,1.0005,1.0010]
        features,events=replay_timeframe(source,"M5")
        self.assertEqual(float(features.loc[2,"liquidity_m5_fvg_new"]),1.0)
        self.assertEqual(float(features.loc[2,"liquidity_m5_fvg_fill"]),0.0)
        self.assertGreater(float(features.loc[3,"liquidity_m5_fvg_fill"]),0.0)
        filled=events.loc[events.event_name.eq("FAIR_VALUE_GAP_FILLED")]
        self.assertTrue((pd.to_datetime(filled.detected_at_utc,utc=True)>=source.loc[3,"bar_close_utc"]).all())

    def test_pattern_lifecycle_status_is_published_only_later(self):
        _,events=replay_timeframe(bars(200),"M5")
        resolved=events.loc[events.event_family.eq("CHART_PATTERN")&events.status.ne("DETECTED")]
        self.assertFalse(resolved.empty)
        self.assertTrue((pd.to_datetime(resolved.detected_at_utc,utc=True)>pd.to_datetime(resolved.origin_detected_at_utc,utc=True)).all())

    def test_replay_feature_artifact_has_no_outcome_columns(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); decisions=bars(40).bar_close_utc
            from src.marketdata.v05a_contract import TIMEFRAMES
            for tf,minutes in (("M5",5),("M15",15),("H1",60)):
                bars(40,minutes).to_parquet(root/TIMEFRAMES[tf].filename,index=False)
            replay=build_replay(decisions,root)
        forbidden=[c for c in replay.features if c.lower().startswith(("future_","target_","outcome_"))]
        self.assertEqual(forbidden,[])
        for column in ("m5_engine_available_at_utc","m15_engine_available_at_utc","h1_engine_available_at_utc"):
            self.assertFalse((pd.to_datetime(replay.features[column],utc=True)>replay.features.decision_timestamp_utc).fillna(False).any())

    def test_h4_absence_is_explicit(self):
        with TemporaryDirectory() as temp:
            audit=audit_bar_history(Path(temp))
        self.assertEqual(audit.loc[audit.timeframe.eq("H4"),"status"].iloc[0],"UNAVAILABLE")


class EvaluationContractTests(unittest.TestCase):
    def test_requested_ablation_order_is_frozen(self):
        self.assertEqual(ABLATION_SEQUENCE[0],"BASELINE")
        self.assertEqual(ABLATION_SEQUENCE[-1],"+ALL_V09")
        frame=pd.DataFrame(columns=["m5_return_5m","technical_m5_rsi14","pattern_m5_double_top"])
        self.assertIn("technical_m5_rsi14",ablation_features(frame,"+TECHNICAL"))
        self.assertNotIn("pattern_m5_double_top",ablation_features(frame,"+TECHNICAL"))

    def test_evidence_threshold_is_not_lowered(self):
        self.assertEqual(grade_evidence(MIN_EVENT_SAMPLES-1,.5,True),"INSUFFICIENT_DATA")
        self.assertEqual(grade_evidence(MIN_EVENT_SAMPLES,.011,True),"STRONG")

    def test_event_statistics_are_suppressed_below_threshold(self):
        events=pd.DataFrame({"timeframe":["M5"]*3,"event_family":["CANDLE"]*3,"event_name":["DOJI"]*3,
            "detected_at_utc":pd.date_range("2026-01-05",periods=3,freq="5min",tz="UTC"),"direction":[0.0]*3,"score":[1.0]*3,"status":["CONFIRMED"]*3})
        target=pd.DataFrame({"decision_timestamp_utc":events.detected_at_utc,"target_class":[0,1,2],"raw_future_return":[-.1,0,.1],
            "is_asia_session":[1]*3,"is_london_session":[0]*3,"is_new_york_session":[0]*3,"is_london_ny_overlap":[0]*3})
        result=evaluate_events(events,{15:target}); row=result.loc[(result.event_name.eq("DOJI"))&(result.subgroup.eq("ALL"))].iloc[0]
        self.assertEqual(row.sample_status,"INSUFFICIENT_DATA"); self.assertTrue(pd.isna(row.directional_accuracy))

    def test_exact_v05c_target_semantics_are_reused(self):
        from src.learning.v05c_contract import MARKET_FEATURES
        count=21; source=pd.DataFrame({"decision_timestamp_utc":pd.date_range("2026-01-05",periods=count,freq="5min",tz="UTC")})
        for name in MARKET_FEATURES: source[name]=1.0
        source["feature_complete"]=True; source["m5_close"]=1.1; source["m5_spread_points"]=1
        for name in ("m1","m5","m15","h1"): source[f"{name}_available_from_utc"]=source.decision_timestamp_utc
        source["outcome_future_timestamp_15m"]=source.decision_timestamp_utc+pd.Timedelta(minutes=15)
        source["outcome_matured_at_utc_15m"]=source.outcome_future_timestamp_15m
        source["outcome_future_return_15m"]=np.linspace(-.001,.001,count)
        result=build_horizon_dataset(source,15,as_of=source.outcome_matured_at_utc_15m.max())
        self.assertEqual(set(result.frame.target_class.unique()),{0,1,2})


if __name__ == "__main__":
    unittest.main()
