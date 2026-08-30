from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.learning.v05c_dataset import build_horizon_dataset
from src.research.v09a2_contract import (
    ABLATION_SEQUENCE, CHART_PATTERNS, MIN_EVENT_SAMPLES, PRODUCTION_INTEGRATION,
    grade_evidence,
)
from src.research.v09a2_evaluation import (
    ablation_features, chronological_research_split, engine_availability,
    engine_leaderboard_summary, evaluate_events, pattern_leaderboard,
)
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

    def test_fvg_partial_and_full_fill_publish_on_observed_bars(self):
        source=bars(9); source.loc[0,["high","low"]]=[1.0010,1.0000]
        source.loc[2,["open","high","low","close"]]=[1.0022,1.0030,1.0020,1.0025]
        source.loc[3,["open","high","low","close"]]=[1.0024,1.0026,1.0015,1.0020]
        source.loc[4,["open","high","low","close"]]=[1.0020,1.0022,1.0005,1.0010]
        features,events=replay_timeframe(source,"M5")
        self.assertGreater(float(features.loc[3,"liquidity_m5_fvg_partial_fill"]),0.0)
        self.assertGreater(float(features.loc[4,"liquidity_m5_fvg_fill"]),0.0)
        partial=events.loc[events.event_name.eq("FAIR_VALUE_GAP_PARTIALLY_FILLED")]
        self.assertTrue((pd.to_datetime(partial.detected_at_utc,utc=True)>=source.loc[3,"bar_close_utc"]).all())

    def test_bos_choch_and_support_events_never_predate_detection_bar(self):
        source=bars(18); source.loc[4,["open","high","low","close"]]=[1.10,1.20,1.09,1.11]
        source.loc[7,["open","high","low","close"]]=[1.19,1.22,1.18,1.21]
        source.loc[8,["open","high","low","close"]]=[1.15,1.16,1.05,1.10]
        source.loc[11,["open","high","low","close"]]=[1.06,1.07,1.03,1.04]
        features,events=replay_timeframe(source,"M5")
        event_rows=events.loc[events.event_name.str.startswith(("BOS_","CHOCH_","SUPPORT_","RESISTANCE_"),na=False)]
        self.assertFalse(event_rows.empty)
        bar_times=set(pd.to_datetime(source.bar_close_utc,utc=True))
        self.assertTrue(set(pd.to_datetime(event_rows.detected_at_utc,utc=True)).issubset(bar_times))
        bos=events.loc[events.event_name.eq("BOS_BULLISH")]
        choch=events.loc[events.event_name.eq("CHOCH_BEARISH")]
        support=events.loc[events.event_name.eq("SUPPORT_TOUCH")]
        self.assertFalse(bos.empty); self.assertFalse(choch.empty); self.assertFalse(support.empty)
        self.assertTrue((pd.to_datetime(bos.detected_at_utc,utc=True)>=source.loc[7,"bar_close_utc"]).all())
        support_time=pd.to_datetime(support.detected_at_utc,utc=True).min()
        support_index=int(source.index[pd.to_datetime(source.bar_close_utc,utc=True).eq(support_time)][0])
        self.assertGreater(float(features.loc[support_index,"support_resistance_m5_support_touch_count"]),0.0)
        self.assertEqual(float(features.loc[support_index-1,"support_resistance_m5_support_touch_count"]),0.0)
        self.assertFalse((features[["structure_m5_bos","structure_m5_choch"]].iloc[:6]!=0).any().any())

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

    def test_event_artifact_is_clipped_to_decision_window(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); all_bars=bars(200); decisions=all_bars.bar_close_utc.iloc[80:120]
            from src.marketdata.v05a_contract import TIMEFRAMES
            for tf,minutes in (("M5",5),("M15",15),("H1",60)):
                bars(200,minutes).to_parquet(root/TIMEFRAMES[tf].filename,index=False)
            replay=build_replay(decisions,root)
        detected=pd.to_datetime(replay.events.detected_at_utc,utc=True)
        self.assertTrue(detected.between(decisions.min(),decisions.max(),inclusive="both").all())

    def test_h4_absence_is_explicit(self):
        with TemporaryDirectory() as temp:
            audit=audit_bar_history(Path(temp))
        self.assertEqual(audit.loc[audit.timeframe.eq("H4"),"status"].iloc[0],"UNAVAILABLE")

    def test_canonical_session_flags_include_named_overlaps(self):
        features,_=replay_timeframe(bars(200),"M5")
        self.assertEqual(float(features.loc[0,"session_m5_sydney"]),1.0)
        self.assertEqual(float(features.loc[0,"session_m5_tokyo"]),1.0)
        self.assertEqual(float(features.loc[96,"session_m5_london"]),1.0)
        self.assertEqual(float(features.loc[156,"session_m5_new_york"]),1.0)
        self.assertGreaterEqual(float(features.loc[156,"session_m5_overlap_count"]),2.0)


class EvaluationContractTests(unittest.TestCase):
    def test_requested_ablation_order_is_frozen(self):
        self.assertEqual(ABLATION_SEQUENCE[0],"BASELINE")
        self.assertEqual(ABLATION_SEQUENCE[-1],"+ALL_V09")
        frame=pd.DataFrame(columns=["m5_return_5m","technical_m5_rsi14","pattern_m5_double_top"])
        self.assertIn("technical_m5_rsi14",ablation_features(frame,"+TECHNICAL"))
        self.assertNotIn("pattern_m5_double_top",ablation_features(frame,"+TECHNICAL"))
        volatility=pd.DataFrame(columns=["m5_return_5m","volatility_m5_rank","regime_m5_trend"])
        self.assertIn("volatility_m5_rank",ablation_features(volatility,"+VOLATILITY"))
        self.assertNotIn("regime_m5_trend",ablation_features(volatility,"+VOLATILITY"))
        self.assertEqual(len(ABLATION_SEQUENCE),10)

    def test_evidence_threshold_is_not_lowered(self):
        self.assertEqual(grade_evidence(MIN_EVENT_SAMPLES-1,.5,True),"INSUFFICIENT_SAMPLE")
        self.assertEqual(grade_evidence(MIN_EVENT_SAMPLES,.011,True),"STRONG_EVIDENCE")

    def test_event_statistics_are_suppressed_below_threshold(self):
        events=pd.DataFrame({"timeframe":["M5"]*3,"event_family":["CANDLE"]*3,"event_name":["DOJI"]*3,
            "detected_at_utc":pd.date_range("2026-01-05",periods=3,freq="5min",tz="UTC"),"direction":[0.0]*3,"score":[1.0]*3,"status":["CONFIRMED"]*3})
        target=pd.DataFrame({"decision_timestamp_utc":events.detected_at_utc,"target_class":[0,1,2],"raw_future_return":[-.1,0,.1],
            "is_asia_session":[1]*3,"is_london_session":[0]*3,"is_new_york_session":[0]*3,"is_london_ny_overlap":[0]*3})
        result=evaluate_events(events,{15:target}); row=result.loc[(result.event_name.eq("DOJI"))&(result.subgroup.eq("ALL"))].iloc[0]
        self.assertEqual(row.sample_status,"INSUFFICIENT_SAMPLE"); self.assertTrue(pd.isna(row.directional_accuracy))

    def test_final_holdout_is_the_untouched_last_chronological_block(self):
        source=pd.DataFrame({"decision_timestamp_utc":pd.date_range("2025-01-01",periods=4000,freq="5min",tz="UTC"),"marker":np.arange(4000)})
        selection,holdout=chronological_research_split(source.sample(frac=1,random_state=7))
        self.assertEqual(len(holdout),600)
        self.assertLess(selection.decision_timestamp_utc.max(),holdout.decision_timestamp_utc.min())
        self.assertEqual(int(holdout.marker.iloc[0]),3400)

    def test_availability_masks_leave_h4_explicitly_unavailable(self):
        frame=pd.DataFrame({"decision_timestamp_utc":pd.date_range("2026-01-01",periods=3,tz="UTC"),
            "technical_m5_rsi14":[1.0,np.nan,2.0],"availability_technical_m5_missing_mask":[0,1,0],
            "availability_technical_m5_partial_mask":[0,0,0]})
        result=engine_availability(frame)
        h4=result.loc[(result.engine.eq("TECHNICAL"))&(result.timeframe.eq("H4"))].iloc[0]
        self.assertEqual(h4.status,"UNAVAILABLE"); self.assertEqual(int(h4.unavailable),3)

    def test_pattern_leaderboard_always_discloses_every_supported_pattern(self):
        result=pattern_leaderboard(pd.DataFrame(),pd.DataFrame())
        self.assertEqual(set(result.pattern),set(CHART_PATTERNS))
        self.assertTrue(result["15m_evidence"].eq("INSUFFICIENT_SAMPLE").all())

    def test_research_contract_cannot_integrate_with_production(self):
        self.assertFalse(PRODUCTION_INTEGRATION)

    def test_session_context_without_separate_ablation_is_held(self):
        ablation=pd.DataFrame([{"ablation":name,"horizon_minutes":horizon,"evidence_grade":"NO_MEANINGFUL_EVIDENCE",
            "validation_rows":1000,"holdout_rows":200,"stability_pass":False}
            for name in ABLATION_SEQUENCE for horizon in (15,60,240)])
        availability=pd.DataFrame({"engine":["SESSION"]*3,"timeframe":["M5","M15","H1"],"availability_percentage":[100.0]*3})
        redundancy=pd.DataFrame(columns=["analysis","feature","recommendation"])
        result=engine_leaderboard_summary(ablation,availability,redundancy)
        session=result.loc[result.engine.eq("SESSION")].iloc[0]
        self.assertEqual(session.recommendation,"HOLD")
        self.assertEqual(session["15m_evidence"],"NO_SEPARATE_ABLATION_BASELINE_CONTEXT")

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
