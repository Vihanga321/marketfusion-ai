"""Vectorized/single-pass causal replay for V0.9A.2.

All output values are available at the completed bar timestamp carrying them.
Pivots use two left and two right bars and are published two closes after the
pivot. Future outcomes are deliberately stored by the runner in a separate file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.engines.patterns import PATTERN_TYPES
from src.marketdata.market_calendar import SESSION_SCHEDULES
from src.marketdata.v05a_contract import CONTINUOUS_DIR, TIMEFRAMES
from src.research.v09a2_contract import AVAILABLE_TIMEFRAMES, CONTRACT_VERSION


@dataclass(frozen=True)
class ReplayResult:
    features: pd.DataFrame
    events: pd.DataFrame
    audit: pd.DataFrame


DIRECTION = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}


def _utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise")


def audit_bar_history(root: Path = CONTINUOUS_DIR) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for timeframe in ("M5", "M15", "H1", "H4"):
        spec = TIMEFRAMES.get(timeframe)
        path = None if spec is None else root / spec.filename
        if path is None or not path.exists():
            rows.append({"timeframe": timeframe, "status": "UNAVAILABLE", "rows": 0,
                         "start_utc": None, "end_utc": None, "duplicate_timestamps": None,
                         "nonpositive_ohlc": None, "gaps_over_nominal": None,
                         "gaps_under_24h": None, "weekend_or_long_gaps": None,
                         "largest_gap_minutes": None, "timezone": None})
            continue
        frame = pd.read_parquet(path)
        times = _utc(frame["bar_close_utc"])
        delta = times.sort_values().diff().dt.total_seconds().div(60)
        bad_ohlc = ((frame[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum()
        rows.append({"timeframe": timeframe, "status": "AVAILABLE", "rows": len(frame),
                     "start_utc": times.min(), "end_utc": times.max(),
                     "duplicate_timestamps": int(times.duplicated().sum()),
                     "nonpositive_ohlc": int(bad_ohlc),
                     "gaps_over_nominal": int((delta > spec.minutes).sum()),
                     "gaps_under_24h": int(((delta > spec.minutes)&(delta < 24*60)).sum()),
                     "weekend_or_long_gaps": int((delta >= 24*60).sum()),
                     "largest_gap_minutes": float(delta.max()), "timezone": str(times.dt.tz)})
    return pd.DataFrame(rows)


def _event(rows: list[dict[str, Any]], timeframe: str, family: str, name: str,
           timestamp: pd.Timestamp, direction: float, score: float = 1.0,
           status: str = "CONFIRMED", **extra: Any) -> None:
    rows.append({"timeframe": timeframe, "event_family": family, "event_name": name,
                 "detected_at_utc": timestamp, "direction": direction,
                 "score": float(np.clip(score, 0.0, 1.0)), "status": status, **extra})


def _resolve_pattern_lifecycles(events: list[dict[str, Any]], bars: pd.DataFrame,
                                close: pd.Series, max_bars: int = 120) -> tuple[np.ndarray,np.ndarray]:
    """Publish resolution only on the later close that actually resolves it."""
    times=pd.to_datetime(bars["bar_close_utc"],utc=True); locations={value.value:index for index,value in enumerate(times)}
    confirmed=np.zeros(len(bars)); invalidated=np.zeros(len(bars)); resolutions=[]
    detections=[item for item in events if item["event_family"]=="CHART_PATTERN" and item["status"]=="DETECTED"]
    for item in detections:
        start=locations[pd.Timestamp(item["detected_at_utc"]).value]; upper=item.get("upper_boundary"); lower=item.get("lower_boundary")
        if upper is None or lower is None or not np.isfinite([upper,lower]).all() or upper<=lower: continue
        expected=float(item["direction"]); resolved=False
        stop=min(len(bars),start+max_bars+1)
        for index in range(start+1,stop):
            actual=1.0 if close.iloc[index]>upper else -1.0 if close.iloc[index]<lower else 0.0
            if actual==0: continue
            status="CONFIRMED" if expected==0 or actual==expected else "INVALIDATED"
            (confirmed if status=="CONFIRMED" else invalidated)[index]+=actual
            resolutions.append({"timeframe":item["timeframe"],"event_family":"CHART_PATTERN","event_name":item["event_name"],
                "detected_at_utc":times.iloc[index],"direction":actual,"score":item["score"],"status":status,
                "origin_detected_at_utc":item["detected_at_utc"],"expected_direction":expected,
                "upper_boundary":upper,"lower_boundary":lower})
            resolved=True; break
        if not resolved and start+max_bars<len(bars):
            resolutions.append({"timeframe":item["timeframe"],"event_family":"CHART_PATTERN","event_name":item["event_name"],
                "detected_at_utc":times.iloc[start+max_bars],"direction":expected,"score":item["score"],"status":"EXPIRED_FORMING",
                "origin_detected_at_utc":item["detected_at_utc"],"expected_direction":expected,
                "upper_boundary":upper,"lower_boundary":lower})
    events.extend(resolutions)
    return confirmed,invalidated


def _chart_name(points: list[tuple[str, float]], tolerance: float) -> list[tuple[str, float]]:
    """Evaluate only geometry already confirmed in ``points``."""
    found: list[tuple[str, float]] = []
    if len(points) >= 3:
        three = points[-3:]
        kinds = [x[0] for x in three]
        if kinds == ["HIGH", "LOW", "HIGH"]:
            error = abs(three[0][1] - three[2][1]); depth = (three[0][1] + three[2][1]) / 2 - three[1][1]
            if error <= tolerance and depth >= 2 * tolerance: found.append(("DOUBLE_TOP", 1 - error / max(tolerance, 1e-12)))
        elif kinds == ["LOW", "HIGH", "LOW"]:
            error = abs(three[0][1] - three[2][1]); height = three[1][1] - (three[0][1] + three[2][1]) / 2
            if error <= tolerance and height >= 2 * tolerance: found.append(("DOUBLE_BOTTOM", 1 - error / max(tolerance, 1e-12)))
    if len(points) >= 5:
        five = points[-5:]; kinds = [x[0] for x in five]
        if kinds == ["HIGH", "LOW", "HIGH", "LOW", "HIGH"]:
            peaks = np.array([five[i][1] for i in (0, 2, 4)])
            spread = float(np.ptp(peaks)); neck = min(five[1][1], five[3][1])
            if spread <= tolerance and peaks.mean() - neck >= 2*tolerance: found.append(("TRIPLE_TOP", 1-spread/max(tolerance,1e-12)))
            shoulder = abs(five[0][1]-five[4][1]); head = five[2][1]-max(five[0][1],five[4][1])
            if shoulder <= 1.5*tolerance and head >= 1.5*tolerance: found.append(("HEAD_AND_SHOULDERS", min(1,head/max(3*tolerance,1e-12))))
        elif kinds == ["LOW", "HIGH", "LOW", "HIGH", "LOW"]:
            troughs = np.array([five[i][1] for i in (0, 2, 4)])
            spread = float(np.ptp(troughs)); neck = max(five[1][1], five[3][1])
            if spread <= tolerance and neck-troughs.mean() >= 2*tolerance: found.append(("TRIPLE_BOTTOM", 1-spread/max(tolerance,1e-12)))
            shoulder = abs(five[0][1]-five[4][1]); head = min(five[0][1],five[4][1])-five[2][1]
            if shoulder <= 1.5*tolerance and head >= 1.5*tolerance: found.append(("INVERSE_HEAD_AND_SHOULDERS", min(1,head/max(3*tolerance,1e-12))))
    if len(points) >= 6:
        six = points[-6:]
        highs = np.array([x[1] for x in six if x[0] == "HIGH"]); lows = np.array([x[1] for x in six if x[0] == "LOW"])
        if len(highs) >= 3 and len(lows) >= 3:
            hs = np.polyfit(np.arange(len(highs)), highs, 1)[0] / max(tolerance,1e-12)
            ls = np.polyfit(np.arange(len(lows)), lows, 1)[0] / max(tolerance,1e-12)
            flat=.45; name=None
            if abs(hs)<=flat and ls>flat: name="ASCENDING_TRIANGLE"
            elif abs(ls)<=flat and hs < -flat: name="DESCENDING_TRIANGLE"
            elif hs < -flat and ls > flat: name="SYMMETRICAL_TRIANGLE"
            elif hs > flat and ls < -flat: name="EXPANDING_TRIANGLE"
            elif hs > flat and ls > hs+flat: name="RISING_WEDGE"
            elif hs < -flat and ls < -flat and hs > ls+flat: name="FALLING_WEDGE"
            elif abs(hs)<=flat and abs(ls)<=flat: name="RECTANGLE_RANGE"
            if name: found.append((name, min(1.0, (abs(hs)+abs(ls)+.5)/2)))
    return found


def replay_timeframe(frame: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = frame.copy().sort_values("bar_close_utc").drop_duplicates("bar_close_utc", keep="last").reset_index(drop=True)
    bars["bar_close_utc"] = _utc(bars["bar_close_utc"])
    o,h,l,c = [pd.to_numeric(bars[x], errors="raise").astype(float) for x in ("open","high","low","close")]
    prefix = timeframe.lower()
    out = pd.DataFrame({"available_at_utc": bars["bar_close_utc"]})
    prior = c.shift(1); tr = pd.concat([(h-l),(h-prior).abs(),(l-prior).abs()],axis=1).max(axis=1)
    atr = tr.rolling(14,min_periods=3).mean()
    ema20=c.ewm(span=20,adjust=False).mean(); ema50=c.ewm(span=50,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rsi=100-100/(1+gain/loss.replace(0,np.nan)); mid=c.rolling(20,min_periods=5).mean(); std=c.rolling(20,min_periods=5).std()
    out[f"technical_{prefix}_rsi14"]=(rsi-50)/50
    out[f"technical_{prefix}_ema20_gap_atr"]=(c-ema20)/atr.replace(0,np.nan)
    out[f"technical_{prefix}_ema20_50_gap_atr"]=(ema20-ema50)/atr.replace(0,np.nan)
    out[f"technical_{prefix}_bollinger_z"]=(c-mid)/std.replace(0,np.nan)
    out[f"technical_{prefix}_atr_fraction"]=atr/c
    out[f"technical_{prefix}_range_atr"]=(h-l)/atr.replace(0,np.nan)

    body=(c-o).abs(); span=(h-l).replace(0,np.nan); upper=h-pd.concat([o,c],axis=1).max(axis=1); lower=pd.concat([o,c],axis=1).min(axis=1)-l
    br=body/span; ur=upper/span; lr=lower/span; context=c.shift(1)-c.shift(4)
    candle_flags: dict[str,pd.Series] = {
        "DOJI": br<=.10,
        "HAMMER": (lr>=.60)&(br<=.35)&(ur<=.20)&(context<-.25*atr),
        "HANGING_MAN": (lr>=.60)&(br<=.35)&(ur<=.20)&(context>.25*atr),
        "BULLISH_PIN_BAR": (lr>=.60)&(br<=.35)&(ur<=.20)&(context.abs()<=.25*atr),
        "SHOOTING_STAR": (ur>=.60)&(br<=.35)&(lr<=.20)&(context>.25*atr),
        "INVERTED_HAMMER": (ur>=.60)&(br<=.35)&(lr<=.20)&(context<-.25*atr),
        "BEARISH_PIN_BAR": (ur>=.60)&(br<=.35)&(lr<=.20)&(context.abs()<=.25*atr),
        "BULLISH_ENGULFING": (c>o)&(c.shift(1)<o.shift(1))&(o<=c.shift(1))&(c>=o.shift(1)),
        "BEARISH_ENGULFING": (c<o)&(c.shift(1)>o.shift(1))&(o>=c.shift(1))&(c<=o.shift(1)),
        "INSIDE_BAR": (h<h.shift(1))&(l>l.shift(1)),
        "OUTSIDE_BAR": (h>h.shift(1))&(l<l.shift(1)),
    }
    small_middle=body.shift(1)<=.35*span.shift(1); midpoint=(o.shift(2)+c.shift(2))/2
    candle_flags.update({
        "MORNING_STAR": (c.shift(2)<o.shift(2))&small_middle&(c>o)&(c>midpoint),
        "EVENING_STAR": (c.shift(2)>o.shift(2))&small_middle&(c<o)&(c<midpoint),
        "THREE_WHITE_SOLDIERS": (c>o)&(c.shift(1)>o.shift(1))&(c.shift(2)>o.shift(2))&(c>c.shift(1))&(c.shift(1)>c.shift(2)),
        "THREE_BLACK_CROWS": (c<o)&(c.shift(1)<o.shift(1))&(c.shift(2)<o.shift(2))&(c<c.shift(1))&(c.shift(1)<c.shift(2)),
    })
    directions={"HAMMER":1,"BULLISH_PIN_BAR":1,"INVERTED_HAMMER":1,"BULLISH_ENGULFING":1,"MORNING_STAR":1,"THREE_WHITE_SOLDIERS":1,
                "HANGING_MAN":-1,"SHOOTING_STAR":-1,"BEARISH_PIN_BAR":-1,"BEARISH_ENGULFING":-1,"EVENING_STAR":-1,"THREE_BLACK_CROWS":-1}
    events: list[dict[str,Any]]=[]
    pa_score=pd.Series(0.0,index=bars.index)
    for name, flag in candle_flags.items():
        values=flag.fillna(False); direction=float(directions.get(name,0)); pa_score += values.astype(float)*direction
        for idx in np.flatnonzero(values.to_numpy()): _event(events,timeframe,"CANDLE",name,bars.at[idx,"bar_close_utc"],direction,float(br.iloc[idx]) if pd.notna(br.iloc[idx]) else .5)
    out[f"price_action_{prefix}_direction_score"]=pa_score.clip(-3,3)/3
    out[f"price_action_{prefix}_event_count"]=pd.DataFrame(candle_flags).fillna(False).sum(axis=1)
    out[f"price_action_{prefix}_body_direction"]=(c-o)/atr.replace(0,np.nan)

    # A pivot is published at i+2; no feature at or before the pivot sees its right-hand bars.
    pivot_high=((h.shift(2)>h.shift(3))&(h.shift(2)>h.shift(4))&(h.shift(2)>h.shift(1))&(h.shift(2)>h)).fillna(False)
    pivot_low=((l.shift(2)<l.shift(3))&(l.shift(2)<l.shift(4))&(l.shift(2)<l.shift(1))&(l.shift(2)<l)).fillna(False)
    structure_state=np.zeros(len(bars)); swing_signal=np.zeros(len(bars)); bos=np.zeros(len(bars)); choch=np.zeros(len(bars))
    structure_labels={name:np.zeros(len(bars)) for name in ("HH","HL","LH","LL")}
    structure_regimes={name:np.zeros(len(bars)) for name in ("TREND_UP","TREND_DOWN","RANGE","TRANSITION")}
    pattern_values={name:np.zeros(len(bars)) for name in PATTERN_TYPES}; pattern_count=np.zeros(len(bars))
    equal_high=np.zeros(len(bars)); equal_low=np.zeros(len(bars)); sweep=np.zeros(len(bars)); nearest_liquidity=np.full(len(bars),np.nan)
    sr_support=np.full(len(bars),np.nan); sr_resistance=np.full(len(bars),np.nan); fib_distance=np.full(len(bars),np.nan)
    support_break=np.zeros(len(bars)); resistance_break=np.zeros(len(bars)); support_touches=np.zeros(len(bars)); resistance_touches=np.zeros(len(bars))
    alternating: list[tuple[str,float]]=[]; same_kind:dict[str,float]={}; trend=0.0; eq_levels:list[tuple[str,float]]=[]
    recent_structure_labels:list[str]=[]; structural_regime="RANGE"
    last_high=None; last_low=None; last_support_touch=-10_000; last_resistance_touch=-10_000; support_touch_total=0; resistance_touch_total=0
    for i in range(len(bars)):
        tolerance=max((atr.iloc[i] if pd.notna(atr.iloc[i]) else 0)*.25,c.iloc[i]*.00008)
        new: list[tuple[str,float]]=[]
        if pivot_high.iloc[i]: new.append(("HIGH",float(h.iloc[i-2])))
        if pivot_low.iloc[i]: new.append(("LOW",float(l.iloc[i-2])))
        for kind,price in new:
            prior_same=same_kind.get(kind); same_kind[kind]=price
            label=(1.0 if kind=="HIGH" else -1.0) if prior_same is None else np.sign(price-prior_same)
            swing_signal[i]=label
            if prior_same is not None:
                structure_label=("HH" if price>prior_same else "LH") if kind=="HIGH" else ("HL" if price>prior_same else "LL")
                structure_labels[structure_label][i]=1.0
                recent_structure_labels.append(structure_label); recent_structure_labels=recent_structure_labels[-8:]
                _event(events,timeframe,"STRUCTURE",structure_label,bars.at[i,"bar_close_utc"],1 if structure_label in {"HH","HL"} else -1)
                label_set=set(recent_structure_labels)
                next_regime="TREND_UP" if {"HH","HL"}.issubset(label_set) and not label_set.intersection({"LH","LL"}) else "TREND_DOWN" if {"LH","LL"}.issubset(label_set) and not label_set.intersection({"HH","HL"}) else "TRANSITION" if label_set.intersection({"HH","HL"}) and label_set.intersection({"LH","LL"}) else "RANGE"
                if next_regime!=structural_regime:
                    structural_regime=next_regime
                    _event(events,timeframe,"STRUCTURE",structural_regime,bars.at[i,"bar_close_utc"],1 if structural_regime=="TREND_UP" else -1 if structural_regime=="TREND_DOWN" else 0)
            if prior_same is not None and abs(price-prior_same)<=max(tolerance*.48,c.iloc[i]*.00004):
                eq_levels.append((kind,(price+prior_same)/2)); equal_high[i if kind=="HIGH" else 0]+=1 if kind=="HIGH" else 0; equal_low[i if kind=="LOW" else 0]+=1 if kind=="LOW" else 0
                _event(events,timeframe,"LIQUIDITY","EQUAL_HIGHS" if kind=="HIGH" else "EQUAL_LOWS",bars.at[i,"bar_close_utc"],-1 if kind=="HIGH" else 1)
            if alternating and alternating[-1][0]==kind:
                better=(kind=="HIGH" and price>=alternating[-1][1]) or (kind=="LOW" and price<=alternating[-1][1])
                if better: alternating[-1]=(kind,price)
            else: alternating.append((kind,price))
            alternating=alternating[-14:]
            for name,score in _chart_name(alternating,tolerance):
                direction=-1.0 if name in {"DOUBLE_TOP","TRIPLE_TOP","HEAD_AND_SHOULDERS","RISING_WEDGE","DESCENDING_TRIANGLE"} else 1.0 if name in {"DOUBLE_BOTTOM","TRIPLE_BOTTOM","INVERSE_HEAD_AND_SHOULDERS","FALLING_WEDGE","ASCENDING_TRIANGLE"} else 0.0
                pattern_values[name][i]=direction if direction else score; pattern_count[i]+=1
                prices=[point[1] for point in alternating]
                _event(events,timeframe,"CHART_PATTERN",name,bars.at[i,"bar_close_utc"],direction,score,"DETECTED",
                       upper_boundary=max(prices),lower_boundary=min(prices))
            if kind=="HIGH": last_high=price
            else: last_low=price
        if i>0 and last_high is not None and c.iloc[i-1]<=last_high<c.iloc[i]:
            is_choch=trend<0; (choch if is_choch else bos)[i]=1; trend=1
            _event(events,timeframe,"STRUCTURE","CHOCH_BULLISH" if is_choch else "BOS_BULLISH",bars.at[i,"bar_close_utc"],1)
            resistance_break[i]=1; _event(events,timeframe,"SUPPORT_RESISTANCE","RESISTANCE_BREAK",bars.at[i,"bar_close_utc"],1)
        if i>0 and last_low is not None and c.iloc[i-1]>=last_low>c.iloc[i]:
            is_choch=trend>0; (choch if is_choch else bos)[i]=-1; trend=-1
            _event(events,timeframe,"STRUCTURE","CHOCH_BEARISH" if is_choch else "BOS_BEARISH",bars.at[i,"bar_close_utc"],-1)
            support_break[i]=-1; _event(events,timeframe,"SUPPORT_RESISTANCE","SUPPORT_BREAK",bars.at[i,"bar_close_utc"],-1)
        structure_state[i]=1 if structural_regime=="TREND_UP" else -1 if structural_regime=="TREND_DOWN" else .5 if structural_regime=="TRANSITION" else 0
        structure_regimes[structural_regime][i]=1
        for kind,price in eq_levels[-20:]:
            if kind=="HIGH" and h.iloc[i]>price+tolerance and c.iloc[i]<price:
                sweep[i]-=1
                _event(events,timeframe,"LIQUIDITY","HIGH_SWEEP",bars.at[i,"bar_close_utc"],-1)
                _event(events,timeframe,"LIQUIDITY","BEARISH_LIQUIDITY_SWEEP",bars.at[i,"bar_close_utc"],-1)
                eq_levels.remove((kind,price)); break
            if kind=="LOW" and l.iloc[i]<price-tolerance and c.iloc[i]>price:
                sweep[i]+=1
                _event(events,timeframe,"LIQUIDITY","LOW_SWEEP",bars.at[i,"bar_close_utc"],1)
                _event(events,timeframe,"LIQUIDITY","BULLISH_LIQUIDITY_SWEEP",bars.at[i,"bar_close_utc"],1)
                eq_levels.remove((kind,price)); break
        if eq_levels: nearest_liquidity[i]=min(abs(c.iloc[i]-price) for _,price in eq_levels[-20:])/max(atr.iloc[i],1e-12)
        sr_support[i]=last_low if last_low is not None and last_low<=c.iloc[i] else np.nan
        sr_resistance[i]=last_high if last_high is not None and last_high>=c.iloc[i] else np.nan
        touch_tolerance=max(tolerance*.6,c.iloc[i]*.00004)
        if i-last_support_touch>2 and pd.notna(sr_support[i]) and l.iloc[i]<=sr_support[i]+touch_tolerance and c.iloc[i]>=sr_support[i]:
            support_touch_total+=1; _event(events,timeframe,"SUPPORT_RESISTANCE","SUPPORT_TOUCH",bars.at[i,"bar_close_utc"],1); last_support_touch=i
        if i-last_resistance_touch>2 and pd.notna(sr_resistance[i]) and h.iloc[i]>=sr_resistance[i]-touch_tolerance and c.iloc[i]<=sr_resistance[i]:
            resistance_touch_total+=1; _event(events,timeframe,"SUPPORT_RESISTANCE","RESISTANCE_TOUCH",bars.at[i,"bar_close_utc"],-1); last_resistance_touch=i
        support_touches[i]=min(support_touch_total,10); resistance_touches[i]=min(resistance_touch_total,10)
        if last_high is not None and last_low is not None and last_high!=last_low:
            low_bound,high_bound=sorted((last_low,last_high)); fib_levels=[low_bound+(high_bound-low_bound)*ratio for ratio in (.236,.382,.5,.618,.786)]
            fib_distance[i]=min(abs(c.iloc[i]-level) for level in fib_levels)/max(atr.iloc[i],1e-12)
    out[f"structure_{prefix}_state"]=structure_state
    out[f"structure_{prefix}_confirmed_swing"]=swing_signal
    out[f"structure_{prefix}_bos"]=bos; out[f"structure_{prefix}_choch"]=choch
    for name,values in structure_labels.items(): out[f"structure_{prefix}_{name.lower()}"]=values
    for name,values in structure_regimes.items(): out[f"structure_{prefix}_{name.lower()}"]=values

    # Continuation geometry is computed from the current and preceding 23 bars only.
    impulse=c.shift(16)-o.shift(23); imp_atr=impulse/atr.replace(0,np.nan)
    high_slope=(h-h.shift(15))/15/atr.replace(0,np.nan); low_slope=(l-l.shift(15))/15/atr.replace(0,np.nan)
    parallel=(high_slope-low_slope).abs()<=.08; converge=(high_slope<-.01)&(low_slope>.01)
    continuation={"BULL_FLAG":(imp_atr>2.5)&parallel&(high_slope<.02),"BEAR_FLAG":(imp_atr< -2.5)&parallel&(low_slope>-.02),
                  "BULL_PENNANT":(imp_atr>2.5)&converge,"BEAR_PENNANT":(imp_atr< -2.5)&converge}
    for name,flag in continuation.items():
        prior_flag=flag.shift(1,fill_value=False).astype(bool)
        values=flag.fillna(False).astype(bool)&prior_flag.eq(False); direction=1.0 if name.startswith("BULL") else -1.0
        pattern_values[name][values.to_numpy()]=direction; pattern_count[values.to_numpy()]+=1
        for idx in np.flatnonzero(values.to_numpy()):
            _event(events,timeframe,"CHART_PATTERN",name,bars.at[idx,"bar_close_utc"],direction,min(1,abs(float(imp_atr.iloc[idx]))/5),"DETECTED",
                   upper_boundary=float(h.iloc[max(0,idx-15):idx+1].max()),lower_boundary=float(l.iloc[max(0,idx-15):idx+1].min()))
    pattern_confirmed,pattern_invalidated=_resolve_pattern_lifecycles(events,bars,c)
    for name,values in pattern_values.items(): out[f"pattern_{prefix}_{name.lower()}"]=values
    out[f"pattern_{prefix}_event_count"]=pattern_count
    # Multiple patterns can resolve on one bar, but multiplicity is not a
    # meaningful ordinal magnitude for the model. Preserve only causal direction.
    out[f"pattern_{prefix}_confirmed_transition"]=np.sign(pattern_confirmed)
    out[f"pattern_{prefix}_invalidated_transition"]=np.sign(pattern_invalidated)

    bullish_fvg=(l>h.shift(2)); bearish_fvg=(h<l.shift(2)); fvg=bullish_fvg.astype(float)-bearish_fvg.astype(float)
    # A bounded 240-bar lifecycle prevents stale gaps from creating unbounded
    # replay work. Expiry is an observed state transition, never a retroactive fill.
    open_gaps: list[tuple[int,float,float,float,bool]]=[]; open_count=np.zeros(len(bars)); fill_signal=np.zeros(len(bars)); partial_fill=np.zeros(len(bars))
    for idx in range(len(bars)):
        still_open=[]
        for born,direction,lower_bound,upper_bound,was_partial in open_gaps:
            if idx-born>240:
                _event(events,timeframe,"LIQUIDITY","FAIR_VALUE_GAP_EXPIRED",bars.at[idx,"bar_close_utc"],direction,status="EXPIRED_UNFILLED")
                continue
            filled=(direction>0 and l.iloc[idx]<=lower_bound) or (direction<0 and h.iloc[idx]>=upper_bound)
            if filled:
                fill_signal[idx]+=direction
                _event(events,timeframe,"LIQUIDITY","FAIR_VALUE_GAP_FILLED",bars.at[idx,"bar_close_utc"],direction)
            else:
                is_partial=(direction>0 and l.iloc[idx]<upper_bound) or (direction<0 and h.iloc[idx]>lower_bound)
                if is_partial and not was_partial:
                    partial_fill[idx]+=direction
                    _event(events,timeframe,"LIQUIDITY","FAIR_VALUE_GAP_PARTIALLY_FILLED",bars.at[idx,"bar_close_utc"],direction,status="PARTIALLY_FILLED")
                still_open.append((born,direction,lower_bound,upper_bound,was_partial or is_partial))
        open_gaps=still_open
        if bool(bullish_fvg.iloc[idx]):
            open_gaps.append((idx,1.0,float(h.iloc[idx-2]),float(l.iloc[idx]),False))
        elif bool(bearish_fvg.iloc[idx]):
            open_gaps.append((idx,-1.0,float(h.iloc[idx]),float(l.iloc[idx-2]),False))
        if fvg.iloc[idx] != 0:
            size=abs((l.iloc[idx]-h.iloc[idx-2]) if fvg.iloc[idx]>0 else (l.iloc[idx-2]-h.iloc[idx]))
            _event(events,timeframe,"LIQUIDITY","FAIR_VALUE_GAP",bars.at[idx,"bar_close_utc"],float(fvg.iloc[idx]),min(1,float(size/max(atr.iloc[idx],1e-12))),"OPEN")
            _event(events,timeframe,"LIQUIDITY","BULLISH_FVG" if fvg.iloc[idx]>0 else "BEARISH_FVG",
                   bars.at[idx,"bar_close_utc"],float(fvg.iloc[idx]),min(1,float(size/max(atr.iloc[idx],1e-12))),"OPEN")
        open_count[idx]=len(open_gaps)
    out[f"liquidity_{prefix}_fvg_new"]=fvg
    out[f"liquidity_{prefix}_fvg_fill"]=fill_signal
    out[f"liquidity_{prefix}_fvg_partial_fill"]=partial_fill
    out[f"liquidity_{prefix}_fvg_open_count"]=open_count
    out[f"liquidity_{prefix}_equal_high"]=equal_high; out[f"liquidity_{prefix}_equal_low"]=equal_low
    out[f"liquidity_{prefix}_sweep"]=sweep
    out[f"liquidity_{prefix}_nearest_cluster_distance_atr"]=nearest_liquidity
    out[f"support_resistance_{prefix}_support_distance_atr"]=(c-sr_support)/atr.replace(0,np.nan)
    out[f"support_resistance_{prefix}_resistance_distance_atr"]=(sr_resistance-c)/atr.replace(0,np.nan)
    out[f"support_resistance_{prefix}_ema20_distance_atr"]=(c-ema20)/atr.replace(0,np.nan)
    out[f"support_resistance_{prefix}_ema50_distance_atr"]=(c-ema50)/atr.replace(0,np.nan)
    out[f"support_resistance_{prefix}_fib_nearest_distance_atr"]=fib_distance
    out[f"support_resistance_{prefix}_support_touch_count"]=support_touches
    out[f"support_resistance_{prefix}_resistance_touch_count"]=resistance_touches
    out[f"support_resistance_{prefix}_support_break"]=support_break
    out[f"support_resistance_{prefix}_resistance_break"]=resistance_break
    day=bars["bar_close_utc"].dt.floor("D")
    prev_day_high=h.groupby(day).transform("max").groupby(day).first().shift(1); prev_day_low=l.groupby(day).transform("min").groupby(day).first().shift(1)
    out[f"support_resistance_{prefix}_previous_day_high_distance_atr"]=(day.map(prev_day_high)-c)/atr.replace(0,np.nan)
    out[f"support_resistance_{prefix}_previous_day_low_distance_atr"]=(c-day.map(prev_day_low))/atr.replace(0,np.nan)
    week=day-pd.to_timedelta(bars["bar_close_utc"].dt.dayofweek,unit="D")
    week_high=h.groupby(week).max().shift(1); week_low=l.groupby(week).min().shift(1)
    out[f"support_resistance_{prefix}_previous_week_high_distance_atr"]=(week.map(week_high)-c)/atr.replace(0,np.nan)
    out[f"support_resistance_{prefix}_previous_week_low_distance_atr"]=(c-week.map(week_low))/atr.replace(0,np.nan)
    vol=atr/c; vol_rank=vol.rolling(288,min_periods=30).rank(pct=True)
    out[f"volatility_{prefix}_atr_fraction"]=vol
    out[f"volatility_{prefix}_rank"]=vol_rank
    out[f"volatility_{prefix}_low"]=(vol_rank<=.20).astype(float)
    out[f"volatility_{prefix}_normal"]=((vol_rank>.20)&(vol_rank<.80)).astype(float)
    out[f"volatility_{prefix}_high"]=((vol_rank>=.80)&(vol_rank<.95)).astype(float)
    out[f"volatility_{prefix}_extreme"]=(vol_rank>=.95).astype(float)
    out[f"regime_{prefix}_trend"]=np.sign(ema20-ema50)
    volume=pd.to_numeric(bars.get("tick_volume",pd.Series(0,index=bars.index)),errors="coerce")
    out[f"regime_{prefix}_volume_ratio"]=volume/volume.rolling(24,min_periods=5).mean().replace(0,np.nan)
    out[f"regime_{prefix}_range_flag"]=(out[f"technical_{prefix}_ema20_50_gap_atr"].abs()<.25).astype(float)
    trend_code=out[f"regime_{prefix}_trend"]; range_flag=out[f"regime_{prefix}_range_flag"]
    transition=trend_code.ne(trend_code.shift(1)).astype(float)
    out[f"regime_{prefix}_transition"]=transition
    out[f"regime_{prefix}_trending_up"]=((trend_code>0)&range_flag.eq(0)).astype(float)
    out[f"regime_{prefix}_trending_down"]=((trend_code<0)&range_flag.eq(0)).astype(float)
    out[f"regime_{prefix}_ranging"]=range_flag
    out[f"market_regime_{prefix}"]=np.select([vol_rank>=.80,vol_rank<=.20,transition.eq(1),out[f"regime_{prefix}_trending_up"].eq(1),out[f"regime_{prefix}_trending_down"].eq(1)],
        ["HIGH_VOLATILITY","LOW_VOLATILITY","TRANSITION","TRENDING_UP","TRENDING_DOWN"],default="RANGING")
    times=bars["bar_close_utc"]
    active_columns=[]
    for session_name,zone,opens,closes in SESSION_SCHEDULES:
        local=times.dt.tz_convert(zone); minutes=local.dt.hour*60+local.dt.minute
        active=((local.dt.dayofweek<5)&minutes.ge(opens.hour*60+opens.minute)&minutes.lt(closes.hour*60+closes.minute)).astype(float)
        column=f"session_{prefix}_{session_name.lower()}"; out[column]=active; active_columns.append(column)
    out[f"session_{prefix}_overlap_count"]=out[active_columns].sum(axis=1)
    out[f"session_{prefix}_day_of_week"]=times.dt.dayofweek.astype(float)
    return out, pd.DataFrame(events)


def build_replay(decisions: pd.Series, root: Path = CONTINUOUS_DIR) -> ReplayResult:
    decision_frame=pd.DataFrame({"decision_timestamp_utc":_utc(decisions)}).sort_values("decision_timestamp_utc")
    merged=decision_frame; event_frames=[]
    for timeframe in AVAILABLE_TIMEFRAMES:
        path=root/TIMEFRAMES[timeframe].filename
        if not path.exists(): continue
        features,events=replay_timeframe(pd.read_parquet(path),timeframe)
        merged=pd.merge_asof(merged.sort_values("decision_timestamp_utc"),features.sort_values("available_at_utc"),left_on="decision_timestamp_utc",right_on="available_at_utc",direction="backward",allow_exact_matches=True)
        merged=merged.rename(columns={"available_at_utc":f"{timeframe.lower()}_engine_available_at_utc"})
        event_frames.append(events)
    availability=[c for c in merged if c.endswith("_engine_available_at_utc")]
    violations=sum(int((pd.to_datetime(merged[c],utc=True)>merged["decision_timestamp_utc"]).fillna(False).sum()) for c in availability)
    if violations: raise ValueError(f"Causal replay availability violations: {violations}")
    merged=merged.copy()
    engine_prefixes={"technical":"technical_","price_action":"price_action_","chart_patterns":"pattern_",
        "structure":"structure_","liquidity":"liquidity_","support_resistance":"support_resistance_",
        "volatility":"volatility_","session":"session_","regime":"regime_"}
    for timeframe in (*AVAILABLE_TIMEFRAMES,"H4"):
        tf=timeframe.lower()
        for engine,prefix in engine_prefixes.items():
            columns=[name for name in merged if name.startswith(f"{prefix}{tf}_")]
            if not columns:
                merged[f"availability_{engine}_{tf}_missing_mask"]=1
                merged[f"availability_{engine}_{tf}_partial_mask"]=0
                continue
            missing=merged[columns].isna()
            merged[f"availability_{engine}_{tf}_missing_mask"]=missing.all(axis=1).astype("int8")
            merged[f"availability_{engine}_{tf}_partial_mask"]=(missing.any(axis=1)&~missing.all(axis=1)).astype("int8")
    merged["contract_version"]=CONTRACT_VERSION
    audit=audit_bar_history(root)
    events=pd.concat(event_frames,ignore_index=True) if event_frames else pd.DataFrame()
    if not events.empty:
        start=decision_frame["decision_timestamp_utc"].min(); end=decision_frame["decision_timestamp_utc"].max()
        detected=pd.to_datetime(events["detected_at_utc"],utc=True)
        keep=detected.between(start,end,inclusive="both")
        if "origin_detected_at_utc" in events:
            origin=pd.to_datetime(events["origin_detected_at_utc"],utc=True,errors="coerce")
            keep&=origin.isna()|origin.ge(start)
        events=events.loc[keep].reset_index(drop=True)
    return ReplayResult(merged,events,audit)
