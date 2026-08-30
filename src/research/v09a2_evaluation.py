"""Chronological ablation, event evidence, and redundancy analysis for V0.9A.2."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.learning.v05c_contract import CLASS_LABELS
from src.learning.v05c_models import evaluate_probabilities
from src.learning.v05c_walk_forward import purged_walk_forward
from src.research.v09a2_contract import (
    ABLATION_SEQUENCE, CHART_PATTERNS, FEATURE_GROUPS, MIN_EVENT_SAMPLES,
    MODEL_FAMILY, grade_evidence,
)


def ablation_features(frame: pd.DataFrame, label: str) -> tuple[str, ...]:
    baseline = [name for name in FEATURE_GROUPS["BASELINE"] if name in frame]
    if label == "BASELINE":
        return tuple(baseline)
    groups = list(FEATURE_GROUPS) if label == "+ALL_V09" else [label.lstrip("+")]
    derived: list[str] = []
    for group in groups:
        if group == "BASELINE":
            continue
        prefixes = FEATURE_GROUPS[group]
        derived.extend(name for name in frame if any(name.startswith(prefix) for prefix in prefixes))
    return tuple(dict.fromkeys([*baseline, *derived]))


def _aligned(model: Any, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    result = np.full((len(x), len(CLASS_LABELS)), 1e-12)
    for index, label in enumerate(model.classes_):
        result[:, CLASS_LABELS.index(int(label))] = raw[:, index]
    return result / result.sum(axis=1, keepdims=True)


def build_research_estimator() -> Pipeline:
    """Same V0.5C family with predeclared bounded research convergence settings."""
    return Pipeline([
        ("imputer",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),
        ("scaler",RobustScaler()),
        ("classifier",LogisticRegression(max_iter=100,tol=0.01,C=0.3,class_weight="balanced",
            random_state=1705,solver="newton-cholesky")),
    ])


def _ece(probability: np.ndarray, target: np.ndarray) -> float:
    confidence = probability.max(axis=1); predicted = probability.argmax(axis=1)
    bins = np.minimum((confidence * 10).astype(int), 9); value = 0.0
    for bin_id in range(10):
        mask = bins == bin_id
        if mask.any():
            value += float(mask.mean()) * abs(float(confidence[mask].mean()) - float((predicted[mask] == target[mask]).mean()))
    return value


def run_ablation(dataset: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Use only expanding, purged V0.5C folds and train-fitted preprocessing."""
    work = dataset.sort_values("decision_timestamp_utc").reset_index(drop=True)
    folds = purged_walk_forward(work["decision_timestamp_utc"], horizon)
    raw_rows: list[dict[str, Any]] = []
    for label in ABLATION_SEQUENCE:
        features = ablation_features(work, label)
        if not features:
            continue
        all_y: list[np.ndarray] = []; all_p: list[np.ndarray] = []; all_r: list[np.ndarray] = []; all_c: list[np.ndarray] = []
        fold_ba: list[float] = []
        for fold in folds:
            train, valid = work.iloc[fold.train_indices], work.iloc[fold.validation_indices]
            model = build_research_estimator()
            model.fit(train.loc[:, list(features)], train["target_class"].astype(int))
            probability = _aligned(model, valid.loc[:, list(features)])
            metrics = evaluate_probabilities(valid["target_class"], probability, valid["raw_future_return"], valid["decision_cost_band"])
            fold_ba.append(float(metrics["balanced_accuracy"]))
            all_y.append(valid["target_class"].to_numpy(int)); all_p.append(probability)
            all_r.append(valid["raw_future_return"].to_numpy(float)); all_c.append(valid["decision_cost_band"].to_numpy(float))
        y=np.concatenate(all_y); probability=np.vstack(all_p); raw=np.concatenate(all_r); cost=np.concatenate(all_c)
        metrics=evaluate_probabilities(y,probability,raw,cost); predicted=np.asarray(CLASS_LABELS)[probability.argmax(axis=1)]
        row: dict[str,Any]={"horizon_minutes":horizon,"ablation":label,"model_family":MODEL_FAMILY,
            "feature_count":len(features),"validation_rows":len(y),"fold_count":len(folds),
            "accuracy":float(accuracy_score(y,predicted)),**metrics,
            "calibration_ece":_ece(probability,y),"fold_ba_mean":float(np.mean(fold_ba)),
            "fold_ba_std":float(np.std(fold_ba)),"fold_ba_worst":float(np.min(fold_ba)),
            "fold_ba_recent":float(fold_ba[-1]),"chronological_only":True,"purge_minutes":horizon,
            "preprocessing_fit_scope":"TRAIN_ONLY"}
        for class_id,name in ((0,"down"),(1,"neutral"),(2,"up")):
            row[f"precision_{name}"]=float(precision_score(y,predicted,labels=[class_id],average="macro",zero_division=0))
            row[f"recall_{name}"]=float(recall_score(y,predicted,labels=[class_id],average="macro",zero_division=0))
            row[f"f1_{name}"]=float(f1_score(y,predicted,labels=[class_id],average="macro",zero_division=0))
        raw_rows.append(row)
    result=pd.DataFrame(raw_rows)
    if result.empty: return result
    base=float(result.loc[result["ablation"].eq("BASELINE"),"balanced_accuracy"].iloc[0])
    base_loss=float(result.loc[result["ablation"].eq("BASELINE"),"log_loss"].iloc[0])
    result["balanced_accuracy_delta_vs_baseline"]=result["balanced_accuracy"]-base
    result["log_loss_delta_vs_baseline"]=result["log_loss"]-base_loss
    result["stability_pass"]=(result["fold_ba_recent"]>=result["fold_ba_mean"]-.03)&(result["fold_ba_worst"]>=base-.05)
    result["evidence_grade"]=[grade_evidence(int(n),float(d),bool(s),float(loss)) for n,d,s,loss in zip(result["validation_rows"],result["balanced_accuracy_delta_vs_baseline"],result["stability_pass"],result["log_loss_delta_vs_baseline"])]
    result.loc[result["ablation"].eq("BASELINE"),"evidence_grade"]="REFERENCE"
    return result


def _session(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(np.select([frame["is_london_ny_overlap"].eq(1),frame["is_london_session"].eq(1),frame["is_new_york_session"].eq(1),frame["is_asia_session"].eq(1)],
                               ["LONDON_NEW_YORK_OVERLAP","LONDON","NEW_YORK","ASIA"],default="OTHER"),index=frame.index)


def evaluate_events(events: pd.DataFrame, horizons: dict[int,pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if events.empty:
        return pd.DataFrame()
    source=events.copy(); source["detected_at_utc"]=pd.to_datetime(source["detected_at_utc"],utc=True)
    for horizon,target in horizons.items():
        fields=["decision_timestamp_utc","target_class","raw_future_return","is_asia_session","is_london_session","is_new_york_session","is_london_ny_overlap"]
        if "regime_m5_volatility_rank" in target:
            fields.append("regime_m5_volatility_rank")
        joined=source.merge(target[fields],left_on="detected_at_utc",right_on="decision_timestamp_utc",how="inner")
        if joined.empty: continue
        joined["session"]=_session(joined); joined["expected_class"]=np.where(joined["direction"]>0,2,np.where(joined["direction"]<0,0,1))
        if "regime_m5_volatility_rank" in joined:
            joined["regime"]=pd.cut(joined["regime_m5_volatility_rank"],[-np.inf,.33,.67,np.inf],labels=["LOW_VOLATILITY","NORMAL_VOLATILITY","HIGH_VOLATILITY"])
        specifications=[("ALL",["event_family","event_name","timeframe"]),("SESSION",["event_family","event_name","timeframe","session"])]
        if "regime" in joined: specifications.append(("REGIME",["event_family","event_name","timeframe","regime"]))
        class_rate=joined["target_class"].value_counts(normalize=True).reindex(CLASS_LABELS,fill_value=0.0)
        specifications=[(label,[*keys,"status"]) for label,keys in specifications]
        for subgroup,keys in specifications:
            records=[]
            for values,group in joined.groupby(keys,dropna=False):
                values=values if isinstance(values,tuple) else (values,); count=len(group); sufficient=count>=MIN_EVENT_SAMPLES
                record={name:value for name,value in zip(keys,values)}
                expected=group["expected_class"].astype(int); baseline=float(expected.map(class_rate).mean())
                accuracy=float((group["target_class"]==expected).mean()); standard_error=float(np.sqrt(max(baseline*(1-baseline),1e-12)/max(count,1)))
                # 4.5 standard errors is deliberately conservative for the large,
                # predeclared family of event/session/regime comparisons.
                delta=accuracy-baseline; lower=delta-4.5*standard_error
                if not sufficient: evidence="INSUFFICIENT_DATA"
                elif subgroup=="ALL" and count>=500 and lower>0 and delta>=.05: evidence="STRONG"
                elif subgroup=="ALL" and count>=300 and lower>0 and delta>=.03: evidence="MODERATE"
                elif lower>0 and delta>=.02: evidence="WEAK"
                else: evidence="NO_EVIDENCE"
                record.update({"subgroup":subgroup,"horizon_minutes":horizon,"sample_count":count,
                    "sample_status":"SUFFICIENT_DESCRIPTIVE_SAMPLE" if sufficient else "INSUFFICIENT_DATA",
                    "directional_accuracy":accuracy if sufficient else np.nan,
                    "baseline_direction_rate":baseline if sufficient else np.nan,
                    "directional_accuracy_delta":delta if sufficient else np.nan,
                    "up_rate":float(group["target_class"].eq(2).mean()) if sufficient else np.nan,
                    "neutral_rate":float(group["target_class"].eq(1).mean()) if sufficient else np.nan,
                    "down_rate":float(group["target_class"].eq(0).mean()) if sufficient else np.nan,
                    "mean_return":float(group["raw_future_return"].mean()) if sufficient else np.nan,
                    "median_return":float(group["raw_future_return"].median()) if sufficient else np.nan,"evidence_grade":evidence,
                    "claim":"DESCRIPTIVE_CONDITIONAL_EVIDENCE_ONLY"})
                records.append(record)
            rows.append(pd.DataFrame(records))
    result=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    # Make absence explicit for every chart pattern instead of silently dropping it.
    existing=set(result.loc[(result.get("event_family")=="CHART_PATTERN")&(result.get("subgroup")=="ALL"),"event_name"]) if not result.empty else set()
    missing=[]
    for name in CHART_PATTERNS:
        if name not in existing:
            for horizon in horizons:
                missing.append({"event_family":"CHART_PATTERN","event_name":name,"timeframe":"ALL","subgroup":"ALL","status":"DETECTED","horizon_minutes":horizon,"sample_count":0,"sample_status":"INSUFFICIENT_DATA","claim":"NO_DETECTIONS_NO_EDGE_CLAIM"})
    return pd.concat([result,pd.DataFrame(missing)],ignore_index=True,sort=False)


def engine_leaderboard(features: pd.DataFrame, horizons: dict[int,pd.DataFrame]) -> pd.DataFrame:
    specifications={
        "TECHNICAL":lambda x: x.filter(regex=r"^technical_.*(rsi14|ema20_50_gap_atr)$").mean(axis=1),
        "PRICE_ACTION":lambda x: x.filter(regex=r"^price_action_.*direction_score$").mean(axis=1),
        "PATTERNS":lambda x: x.filter(regex=r"^pattern_.*_(double|triple|head|inverse|wedge|flag|pennant|triangle|range)").sum(axis=1),
        "STRUCTURE":lambda x: x.filter(regex=r"^structure_.*(state|bos|choch)$").mean(axis=1),
        "LIQUIDITY":lambda x: x.filter(regex=r"^liquidity_.*(fvg_new|sweep)$").mean(axis=1),
        "SUPPORT_RESISTANCE":lambda x: -x.filter(regex=r"^support_resistance_.*(support|resistance)_distance_atr$").mean(axis=1),
        "REGIME":lambda x: x.filter(regex=r"^regime_.*trend$").mean(axis=1),
    }
    rows=[]
    for horizon,target in horizons.items():
        base_fields=["decision_timestamp_utc","target_class","raw_future_return","is_asia_session","is_london_session","is_new_york_session","is_london_ny_overlap"]
        joined=target[base_fields].merge(features,on="decision_timestamp_utc",how="inner")
        joined["session"]=_session(joined)
        joined["regime"]=pd.cut(joined.get("regime_m5_volatility_rank",pd.Series(.5,index=joined.index)),[-np.inf,.33,.67,np.inf],labels=["LOW_VOLATILITY","NORMAL_VOLATILITY","HIGH_VOLATILITY"])
        for engine,builder in specifications.items():
            joined["_signal"]=builder(joined).replace([np.inf,-np.inf],np.nan).fillna(0)
            slices=[("ALL","ALL",joined)]
            slices.extend(("SESSION",str(name),group) for name,group in joined.groupby("session"))
            slices.extend(("REGIME",str(name),group) for name,group in joined.groupby("regime",observed=True))
            for subgroup_type,subgroup_value,group in slices:
                signal=group["_signal"]; acted=signal.ne(0); count=int(acted.sum()); predicted=np.where(signal>0,2,np.where(signal<0,0,1)); sufficient=count>=MIN_EVENT_SAMPLES
                ba=float(np.mean([np.mean(predicted[group["target_class"].eq(k)]==k) for k in CLASS_LABELS if group["target_class"].eq(k).any()])) if sufficient else np.nan
                chronological=[]
                for section in np.array_split(np.arange(len(group)),3):
                    part=group.iloc[section]; part_signal=part["_signal"]; part_pred=np.where(part_signal>0,2,np.where(part_signal<0,0,1))
                    chronological.append(float(np.mean([np.mean(part_pred[part["target_class"].eq(k)]==k) for k in CLASS_LABELS if part["target_class"].eq(k).any()])))
                stable=bool(max(chronological)-min(chronological)<=.08)
                grade="INSUFFICIENT_DATA" if not sufficient else "MODERATE" if ba>=.36 and stable else "WEAK" if ba>=.345 and stable else "NO_EVIDENCE"
                rows.append({"engine":engine,"horizon_minutes":horizon,"subgroup_type":subgroup_type,"subgroup_value":subgroup_value,"rows":len(group),"acted_rows":count,
                    "coverage":float(acted.mean()),"sample_status":"SUFFICIENT_DESCRIPTIVE_SAMPLE" if sufficient else "INSUFFICIENT_DATA",
                    "directional_accuracy_acted":float((predicted[acted]==group.loc[acted,"target_class"]).mean()) if sufficient else np.nan,
                    "balanced_accuracy":ba,"chronological_third_ba_std":float(np.std(chronological)),"stability_pass":stable,
                    "mean_signed_return_acted":float(np.mean(np.sign(signal[acted])*group.loc[acted,"raw_future_return"])) if sufficient else np.nan,
                    "evidence_grade":grade,"standalone":True})
    result=pd.DataFrame(rows)
    return result.sort_values(["horizon_minutes","subgroup_type","subgroup_value","balanced_accuracy"],ascending=[True,True,True,False])


def redundancy_analysis(features: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    derived=[name for name in features if name.startswith(tuple(prefix for group,prefixes in FEATURE_GROUPS.items() if group!="BASELINE" for prefix in prefixes))]
    numeric=features[derived].select_dtypes(include=[np.number]).replace([np.inf,-np.inf],np.nan)
    sample=numeric.tail(min(20_000,len(numeric)))
    corr=sample.corr(method="spearman",min_periods=100)
    rows=[]
    for i,left in enumerate(corr.columns):
        for right in corr.columns[i+1:]:
            value=corr.at[left,right]
            if pd.notna(value) and abs(value)>=.75:
                rows.append({"analysis":"SPEARMAN_PAIR","feature":left,"related_feature":right,"value":float(value),"interpretation":"HIGH_REDUNDANCY" if abs(value)>=.90 else "MODERATE_REDUNDANCY"})
    joined=target[["decision_timestamp_utc","target_class"]].merge(features[["decision_timestamp_utc",*numeric.columns]],on="decision_timestamp_utc",how="inner").tail(20_000)
    x=joined[numeric.columns].replace([np.inf,-np.inf],np.nan).fillna(joined[numeric.columns].median()).fillna(0)
    if len(joined)>=MIN_EVENT_SAMPLES and len(x.columns):
        mi=mutual_info_classif(x,joined["target_class"].astype(int),discrete_features=False,random_state=1705)
        rows.extend({"analysis":"MUTUAL_INFORMATION_15M","feature":name,"related_feature":"target_class","value":float(value),"interpretation":"UNIVARIATE_ASSOCIATION_NOT_CAUSAL_IMPORTANCE"} for name,value in zip(x.columns,mi))
    return pd.DataFrame(rows).sort_values(["analysis","value"],ascending=[True,False]) if rows else pd.DataFrame(columns=["analysis","feature","related_feature","value","interpretation"])
