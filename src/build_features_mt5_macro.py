"""As-of join validated macro vintages onto the clean MT5 H1 feature set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_features_mt5 import FEATURES_MT5, OUTPUT_FILE as PRICE_FEATURE_FILE
from macro.common import REPORT_DIRECTORY
from macro.validate_macro_data import SAFE_OUTPUT, validate


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "data" / "processed" / "eurusd_mt5_macro_features.parquet"
MANIFEST_FILE = REPORT_DIRECTORY / "macro_feature_manifest.json"

BASE_MACRO_FEATURES = [
    "fed_funds_rate", "us_2y_yield", "us_10y_yield", "us_cpi_yoy", "us_core_cpi_yoy",
    "us_unemployment", "us_payroll_growth", "us_gdp_growth", "us_pce", "us_core_pce",
    "us_retail_sales_yoy", "ecb_rate", "euro_hicp", "euro_core_hicp",
    "euro_unemployment", "euro_gdp_growth", "euro_2y_yield", "euro_10y_yield",
]
CHANGE_FEATURE_BASES = [
    "fed_funds_rate", "us_2y_yield", "us_10y_yield", "us_cpi_yoy", "us_core_cpi_yoy",
    "us_unemployment", "us_payroll_growth", "ecb_rate", "euro_hicp", "euro_core_hicp",
    "euro_unemployment", "euro_2y_yield", "euro_10y_yield",
]
DERIVED_MACRO_FEATURES = [
    "yield_curve_10y_2y", "euro_yield_curve_10y_2y", "rate_differential", "yield_differential",
]
CHANGE_FEATURES = [
    f"{feature}_change_{label}"
    for feature in CHANGE_FEATURE_BASES
    for label in ("1d", "1w", "1m")
]
AVAILABILITY_FEATURES = [f"{feature}_available" for feature in BASE_MACRO_FEATURES]
MACRO_FEATURES = BASE_MACRO_FEATURES + DERIVED_MACRO_FEATURES + CHANGE_FEATURES + AVAILABILITY_FEATURES + [
    "macro_release_day"
]

# Release-day regimes should represent discrete macro/policy events, not daily
# market context such as Treasury yields or effective overnight rates.
RELEASE_EVENT_SERIES = {
    "us_cpi_yoy", "us_core_cpi_yoy", "us_unemployment", "us_payroll_growth",
    "us_gdp_growth", "us_pce", "us_core_pce", "us_retail_sales_yoy",
    "ecb_rate", "euro_hicp", "euro_core_hicp", "euro_unemployment", "euro_gdp_growth",
}
POLICY_CHANGE_SERIES = {"ecb_rate"}


def latest_observation_events(series: pd.DataFrame) -> pd.DataFrame:
    """Create an event stream of the latest observation known after each vintage update."""
    series = series.sort_values(["available_from_utc", "observation_period"]).copy()
    state: dict[pd.Timestamp, tuple[float, pd.Timestamp]] = {}
    events = []
    previous: tuple[pd.Timestamp, float] | None = None
    for available, updates in series.groupby("available_from_utc", sort=True):
        for row in updates.itertuples():
            state[row.observation_period] = (float(row.value), available)
        latest_period = max(state)
        value, source_available = state[latest_period]
        current = (latest_period, value)
        if previous != current:
            events.append(
                {
                    "available_from_utc": available,
                    "observation_period": latest_period,
                    "value": value,
                    "source_available_from_utc": source_available,
                }
            )
            previous = current
    return pd.DataFrame(events)


def values_asof(events: pd.DataFrame, decisions: pd.Series) -> pd.DataFrame:
    left = pd.DataFrame({"decision": pd.to_datetime(decisions, utc=True)}).sort_values("decision")
    right = events.sort_values("available_from_utc")
    return pd.merge_asof(
        left,
        right,
        left_on="decision",
        right_on="available_from_utc",
        direction="backward",
        allow_exact_matches=True,
    ).sort_index()


def build_features() -> pd.DataFrame:
    result = validate(write_report=True, require_complete=True)
    if not result.passed:
        raise RuntimeError("Complete macro validation failed; feature generation blocked: " + "; ".join(result.failures))
    if not PRICE_FEATURE_FILE.exists():
        raise FileNotFoundError(f"Clean MT5 features not found: {PRICE_FEATURE_FILE}")

    market = pd.read_parquet(PRICE_FEATURE_FILE, engine="pyarrow").sort_values("decision_timestamp").reset_index(drop=True)
    market["decision_timestamp"] = pd.to_datetime(market["decision_timestamp"], utc=True)
    macro = pd.read_parquet(SAFE_OUTPUT, engine="pyarrow")
    for column in ("observation_period", "available_from_utc", "vintage_date"):
        macro[column] = pd.to_datetime(macro[column], utc=True)
    event_streams: dict[str, pd.DataFrame] = {}

    for feature in BASE_MACRO_FEATURES:
        source = macro[macro["series_id"] == feature]
        if source.empty:
            market[feature] = np.nan
            market[f"{feature}__available_from_utc"] = pd.NaT
            market[f"{feature}_available"] = 0
            continue
        events = latest_observation_events(source)
        event_streams[feature] = events
        joined = values_asof(events, market["decision_timestamp"])
        market[feature] = joined["value"].to_numpy()
        market[f"{feature}__available_from_utc"] = joined["available_from_utc"].to_numpy()
        market[f"{feature}_available"] = market[feature].notna().astype("int8")
        leak = (
            pd.to_datetime(market[f"{feature}__available_from_utc"], utc=True)
            > market["decision_timestamp"]
        ).fillna(False)
        if leak.any():
            raise AssertionError(f"As-of join leaked {int(leak.sum())} future {feature} observations")

    for feature in CHANGE_FEATURE_BASES:
        events = event_streams.get(feature)
        for label, delta in (("1d", pd.Timedelta(days=1)), ("1w", pd.Timedelta(days=7)), ("1m", pd.Timedelta(days=30))):
            output = f"{feature}_change_{label}"
            if events is None:
                market[output] = np.nan
            else:
                prior = values_asof(events, market["decision_timestamp"] - delta)["value"].to_numpy()
                market[output] = market[feature].to_numpy() - prior

    market["yield_curve_10y_2y"] = market["us_10y_yield"] - market["us_2y_yield"]
    market["euro_yield_curve_10y_2y"] = market["euro_10y_yield"] - market["euro_2y_yield"]
    market["rate_differential"] = market["fed_funds_rate"] - market["ecb_rate"]
    market["yield_differential"] = market["us_10y_yield"] - market["euro_10y_yield"]

    release_timestamps = []
    release_source = macro[macro["series_id"].isin(RELEASE_EVENT_SERIES)].copy()
    for series_id, source in release_source.groupby("series_id"):
        source = source.sort_values(["available_from_utc", "observation_period"])
        if series_id in POLICY_CHANGE_SERIES:
            source = source[source["value"].ne(source["value"].shift())]
        release_timestamps.extend(source["available_from_utc"].dropna().tolist())
    release_dates = pd.DatetimeIndex(release_timestamps).normalize().unique()
    market["macro_release_day"] = market["decision_timestamp"].dt.normalize().isin(release_dates).astype("int8")
    release_fraction = float(market["macro_release_day"].mean())
    if release_fraction > 0.75:
        raise ValueError(
            f"Macro release-day flag covers {release_fraction:.1%} of market rows; "
            "daily context appears to have leaked into the event regime definition"
        )

    missing_columns = sorted(set(FEATURES_MT5 + MACRO_FEATURES).difference(market.columns))
    if missing_columns:
        raise AssertionError("Missing model features: " + ", ".join(missing_columns))
    finite_or_nan = market[FEATURES_MT5 + MACRO_FEATURES].replace([np.inf, -np.inf], np.nan)
    if finite_or_nan[FEATURES_MT5].isna().any().any():
        raise ValueError("Price features unexpectedly contain missing/non-finite values")
    market[FEATURES_MT5 + MACRO_FEATURES] = finite_or_nan

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_FILE.with_suffix(".tmp.parquet")
    market.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(OUTPUT_FILE)
    manifest = {
        "price_features": FEATURES_MT5,
        "macro_features": MACRO_FEATURES,
        "audit_timestamp_suffix": "__available_from_utc",
        "safe_macro_input": str(SAFE_OUTPUT),
        "rule": "available_from_utc <= decision_timestamp",
        "requires_complete_macro": True,
        "release_event_series": sorted(RELEASE_EVENT_SERIES),
        "release_day_definition": "non-daily macro/policy availability day; daily yield/rate context excluded",
    }
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Complete macro validation: PASS")
    print(f"Market rows: {len(market):,}")
    print(f"Macro model features: {len(MACRO_FEATURES)}")
    print(
        f"Rows on macro release days: {int(market['macro_release_day'].sum()):,} "
        f"({release_fraction:.1%})"
    )
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Manifest: {MANIFEST_FILE}")
    return market


if __name__ == "__main__":
    try:
        build_features()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
