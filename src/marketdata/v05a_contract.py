"""V0.5A contract for continuous EURUSD market-data capture and causal history growth.

V0.5A is deliberately data infrastructure only. It continuously collects read-only
MT5 market data, stores completed bars, builds point-in-time features, and matures
future-return labels only after their horizons have elapsed. It does not train or
promote a trading model and contains no trading execution path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYMBOL = "EURUSD"
CONTRACT_VERSION = "v0.5a-continuous-market-data-v1"
CONTINUOUS_DIR = ROOT / "data" / "mt5" / "continuous"
FEATURE_FILE = ROOT / "data" / "processed" / "v05a_eurusd_continuous_features.parquet"
STATUS_FILE = CONTINUOUS_DIR / "runtime_status.json"
QUALITY_REPORT = ROOT / "reports" / "v05a_continuous_market_data_quality.txt"
CONFLICT_DIR = CONTINUOUS_DIR / "conflicts"
QUOTE_DIR = CONTINUOUS_DIR / "quotes"

BAR_COLUMNS = [
    "bar_open_utc",
    "bar_close_utc",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread_points",
    "real_volume",
    "first_observed_utc",
    "source",
]

QUOTE_COLUMNS = [
    "captured_at_utc",
    "broker_tick_time_utc",
    "bid",
    "ask",
    "mid",
    "spread_price",
    "spread_points",
    "point_size",
    "source",
]

PREDICTION_HORIZONS_MINUTES = (15, 60, 240)
DECISION_TIMEFRAME = "M5"
STABILIZATION_SECONDS = 5
DEFAULT_POLL_SECONDS = 15
DEFAULT_BOOTSTRAP_DAYS = 30
DEFAULT_OVERLAP_BARS = 8


@dataclass(frozen=True)
class TimeframeSpec:
    label: str
    mt5_attribute: str
    minutes: int
    bootstrap_days: int

    @property
    def duration_seconds(self) -> int:
        return self.minutes * 60

    @property
    def filename(self) -> str:
        return f"{SYMBOL}_{self.label}.parquet"


TIMEFRAMES: dict[str, TimeframeSpec] = {
    "M1": TimeframeSpec("M1", "TIMEFRAME_M1", 1, 14),
    "M5": TimeframeSpec("M5", "TIMEFRAME_M5", 5, 45),
    "M15": TimeframeSpec("M15", "TIMEFRAME_M15", 15, 120),
    "H1": TimeframeSpec("H1", "TIMEFRAME_H1", 60, 365),
}

# Feature fields only. Outcome/label fields must never enter this registry.
FEATURE_COLUMNS = [
    "m1_return_1m", "m1_return_5m", "m1_volatility_15m",
    "m1_spread_points", "m1_tick_volume",
    "m5_return_5m", "m5_return_15m", "m5_return_60m", "m5_return_240m",
    "m5_volatility_15m", "m5_volatility_60m", "m5_volatility_240m",
    "m5_spread_points", "m5_spread_median_60m", "m5_spread_max_60m",
    "m5_tick_volume", "m5_tick_volume_mean_60m",
    "m15_return_15m", "m15_return_60m", "m15_return_240m",
    "m15_volatility_60m", "m15_volatility_240m",
    "h1_return_60m", "h1_return_240m", "h1_volatility_240m", "h1_volatility_1440m",
    "utc_hour", "day_of_week",
    "is_asia_session", "is_london_session", "is_new_york_session", "is_london_ny_overlap",
]

OUTCOME_COLUMNS = [
    item
    for horizon in PREDICTION_HORIZONS_MINUTES
    for item in (
        f"outcome_future_timestamp_{horizon}m",
        f"outcome_future_return_{horizon}m",
        f"outcome_direction_{horizon}m",
        f"outcome_matured_at_utc_{horizon}m",
    )
]
