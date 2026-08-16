"""Download official ECB data with SDMX revision-validity timestamps."""

from __future__ import annotations

import argparse
import io
from datetime import date

import pandas as pd

try:
    from .common import DATA_DIRECTORY, QUARANTINE_DIRECTORY, atomic_parquet, http_session, normalize_schema
except ImportError:  # Support direct execution.
    from common import DATA_DIRECTORY, QUARANTINE_DIRECTORY, atomic_parquet, http_session, normalize_schema


BASE_URL = "https://data-api.ecb.europa.eu/service/data"
OUTPUT_FILE = DATA_DIRECTORY / "ecb_macro.parquet"
QUARANTINE_FILE = QUARANTINE_DIRECTORY / "ecb_unusable_rows.parquet"
SERIES = {
    "ecb_rate": {"flow": "FM", "key": "D.U2.EUR.4F.KR.MRR_RT.LEV", "frequency": "D", "nonrevised": True},
    "euro_hicp": {"flow": "HICP", "key": "M.U2.N.000000.4D0.ANR", "frequency": "M"},
    "euro_core_hicp": {"flow": "HICP", "key": "M.U2.N.XEF000.4D0.ANR", "frequency": "M"},
    "euro_unemployment": {"flow": "LFSI", "key": "M.U2.S.UNEHRT.TOTAL0.15_74.T", "frequency": "M"},
    "euro_gdp_growth": {"flow": "MNA", "key": "Q.Y.I9.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY", "frequency": "Q"},
    "euro_2y_yield": {"flow": "FM", "key": "M.U2.EUR.4F.BB.U2_2Y.YLD", "frequency": "M"},
    "euro_10y_yield": {"flow": "FM", "key": "M.U2.EUR.4F.BB.U2_10Y.YLD", "frequency": "M"},
}


def parse_period(value: str, frequency: str) -> pd.Timestamp:
    if frequency == "M":
        return pd.Timestamp(f"{value}-01", tz="UTC")
    if frequency == "Q":
        year, quarter = value.split("-Q")
        return pd.Timestamp(year=int(year), month=(int(quarter) - 1) * 3 + 1, day=1, tz="UTC")
    return pd.to_datetime(value, utc=True)


def fetch_series(name: str, spec: dict, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    url = f"{BASE_URL}/{spec['flow']}/{spec['key']}"
    response = http_session().get(
        url,
        params={
            "startPeriod": start,
            "endPeriod": end,
            "format": "csvdata",
            "includeHistory": "true",
        },
        timeout=120,
    )
    response.raise_for_status()
    raw = pd.read_csv(io.StringIO(response.text), low_memory=False)
    required = {"TIME_PERIOD", "OBS_VALUE"}
    if not required.issubset(raw.columns):
        raise ValueError(f"ECB {name} response missing columns: {sorted(required.difference(raw.columns))}")
    safe_rows: list[dict] = []
    quarantine_rows: list[dict] = []
    for record in raw.to_dict("records"):
        observation_period = parse_period(str(record["TIME_PERIOD"]), spec["frequency"])
        action = str(record.get("ACTION", "Replace"))
        valid_from = pd.to_datetime(record.get("VALID_FROM"), utc=True, errors="coerce")
        nonrevised = bool(spec.get("nonrevised"))
        if nonrevised:
            available = observation_period
            vintage = observation_period
            method = "effective_date_nonrevised_policy_rate"
        else:
            available = valid_from
            vintage = valid_from
            method = "ecb_sdmx_valid_from"
        row = {
            "series_id": name,
            "source": "ECB",
            "observation_period": observation_period,
            "available_from_utc": available,
            "value": record.get("OBS_VALUE"),
            "vintage_date": vintage,
            "frequency": spec["frequency"],
            "source_series_id": f"{spec['flow']}.{spec['key']}",
            "valid_to": record.get("VALID_TO"),
            "action": action,
            "availability_method": method,
            "model_eligible": action.lower() != "delete" and pd.notna(available),
        }
        if row["model_eligible"]:
            safe_rows.append(row)
        else:
            row["quarantine_reason"] = "Delete action or missing VALID_FROM"
            quarantine_rows.append(row)
    return pd.DataFrame(safe_rows), pd.DataFrame(quarantine_rows)


def download(start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    end = end or date.today().isoformat()
    safe_frames = []
    quarantine_frames = []
    failures = []
    for name, spec in SERIES.items():
        try:
            safe, quarantine = fetch_series(name, spec, start, end)
            safe_frames.append(safe)
            if not quarantine.empty:
                quarantine_frames.append(quarantine)
            print(f"ECB {name}: {len(safe):,} safe rows, {len(quarantine):,} quarantined")
        except Exception as exc:  # Continue so one unavailable ECB series cannot erase valid downloads.
            failures.append(f"{name}: {exc}")
            print(f"WARNING ECB {name}: {exc}")
    if not safe_frames:
        raise RuntimeError("All ECB series downloads failed: " + "; ".join(failures))
    result = normalize_schema(pd.concat(safe_frames, ignore_index=True))
    result = result.dropna(subset=["value"]).drop_duplicates(
        ["series_id", "observation_period", "available_from_utc", "value"], keep="last"
    )
    atomic_parquet(result, OUTPUT_FILE)
    if quarantine_frames:
        quarantine = normalize_schema(pd.concat(quarantine_frames, ignore_index=True))
        atomic_parquet(quarantine, QUARANTINE_FILE)
    print(f"Saved {len(result):,} safe ECB rows: {OUTPUT_FILE}")
    if failures:
        print("PARTIAL_DOWNLOAD_WARNINGS: " + "; ".join(failures))
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    try:
        download(args.start, args.end)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
