"""Run MarketFusion V0.5A as a local read-only MT5 continuous collector."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.marketdata.mt5_continuous_store import (
    Mt5ReadOnlySession,
    append_quote_partition,
    read_bar_store,
    sync_timeframe,
)
from src.marketdata.v05a_contract import (
    CONTINUOUS_DIR,
    DEFAULT_OVERLAP_BARS,
    DEFAULT_POLL_SECONDS,
    FEATURE_FILE,
    STATUS_FILE,
    SYMBOL,
    TIMEFRAMES,
)
from src.marketdata.v05a_dataset import build_and_save_dataset, load_continuous_frames
from src.marketdata.v05a_quality import build_quality_report, write_quality_report


class V05AIntegrityError(RuntimeError):
    pass


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    temporary.replace(path)


def _read_status() -> dict[str, object]:
    if not STATUS_FILE.exists():
        return {}
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def run_cycle(session: Mt5ReadOnlySession, overlap_bars: int = DEFAULT_OVERLAP_BARS) -> dict[str, object]:
    captured = datetime.now(timezone.utc)
    quote = session.latest_quote(captured)
    quote_path = append_quote_partition(quote)

    sync_results: dict[str, dict[str, object]] = {}
    for label, spec in TIMEFRAMES.items():
        result = sync_timeframe(session, spec, captured, overlap_bars)
        sync_results[label] = result

    conflict_rows = sum(int(result["conflict_rows"]) for result in sync_results.values())
    if conflict_rows:
        payload = {
            "status": "FAIL_IMMUTABLE_BAR_CONFLICT",
            "captured_at_utc": captured.isoformat(),
            "symbol": SYMBOL,
            "conflict_rows": conflict_rows,
            "timeframes": sync_results,
        }
        _atomic_json(payload, STATUS_FILE)
        raise V05AIntegrityError(
            f"V0.5A detected {conflict_rows} previously-observed bar conflicts; original history was preserved and conflicts were quarantined"
        )

    previous_status = _read_status()
    latest_m5 = sync_results["M5"]["last_bar_close_utc"]
    rebuild = (
        not FEATURE_FILE.exists()
        or previous_status.get("last_dataset_decision_utc") != latest_m5
        or int(sync_results["M5"]["new_rows"]) > 0
    )
    if rebuild:
        dataset = build_and_save_dataset()
    else:
        dataset = pd.read_parquet(FEATURE_FILE, engine="pyarrow")
        dataset["decision_timestamp_utc"] = pd.to_datetime(dataset["decision_timestamp_utc"], utc=True)

    frames = load_continuous_frames()
    quality = build_quality_report(frames, dataset, pd.Timestamp(captured))
    write_quality_report(quality)
    if not quality.passed:
        payload = {
            "status": "FAIL_QUALITY_GATE",
            "captured_at_utc": captured.isoformat(),
            "symbol": SYMBOL,
            "quality_failures": quality.failures,
            "timeframes": sync_results,
        }
        _atomic_json(payload, STATUS_FILE)
        raise V05AIntegrityError("V0.5A quality gate failed: " + "; ".join(quality.failures))

    feature_complete = int(dataset["feature_complete"].fillna(False).astype(bool).sum())
    label_counts = {
        f"{minutes}m": int(dataset[f"outcome_future_return_{minutes}m"].notna().sum())
        for minutes in (15, 60, 240)
    }
    payload = {
        "status": "PASS_RUNNING" if quality.passed else "FAIL",
        "captured_at_utc": captured.isoformat(),
        "symbol": SYMBOL,
        "read_only": True,
        "polling_note": "collection only; model retraining is outside V0.5A",
        "quote_partition": str(quote_path),
        "latest_bid": float(quote["bid"].iloc[0]),
        "latest_ask": float(quote["ask"].iloc[0]),
        "latest_spread_points": float(quote["spread_points"].iloc[0]),
        "timeframes": sync_results,
        "dataset_rows": len(dataset),
        "feature_complete_rows": feature_complete,
        "matured_label_rows": label_counts,
        "last_dataset_decision_utc": (
            pd.Timestamp(dataset["decision_timestamp_utc"].iloc[-1]).isoformat() if len(dataset) else None
        ),
        "dataset_rebuilt_this_cycle": rebuild,
        "quality_status": "PASS",
    }
    _atomic_json(payload, STATUS_FILE)
    return payload


def print_summary(payload: dict[str, object]) -> None:
    print("MARKETFUSION V0.5A CONTINUOUS MARKET DATA")
    print(f"status: {payload['status']}")
    print(f"captured_at_utc: {payload['captured_at_utc']}")
    print(f"symbol: {payload['symbol']}")
    print(f"spread_points: {payload['latest_spread_points']}")
    timeframes = payload["timeframes"]
    for label in TIMEFRAMES:
        result = timeframes[label]
        print(
            f"{label}: rows={result['rows_after']} new={result['new_rows']} conflicts={result['conflict_rows']} "
            f"last_close={result['last_bar_close_utc']}"
        )
    print(f"dataset_rows: {payload['dataset_rows']}")
    print(f"feature_complete_rows: {payload['feature_complete_rows']}")
    print(f"matured_label_rows: {payload['matured_label_rows']}")
    print(f"dataset_rebuilt_this_cycle: {payload['dataset_rebuilt_this_cycle']}")
    print("trading: DISABLED / NOT IMPLEMENTED")
    print("V05A_STATUS: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run one collection/audit cycle and exit")
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--overlap-bars", type=int, default=DEFAULT_OVERLAP_BARS)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    if args.overlap_bars < 1:
        parser.error("--overlap-bars must be at least 1")

    CONTINUOUS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with Mt5ReadOnlySession(SYMBOL) as session:
            while True:
                payload = run_cycle(session, args.overlap_bars)
                print_summary(payload)
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("V0.5A stopped by user. Stored history remains intact.")
        return 0
    except V05AIntegrityError as exc:
        print(f"V05A_FAIL_CLOSED: {exc}")
        return 2
    except (RuntimeError, ValueError, OSError) as exc:
        _atomic_json({
            "status": "FAIL_RUNTIME",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": SYMBOL,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, STATUS_FILE)
        print(f"V05A_RUNTIME_ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
