"""CLI/runtime loop for MarketFusion V0.6A shadow inference."""
from __future__ import annotations

import argparse
import json
import time

from src.inference.v06a_contract import DEFAULT_POLL_SECONDS, LATEST_STATUS_FILE
from src.inference.v06a_engine import run_shadow_cycle


def _print_payload(payload: dict[str, object]) -> None:
    print("MARKETFUSION V0.6A REAL-TIME SHADOW INFERENCE")
    print(f"status: {payload.get('status')}")
    print(f"captured_at_utc: {payload.get('captured_at_utc')}")
    print(f"decision_timestamp_utc: {payload.get('decision_timestamp_utc')}")
    print(f"market_age_minutes: {payload.get('market_age_minutes')}")
    print(f"market_fresh: {payload.get('market_fresh')}")
    event = payload.get("event_guard") or {}
    print(f"target_event_guard: {event.get('status') if isinstance(event, dict) else 'NA'}")
    print("trading: DISABLED / SHADOW ONLY")
    for horizon in (15, 60, 240):
        result = (payload.get("horizons") or {}).get(str(horizon), {})
        print("")
        print(f"{horizon}m")
        print(f"model: {result.get('model_id') or 'NONE'}")
        print(f"model_status: {result.get('model_status')}")
        print(f"DOWN: {result.get('prob_down')}")
        print(f"NEUTRAL: {result.get('prob_neutral')}")
        print(f"UP: {result.get('prob_up')}")
        print(f"shadow_direction: {result.get('shadow_direction', 'WAIT')}")
        print(f"gate: {result.get('decision_gate')}")
    print("")
    print(f"V06A_STATUS: {payload.get('status')}")


def show_status() -> None:
    if not LATEST_STATUS_FILE.exists():
        print("MARKETFUSION V0.6A")
        print("status: NOT_RUN")
        print("trading: DISABLED")
        return
    payload = json.loads(LATEST_STATUS_FILE.read_text(encoding="utf-8"))
    _print_payload(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    args = parser.parse_args()
    if args.status:
        show_status()
        return
    if not args.continuous:
        _print_payload(run_shadow_cycle())
        return
    interval = max(10, int(args.interval_seconds))
    print(f"Starting V0.6A shadow inference every {interval}s. Ctrl+C to stop.")
    print("Research-only: this process cannot place trades.")
    try:
        while True:
            _print_payload(run_shadow_cycle())
            time.sleep(interval)
    except KeyboardInterrupt:
        print("V0.6A stopped by user. Shadow history remains intact; no trade was sent.")


if __name__ == "__main__":
    main()
