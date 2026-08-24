"""CLI for a V0.6B advisory cycle or status display."""
from __future__ import annotations

import argparse
import json

from src.fusion.v06b_contract import LATEST_ADVISORY_FILE
from src.fusion.v06b_engine import run_fusion_cycle


def print_advisory(payload: dict[str, object]) -> None:
    print("MARKETFUSION V0.6B RISK FUSION + ADVISORY")
    for key in ("status", "generated_at_utc", "decision_timestamp_utc", "action", "confidence", "decision_gate", "next_reassessment_utc"):
        print(f"{key}: {payload.get(key)}")
    market = payload.get("market") or {}
    event = payload.get("event") or {}
    print(f"session: {market.get('session') if isinstance(market, dict) else None}")
    print(f"event_risk: {event.get('status') if isinstance(event, dict) else None}")
    print(f"blocking_reasons: {', '.join(payload.get('blocking_reason_codes') or []) or 'NONE'}")
    print("manual_confirmation_required: true")
    print("trading_enabled: false")
    print(f"V06B_STATUS: {payload.get('status')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        payload = json.loads(LATEST_ADVISORY_FILE.read_text(encoding="utf-8")) if LATEST_ADVISORY_FILE.exists() else {"status": "NOT_RUN", "action": "WAIT", "confidence": "VERY_LOW"}
    else:
        payload = run_fusion_cycle()
    print_advisory(payload)


if __name__ == "__main__":
    main()
