"""CLI and continuous loop for the MarketFusion V0.6 unified runtime."""
from __future__ import annotations

import argparse
import json
import os
import time

from src.runtime.v06c_contract import DEFAULT_POLL_SECONDS, LATEST_STATE_FILE, RUNTIME_LOCK_FILE
from src.runtime.v06c_engine import run_runtime_cycle


def print_state(state: dict[str, object]) -> None:
    system = state.get("system") or {}
    decision = state.get("decision") or {}
    market = state.get("market") or {}
    event = state.get("event") or {}
    print("MARKETFUSION V0.6 UNIFIED RUNTIME")
    print(f"status: {system.get('status')}")
    print(f"decision: {decision.get('action', 'WAIT')}")
    print(f"confidence: {decision.get('confidence', 'VERY_LOW')}")
    print(f"gate: {decision.get('gate')}")
    print(f"decision_utc: {(decision.get('decision_time') or {}).get('utc')}")
    print(f"decision_asia_colombo: {(decision.get('decision_time') or {}).get('asia_colombo')}")
    print(f"session: {market.get('session')}")
    print(f"event_risk: {event.get('status')}")
    print("trading_enabled: false")
    print("manual_confirmation_required: true")
    print(f"V06_STATUS: {system.get('status')}")


def _pid_running(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _acquire_runtime_lock() -> bool:
    RUNTIME_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if RUNTIME_LOCK_FILE.exists():
        try:
            existing = int(RUNTIME_LOCK_FILE.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            existing = -1
        if existing > 0 and _pid_running(existing):
            return False
        RUNTIME_LOCK_FILE.unlink(missing_ok=True)
    try:
        descriptor = os.open(RUNTIME_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(str(os.getpid()))
    return True


def _release_runtime_lock() -> None:
    try:
        if int(RUNTIME_LOCK_FILE.read_text(encoding="ascii").strip()) == os.getpid():
            RUNTIME_LOCK_FILE.unlink(missing_ok=True)
    except (OSError, ValueError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    args = parser.parse_args()
    if args.status:
        state = json.loads(LATEST_STATE_FILE.read_text(encoding="utf-8")) if LATEST_STATE_FILE.exists() else {"system": {"status": "NOT_RUN"}, "decision": {"action": "WAIT", "confidence": "VERY_LOW"}}
        print_state(state)
        return
    if not args.continuous:
        print_state(run_runtime_cycle())
        return
    if not _acquire_runtime_lock():
        print("MarketFusion V0.6 runtime is already active; duplicate start rejected.")
        return
    interval = max(10, int(args.interval_seconds))
    print(f"Starting MarketFusion V0.6 unified runtime every {interval}s. Ctrl+C to stop.")
    print("Research-only advisory runtime; trading execution is disabled.")
    try:
        while True:
            print_state(run_runtime_cycle())
            time.sleep(interval)
    except KeyboardInterrupt:
        print("MarketFusion runtime stopped safely. No trade was sent. History remains intact.")
    finally:
        _release_runtime_lock()


if __name__ == "__main__":
    main()
