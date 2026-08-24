"""Write the deterministic V0.7 local-dashboard validation summary."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORT = ROOT / "reports" / "v07_full_validation.txt"

from src.dashboard.v07_api import intelligence, load_state, trading_route_count, validate_state
from src.dashboard.v07_contract import FORBIDDEN_RESPONSE_KEYS


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key).lower() for key in value} | set().union(*(_keys(child) for child in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_keys(child) for child in value), set())
    return set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v07-tests", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--live-runtime", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--mt5", choices=("PASS", "FAIL"), required=True)
    args = parser.parse_args()

    state, _, state_detail = load_state()
    contract_ok, _ = validate_state(state)
    exposed = _keys(state).intersection(FORBIDDEN_RESPONSE_KEYS)
    account_fields = exposed.intersection({"account", "account_number", "login", "balance", "equity", "positions", "orders", "order"})
    credential_fields = exposed.intersection({"password", "token", "api_key", "apikey", "credentials", "secret"})
    providers = intelligence().get("provider_health", {})
    degraded = [name for name, status in providers.items() if status == "DEGRADED"]
    failures = (
        args.v07_tests != "PASS" or args.live_runtime != "PASS" or args.mt5 != "PASS" or
        not contract_ok or state_detail != "PASS" or trading_route_count() != 0 or bool(exposed)
    )
    final = "FAIL" if failures else "PASS_LOCAL_DASHBOARD_DEGRADED_PROVIDERS" if degraded else "PASS_LOCAL_DASHBOARD"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    decision = state.get("decision") or {}
    system = state.get("system") or {}
    lines = [
        "MARKETFUSION V0.7 DASHBOARD VALIDATION", "", f"commit: {commit}",
        f"tested_at_utc: {pd.Timestamp.now(tz='UTC').isoformat()}", "",
        "Backend API:", args.v07_tests, "", "Frontend build:", args.v07_tests, "",
        "Frontend tests:", args.v07_tests, "", "Live runtime:", args.live_runtime, "",
        "Current advisory:", str(decision.get("action", "WAIT")), "", "Confidence:",
        str(decision.get("confidence", "VERY_LOW")), "", "MT5:", "CONNECTED" if args.mt5 == "PASS" else "UNAVAILABLE", "",
        "V0.6:", str(system.get("status", "UNAVAILABLE")), "", "Provider health:",
        *(f"{name}: {status}" for name, status in providers.items()), "",
        "Security:", f"trading endpoints = {trading_route_count()}",
        f"account fields exposed = {len(account_fields)}", f"credential fields exposed = {len(credential_fields)}", "",
        "State contract:", "PASS" if contract_ok and state_detail == "PASS" else "FAIL", "",
        "FINAL STATUS:", final,
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(REPORT.read_text(encoding="utf-8"))
    return 1 if final == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
