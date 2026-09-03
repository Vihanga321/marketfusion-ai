"""MarketFusion V1.0C.1 reliability supervisor for frozen XAUUSD forward validation.

This module intentionally does not alter the frozen V1.0C prediction, target,
feature, model, or evaluation logic.  It supervises the existing one-cycle
collector with reconnect/retry, a single-instance lock, health telemetry and
provider-outage logging.

Safety invariants:
- automatic execution remains disabled;
- the collector remains SHADOW_ADVISORY_ONLY;
- missed forward decisions are never backfilled by this supervisor;
- one failed MT5/clock cycle records nothing and retries later;
- initialization remains an explicit separate one-time operator action.
"""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
from typing import Any

from src.assets.contracts import ROOT, asset_paths, discover_broker_symbol
from src.evaluation import v10c_xauusd_forward_validation as frozen
from src.research.v10b4_xauusd_intraday_context_audit import run_audit
from src.research.v10b5_xauusd_intraday_intermarket_evidence import CONTEXT_GROUPS, _ready_symbols

RUNTIME_CONTRACT_VERSION = "v1.0c1-xauusd-reliability-supervisor-v1"
RUNTIME_DIR = ROOT / "data" / "runtime" / "v10c1"
HEALTH_FILE = RUNTIME_DIR / "collector_health.json"
OUTAGE_FILE = RUNTIME_DIR / "provider_outages.jsonl"
LOCK_FILE = RUNTIME_DIR / "collector.lock"
MIN_DISK_FREE_GB = 5.0
DEFAULT_INTERVAL_SECONDS = 60
MIN_INTERVAL_SECONDS = 30
DEFAULT_MAX_RETRY_SECONDS = 300


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8", newline="\n")
    temp.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


def _total_memory_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            state = MEMORYSTATUSEX()
            state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
                return int(state.ullTotalPhys)
        except Exception:
            return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, ValueError, OSError):
        return None


def machine_snapshot(root: Path = ROOT) -> dict[str, Any]:
    disk = shutil.disk_usage(root)
    memory = _total_memory_bytes()
    return {
        "hostname": socket.gethostname(),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "cpu_logical_count": os.cpu_count(),
        "memory_total_gb": round(memory / (1024 ** 3), 2) if memory else None,
        "disk_total_gb": round(disk.total / (1024 ** 3), 2),
        "disk_free_gb": round(disk.free / (1024 ** 3), 2),
        "disk_free_status": "PASS" if disk.free >= MIN_DISK_FREE_GB * (1024 ** 3) else "FAIL_LOW_DISK",
    }


class SingleInstanceLock(AbstractContextManager["SingleInstanceLock"]):
    """Cross-platform advisory lock held for the life of the supervisor."""

    def __init__(self, path: Path = LOCK_FILE):
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> "SingleInstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(" ")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError("Another V1.0C.1 reliability collector is already running") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "started_at_utc": _utc_now()}))
        self.handle.flush()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def _frozen_paths(root: Path = ROOT) -> dict[str, Path]:
    paths = asset_paths("XAUUSD", root)
    base = paths.evaluation / frozen.FORWARD_DIR_NAME
    return {
        "manifest": base / "manifest.json",
        "status": base / "latest_status.json",
        "predictions": base / "predictions.parquet",
        "outcomes": base / "outcomes.parquet",
    }


def _assert_initialized_safely(root: Path = ROOT) -> dict[str, Any]:
    path = _frozen_paths(root)["manifest"]
    manifest = _read_json(path)
    if manifest is None:
        raise RuntimeError("Frozen forward validation is not initialized. Initialize exactly once before starting the reliable collector.")
    if manifest.get("automatic_execution") != "DISABLED":
        raise RuntimeError("Safety invariant failed: automatic execution is not DISABLED")
    if manifest.get("runtime") != "SHADOW_ADVISORY_ONLY":
        raise RuntimeError("Safety invariant failed: runtime is not SHADOW_ADVISORY_ONLY")
    if manifest.get("manual_confirmation") != "REQUIRED":
        raise RuntimeError("Safety invariant failed: manual confirmation is not REQUIRED")
    if manifest.get("retraining_during_forward_window") is not False:
        raise RuntimeError("Safety invariant failed: retraining is not frozen")
    return manifest


def _verify_model_hashes(manifest: dict[str, Any], root: Path = ROOT) -> dict[str, str]:
    verified: dict[str, str] = {}
    models = manifest.get("models")
    if not isinstance(models, dict):
        raise RuntimeError("Frozen manifest model section is invalid")
    for contract_id, item in models.items():
        if not isinstance(item, dict):
            raise RuntimeError(f"Frozen model manifest is invalid for {contract_id}")
        for kind in ("baseline", "context"):
            relative = item.get(f"{kind}_model_path")
            expected = item.get(f"{kind}_model_sha256")
            if not relative or not expected:
                raise RuntimeError(f"Missing frozen {kind} model metadata for {contract_id}")
            path = root / str(relative)
            if not path.exists() or frozen._file_sha(path) != str(expected):
                raise RuntimeError(f"Frozen {kind} model SHA mismatch for {contract_id}")
        verified[str(contract_id)] = "PASS"
    return verified


def preflight(*, root: Path = ROOT, require_uninitialized: bool = False) -> dict[str, Any]:
    """Read-only infrastructure preflight. It never records a prediction or outcome."""
    checks: dict[str, Any] = {}
    machine = machine_snapshot(root)
    checks["disk"] = machine["disk_free_status"]
    if checks["disk"] != "PASS":
        return {"contract_version": RUNTIME_CONTRACT_VERSION, "status": "FAIL", "checks": checks, "machine": machine, "automatic_execution": "DISABLED"}

    forward_paths = _frozen_paths(root)
    initialized = forward_paths["manifest"].exists()
    checks["forward_state"] = "INITIALIZED" if initialized else "NOT_INITIALIZED"
    if require_uninitialized and initialized:
        return {
            "contract_version": RUNTIME_CONTRACT_VERSION,
            "status": "FAIL_EXPECTED_CLEAN_UNINITIALIZED_STATE",
            "checks": checks,
            "machine": machine,
            "automatic_execution": "DISABLED",
        }

    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:
        checks["mt5_python"] = f"FAIL: {exc}"
        return {"contract_version": RUNTIME_CONTRACT_VERSION, "status": "FAIL", "checks": checks, "machine": machine, "automatic_execution": "DISABLED"}

    checks["mt5_python"] = "PASS"
    if not mt5.initialize():
        checks["mt5_connection"] = f"FAIL: {mt5.last_error()}"
        return {"contract_version": RUNTIME_CONTRACT_VERSION, "status": "FAIL", "checks": checks, "machine": machine, "automatic_execution": "DISABLED"}

    try:
        checks["mt5_connection"] = "PASS"
        audit = run_audit(mt5, root=root, persist=False)
        offset = frozen._normalized_offset(audit)
        checks["broker_clock"] = "PASS"
        checks["broker_clock_offset_seconds"] = offset
        ready = _ready_symbols(audit)
        required = {family for contract in frozen.FROZEN_CONTRACTS for family in CONTEXT_GROUPS[contract.context_group]}
        missing = sorted(required - set(ready))
        checks["context_symbols"] = "PASS" if not missing else "FAIL: " + ", ".join(missing)
        xau = discover_broker_symbol(mt5, "XAUUSD")
        checks["xauusd"] = f"PASS:{xau.broker_symbol}"
        if initialized:
            manifest = _assert_initialized_safely(root)
            checks["model_hashes"] = _verify_model_hashes(manifest, root)
            frozen_offset = int(manifest["broker_clock_offset_seconds"])
            checks["frozen_broker_offset"] = "PASS" if frozen_offset == offset else f"FAIL_EXPECTED_{frozen_offset}_GOT_{offset}"
        failed = [key for key, value in checks.items() if isinstance(value, str) and value.startswith("FAIL")]
        status = "PASS" if not failed else "FAIL"
        return {
            "contract_version": RUNTIME_CONTRACT_VERSION,
            "status": status,
            "checked_at_utc": _utc_now(),
            "checks": checks,
            "machine": machine,
            "automatic_execution": "DISABLED",
            "runtime": "SHADOW_ADVISORY_ONLY",
            "manual_confirmation": "REQUIRED",
            "records_forward_data": False,
        }
    except Exception as exc:
        checks["exception"] = f"FAIL: {type(exc).__name__}: {exc}"
        return {
            "contract_version": RUNTIME_CONTRACT_VERSION,
            "status": "FAIL",
            "checked_at_utc": _utc_now(),
            "checks": checks,
            "machine": machine,
            "automatic_execution": "DISABLED",
            "records_forward_data": False,
        }
    finally:
        mt5.shutdown()


def _health_payload(
    *,
    state: str,
    started_at_utc: str,
    last_success_utc: str | None,
    last_error: str | None,
    consecutive_failures: int,
    total_cycles: int,
    successful_cycles: int,
    reconnects: int,
    last_cycle: dict[str, Any] | None,
    current_outage_started_utc: str | None,
    root: Path = ROOT,
) -> dict[str, Any]:
    status = _read_json(_frozen_paths(root)["status"])
    return {
        "contract_version": RUNTIME_CONTRACT_VERSION,
        "updated_at_utc": _utc_now(),
        "collector": {
            "state": state,
            "pid": os.getpid(),
            "started_at_utc": started_at_utc,
            "last_success_utc": last_success_utc,
            "last_error": last_error,
            "consecutive_failures": consecutive_failures,
            "total_cycles": total_cycles,
            "successful_cycles": successful_cycles,
            "reconnects": reconnects,
            "current_outage_started_utc": current_outage_started_utc,
            "interval_seconds": None,
        },
        "last_cycle": last_cycle,
        "forward": {
            "decision": status.get("decision") if status else "STATUS_UNAVAILABLE",
            "forward_start_utc": status.get("forward_start_utc") if status else None,
            "contracts": status.get("contracts") if status else {},
        },
        "machine": machine_snapshot(root),
        "safety": {
            "automatic_execution": "DISABLED",
            "runtime": "SHADOW_ADVISORY_ONLY",
            "manual_confirmation": "REQUIRED",
            "backfill": "PROHIBITED",
            "retraining_during_forward_window": False,
        },
    }


def reliable_loop(
    *,
    root: Path = ROOT,
    continuous: bool = True,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    max_retry_seconds: int = DEFAULT_MAX_RETRY_SECONDS,
) -> int:
    interval = max(MIN_INTERVAL_SECONDS, int(interval_seconds))
    max_retry = max(interval, int(max_retry_seconds))
    _assert_initialized_safely(root)
    started_at = _utc_now()
    last_success: str | None = None
    last_error: str | None = None
    last_cycle: dict[str, Any] | None = None
    consecutive_failures = 0
    total_cycles = 0
    successful_cycles = 0
    reconnects = 0
    outage_started: str | None = None
    mt5: Any = None

    with SingleInstanceLock(LOCK_FILE if root == ROOT else root / "data" / "runtime" / "v10c1" / "collector.lock"):
        while True:
            total_cycles += 1
            cycle_started = time.monotonic()
            try:
                if mt5 is None:
                    try:
                        import MetaTrader5 as mt5_module  # type: ignore
                    except ImportError as exc:
                        raise RuntimeError(f"MetaTrader5 unavailable: {exc}") from exc
                    if not mt5_module.initialize():
                        raise RuntimeError(f"mt5.initialize() failed: {mt5_module.last_error()}")
                    mt5 = mt5_module
                    reconnects += 1
                cycle = frozen.run_cycle(mt5, root=root)
                last_cycle = cycle
                last_success = _utc_now()
                successful_cycles += 1
                consecutive_failures = 0
                last_error = None
                if outage_started is not None:
                    ended = _utc_now()
                    try:
                        duration = (datetime.fromisoformat(ended) - datetime.fromisoformat(outage_started)).total_seconds()
                    except ValueError:
                        duration = None
                    _append_jsonl(
                        OUTAGE_FILE if root == ROOT else root / "data" / "runtime" / "v10c1" / "provider_outages.jsonl",
                        {
                            "contract_version": RUNTIME_CONTRACT_VERSION,
                            "outage_started_utc": outage_started,
                            "outage_ended_utc": ended,
                            "duration_seconds": duration,
                            "recovered": True,
                            "automatic_execution": "DISABLED",
                        },
                    )
                    outage_started = None
                health = _health_payload(
                    state="RUNNING",
                    started_at_utc=started_at,
                    last_success_utc=last_success,
                    last_error=None,
                    consecutive_failures=0,
                    total_cycles=total_cycles,
                    successful_cycles=successful_cycles,
                    reconnects=reconnects,
                    last_cycle=last_cycle,
                    current_outage_started_utc=None,
                    root=root,
                )
                health["collector"]["interval_seconds"] = interval
                _atomic_json(HEALTH_FILE if root == ROOT else root / "data" / "runtime" / "v10c1" / "collector_health.json", health)
                print(json.dumps(health, indent=2, default=str))
                if not continuous:
                    return 0
                sleep_for = max(1.0, interval - (time.monotonic() - cycle_started))
                time.sleep(sleep_for)
            except KeyboardInterrupt:
                health = _health_payload(
                    state="STOPPED_BY_OPERATOR",
                    started_at_utc=started_at,
                    last_success_utc=last_success,
                    last_error="KeyboardInterrupt",
                    consecutive_failures=consecutive_failures,
                    total_cycles=total_cycles,
                    successful_cycles=successful_cycles,
                    reconnects=reconnects,
                    last_cycle=last_cycle,
                    current_outage_started_utc=outage_started,
                    root=root,
                )
                health["collector"]["interval_seconds"] = interval
                _atomic_json(HEALTH_FILE if root == ROOT else root / "data" / "runtime" / "v10c1" / "collector_health.json", health)
                return 130
            except Exception as exc:
                consecutive_failures += 1
                last_error = f"{type(exc).__name__}: {exc}"
                if outage_started is None:
                    outage_started = _utc_now()
                    _append_jsonl(
                        OUTAGE_FILE if root == ROOT else root / "data" / "runtime" / "v10c1" / "provider_outages.jsonl",
                        {
                            "contract_version": RUNTIME_CONTRACT_VERSION,
                            "outage_started_utc": outage_started,
                            "error": last_error,
                            "recovered": False,
                            "automatic_execution": "DISABLED",
                        },
                    )
                if mt5 is not None:
                    try:
                        mt5.shutdown()
                    except Exception:
                        pass
                    mt5 = None
                health = _health_payload(
                    state="FAIL_CLOSED_RETRYING",
                    started_at_utc=started_at,
                    last_success_utc=last_success,
                    last_error=last_error,
                    consecutive_failures=consecutive_failures,
                    total_cycles=total_cycles,
                    successful_cycles=successful_cycles,
                    reconnects=reconnects,
                    last_cycle=last_cycle,
                    current_outage_started_utc=outage_started,
                    root=root,
                )
                health["collector"]["interval_seconds"] = interval
                _atomic_json(HEALTH_FILE if root == ROOT else root / "data" / "runtime" / "v10c1" / "collector_health.json", health)
                print(json.dumps(health, indent=2, default=str), file=sys.stderr)
                if not continuous:
                    return 1
                retry = min(max_retry, max(interval, interval * (2 ** min(consecutive_failures - 1, 3))))
                time.sleep(retry)
        # unreachable


def status(root: Path = ROOT) -> dict[str, Any]:
    path = HEALTH_FILE if root == ROOT else root / "data" / "runtime" / "v10c1" / "collector_health.json"
    payload = _read_json(path)
    if payload is not None:
        return payload
    return {
        "contract_version": RUNTIME_CONTRACT_VERSION,
        "updated_at_utc": _utc_now(),
        "collector": {"state": "NOT_STARTED"},
        "machine": machine_snapshot(root),
        "safety": {
            "automatic_execution": "DISABLED",
            "runtime": "SHADOW_ADVISORY_ONLY",
            "manual_confirmation": "REQUIRED",
            "backfill": "PROHIBITED",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--max-retry-seconds", type=int, default=DEFAULT_MAX_RETRY_SECONDS)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--require-uninitialized", action="store_true")
    args = parser.parse_args()

    if args.status:
        print(json.dumps(status(), indent=2, default=str))
        return 0
    if args.preflight:
        result = preflight(require_uninitialized=args.require_uninitialized)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("status") == "PASS" else 1
    return reliable_loop(
        continuous=args.continuous,
        interval_seconds=args.interval_seconds,
        max_retry_seconds=args.max_retry_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
