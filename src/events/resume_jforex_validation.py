"""Integrity-checked planning and merging for resumable JForex validation.

Only results produced by the same event input and validation contract may be
reused. PASS rows are preserved; only INCOMPLETE rows are retried. MISMATCH and
ERROR rows remain visible and are never converted to PASS by orchestration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_VERSION = "v0.4a-native-bid-tick-rebuild-v3"
SUMMARY_FILENAME = "robust_tick_validation_summary.tsv"
CHUNK_FILENAME = "robust_tick_chunk_status.tsv"
METADATA_FILENAME = "result_metadata.json"
VALID_STATUSES = {"PASS", "INCOMPLETE", "MISMATCH", "ERROR"}
VALIDATOR_FILES = (
    ROOT / "jforex-event-exporter" / "pom.xml",
    ROOT / "jforex-event-exporter" / "src" / "main" / "java" / "ai" / "marketfusion" / "jforex" / "TickValidationBatchRobustV2.java",
    ROOT / "jforex-event-exporter" / "src" / "main" / "java" / "ai" / "marketfusion" / "jforex" / "TickBarReconstructor.java",
)
CONTRACT = {
    "instrument": "EURUSD",
    "period": "M1",
    "window_before_minutes": 10,
    "window_after_minutes": 250,
    "expected_minutes": 261,
    "price_tolerance": 1.0e-8,
    "native_reference": "Dukascopy BID M1",
    "tick_rebuild": "BID and ASK, UTC minute, first/max/min/last",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validator_sha256() -> str:
    digest = hashlib.sha256()
    for path in VALIDATOR_FILES:
        if not path.is_file():
            raise RuntimeError(f"Validation-contract source is missing: {path}")
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def fingerprint(input_path: Path) -> dict[str, object]:
    if not input_path.is_file():
        raise RuntimeError(f"Missing batch input: {input_path}")
    payload: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "contract": CONTRACT,
        "input_sha256": sha256_file(input_path),
        "validator_sha256": validator_sha256(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False, encoding="utf-8")
    temporary.replace(path)


def read_events(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"Missing events TSV: {path}")
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if frame.empty or "event_id" not in frame:
        raise RuntimeError(f"Events TSV is empty or lacks event_id: {path}")
    if frame["event_id"].eq("").any() or frame["event_id"].duplicated().any():
        raise RuntimeError("Events TSV must contain unique non-empty event_id values")
    return frame


def read_summary(path: Path, expected_ids: set[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"Missing validation summary: {path}")
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"event_id", "status"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError("Validation summary missing columns: " + ", ".join(missing))
    if frame["event_id"].eq("").any() or frame["event_id"].duplicated().any():
        raise RuntimeError("Validation summary contains empty or duplicate event IDs")
    unknown = sorted(set(frame["status"]).difference(VALID_STATUSES))
    if unknown:
        raise RuntimeError("Validation summary contains unknown statuses: " + ", ".join(unknown))
    if expected_ids is not None and set(frame["event_id"]) != expected_ids:
        raise RuntimeError("Validation summary event set does not match the planned attempt")
    return frame


def compatible_result(input_path: Path, result_dir: Path) -> bool:
    metadata_path = result_dir / METADATA_FILENAME
    summary_path = result_dir / SUMMARY_FILENAME
    chunk_path = result_dir / CHUNK_FILENAME
    if not metadata_path.is_file() or not summary_path.is_file() or not chunk_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        current = fingerprint(input_path)
        if metadata.get("fingerprint_sha256") != current["fingerprint_sha256"]:
            return False
        expected = set(read_events(input_path)["event_id"])
        read_summary(summary_path, expected)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return False
    return True


def plan(input_path: Path, result_dir: Path, attempt_input: Path, plan_file: Path) -> dict[str, object]:
    events = read_events(input_path)
    reuse = compatible_result(input_path, result_dir)
    hard_ids: list[str] = []
    if reuse:
        current = read_summary(result_dir / SUMMARY_FILENAME, set(events["event_id"]))
        retry_ids = current.loc[current["status"].eq("INCOMPLETE"), "event_id"].tolist()
        hard_ids = current.loc[current["status"].isin(["MISMATCH", "ERROR"]), "event_id"].tolist()
        mode = "RETRY_INCOMPLETE" if retry_ids else ("SKIP_HARD_FAILURE" if hard_ids else "SKIP_PASS")
    else:
        retry_ids = events["event_id"].tolist()
        mode = "RETRY_ALL"

    selected = events[events["event_id"].isin(retry_ids)].copy()
    if retry_ids:
        if len(selected) != len(retry_ids):
            raise RuntimeError("Retry plan lost one or more requested event IDs")
        atomic_tsv(selected, attempt_input)
    elif attempt_input.exists():
        attempt_input.unlink()

    output: dict[str, object] = {
        "mode": mode,
        "reuse_compatible_result": reuse,
        "retry_event_ids": retry_ids,
        "hard_failure_event_ids": hard_ids,
        "fingerprint": fingerprint(input_path),
    }
    atomic_text(plan_file, json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"RESUME_PLAN: {mode}")
    print(f"Retry events: {len(retry_ids)}")
    print(f"Preserved hard failures: {len(hard_ids)}")
    return output


def merge(
    input_path: Path,
    result_dir: Path,
    attempt_result: Path,
    plan_file: Path,
) -> pd.DataFrame:
    events = read_events(input_path)
    expected_ids = set(events["event_id"])
    plan_data = json.loads(plan_file.read_text(encoding="utf-8"))
    if plan_data.get("fingerprint", {}).get("fingerprint_sha256") != fingerprint(input_path)["fingerprint_sha256"]:
        raise RuntimeError("Input or validator changed after retry planning; refusing to merge")
    retry_ids = set(plan_data.get("retry_event_ids", []))
    if not retry_ids:
        raise RuntimeError("Retry plan contains no events to merge")
    attempt = read_summary(attempt_result / SUMMARY_FILENAME, retry_ids)

    if bool(plan_data.get("reuse_compatible_result")):
        existing = read_summary(result_dir / SUMMARY_FILENAME, expected_ids)
        preserved = existing[~existing["event_id"].isin(retry_ids)].copy()
        combined = pd.concat([preserved, attempt], ignore_index=True, sort=False)
    else:
        combined = attempt.copy()
    if set(combined["event_id"]) != expected_ids or combined["event_id"].duplicated().any():
        raise RuntimeError("Merged result is not exactly one row per expected event")
    order = {event_id: index for index, event_id in enumerate(events["event_id"])}
    combined["_order"] = combined["event_id"].map(order)
    combined = combined.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    attempt_chunks_path = attempt_result / CHUNK_FILENAME
    chunks = pd.read_csv(attempt_chunks_path, sep="\t", dtype=str, keep_default_na=False)
    if "event_id" not in chunks:
        raise RuntimeError("Attempt chunk status lacks event_id")
    if bool(plan_data.get("reuse_compatible_result")) and (result_dir / CHUNK_FILENAME).is_file():
        existing_chunks = pd.read_csv(result_dir / CHUNK_FILENAME, sep="\t", dtype=str, keep_default_na=False)
        existing_chunks = existing_chunks[~existing_chunks["event_id"].isin(retry_ids)]
        chunks = pd.concat([existing_chunks, chunks], ignore_index=True, sort=False)

    # Metadata is the transaction commit marker. Remove the previous marker
    # before replacing either data file so an interrupted merge can never make
    # a new summary look compatible with stale chunk diagnostics.
    metadata_path = result_dir / METADATA_FILENAME
    if metadata_path.exists():
        metadata_path.unlink()
    atomic_tsv(combined, result_dir / SUMMARY_FILENAME)
    atomic_tsv(chunks, result_dir / CHUNK_FILENAME)
    metadata = fingerprint(input_path)
    metadata["created_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["statuses"] = combined["status"].value_counts().to_dict()
    atomic_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("MERGE_STATUS: PASS")
    print("Statuses: " + str(metadata["statuses"]))
    return combined


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--input", type=Path, required=True)
    plan_parser.add_argument("--result-dir", type=Path, required=True)
    plan_parser.add_argument("--attempt-input", type=Path, required=True)
    plan_parser.add_argument("--plan-file", type=Path, required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--input", type=Path, required=True)
    merge_parser.add_argument("--result-dir", type=Path, required=True)
    merge_parser.add_argument("--attempt-result", type=Path, required=True)
    merge_parser.add_argument("--plan-file", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    try:
        if args.command == "plan":
            plan(args.input.resolve(), args.result_dir.resolve(), args.attempt_input.resolve(), args.plan_file.resolve())
        else:
            merge(args.input.resolve(), args.result_dir.resolve(), args.attempt_result.resolve(), args.plan_file.resolve())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
