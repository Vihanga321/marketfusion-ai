"""Frozen V0.4A production contract and evidence fingerprints."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_VERSION = "v0.4a-native-bid-tick-rebuild-v3"
REACTION_CONTRACT_VERSION = "v0.4a-event-reaction-v1"
REACTION_SOURCE_ID = "DUKASCOPY_JFOREX_HISTORICAL_TICKS"
EXPECTED_PRODUCTION_COUNTS = {"PASS": 242, "INCOMPLETE": 30, "MISMATCH": 4, "ERROR": 0}
EXPECTED_ELIGIBLE_EVENTS = 242
PIP_SIZE = 0.0001
FLAT_TOLERANCE_PIPS = 0.0
WINDOW_BEFORE_MINUTES = 10
WINDOW_AFTER_MINUTES = 250
HORIZON_BAR_OFFSETS = {1: 0, 5: 4, 15: 14, 60: 59, 240: 239}

FROZEN_EVIDENCE_SHA256 = {
    "data/dukascopy/full_validation_summary.tsv":
        "8DE19AA05F970F1211D3C2A9E0E2209BC2CA17D3E622AC49A6B16B2DF6A18FFF",
    "reports/dukascopy_full_validation_summary.txt":
        "5F86379FD79936927CD498E133CA2585F347B3271A27EEAA093C296E1395EED9",
    "reports/dukascopy_incomplete_events.tsv":
        "903DA3E741C89AF516C15ADD933CC2C6FE6DD47423404DEA122E634F46F42BB6",
    "reports/dukascopy_mismatch_minutes.tsv":
        "B25D6608A64CC9D32E62EA37C80379827BDD091372CF738C2701DC42DA020D81",
    "reports/dukascopy_mismatch_tick_context.tsv":
        "8F60BEFDF287B45475BC833D236171ADC7329129264F8029919D82A2761C9F26",
    "reports/dukascopy_mismatch_diagnostic.txt":
        "9F0E27EB9DB8F14C0EEAD9C6F01181FD458F2C56239B47F7330CC6CE45A25D2D",
}

MISMATCH_EVENT_IDS = frozenset(
    {
        "bls:us_employment_situation:20220902T1230Z",
        "bls:us_cpi_release:20230913T1230Z",
        "bls:us_employment_situation:20250404T1230Z",
        "bls:us_cpi_release:20260512T1230Z",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_frozen_evidence(root: Path = ROOT) -> dict[str, str]:
    observed: dict[str, str] = {}
    errors: list[str] = []
    for relative, expected in FROZEN_EVIDENCE_SHA256.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing frozen evidence: {relative}")
            continue
        actual = sha256_file(path)
        observed[relative] = actual
        if actual != expected:
            errors.append(f"frozen evidence changed: {relative}; expected={expected}; actual={actual}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return observed
