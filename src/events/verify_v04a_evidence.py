"""Verify that every frozen V0.4A production/forensic artifact is byte-identical."""

from __future__ import annotations

try:
    from v04a_contract import verify_frozen_evidence
except ImportError:  # pragma: no cover
    from src.events.v04a_contract import verify_frozen_evidence


def main() -> int:
    verified = verify_frozen_evidence()
    print("V04A_FROZEN_EVIDENCE_STATUS: PASS")
    print(f"Artifacts verified: {len(verified)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
