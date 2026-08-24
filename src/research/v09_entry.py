"""Stable local entrypoint for V0.9 research with exact V0.8 reproduction bridge."""
from __future__ import annotations

from src.research import v09_runner
from src.research.v09_reproduction import reproduce_v08


def main() -> None:
    # Keep the main hardening engine isolated while using the corrected frozen-prefix
    # reconstruction required by V0.8's purged development row accounting.
    v09_runner.reproduce_v08 = reproduce_v08
    v09_runner.run_research()


if __name__ == "__main__":
    main()
