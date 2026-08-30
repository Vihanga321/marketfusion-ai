"""Generate the separated V0.9A.1 descriptive pattern report."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engines.research import generate_pattern_research


if __name__ == "__main__":
    print(generate_pattern_research())
