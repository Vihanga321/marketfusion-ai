"""Run V0.9A.2 from the repository root."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from src.research.v09a2_runner import run_v09a2

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--reuse-replay",action="store_true",help="Reuse validated ignored replay parquet files")
    parser.add_argument("--reuse-ablation",action="store_true",help="Reuse a validated complete 30-row ablation report with final holdout")
    args=parser.parse_args()
    print(json.dumps(run_v09a2(reuse_replay=args.reuse_replay,reuse_ablation=args.reuse_ablation),indent=2,default=str))
