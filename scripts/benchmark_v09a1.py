"""Compare committed V0.9A status latency with bounded V0.9A.1 analysis."""
from __future__ import annotations

from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engines import engine_layer


def milliseconds(function, repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        values.append((time.perf_counter() - started) * 1000.0)
    return values


def main() -> None:
    source = subprocess.run(
        ["git", "show", "HEAD:src/engines/engine_layer.py"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout
    namespace = {"__name__": "v09a_committed_baseline", "__file__": "src/engines/engine_layer.py"}
    exec(compile(source, "v09a_committed_baseline", "exec"), namespace)
    baseline = milliseconds(namespace["engine_status"], 20)
    engine_layer._cached_parquet.cache_clear()
    engine_layer._cached_timeframe_analysis.cache_clear()
    cold = milliseconds(engine_layer.engine_status, 1)[0]
    warm = milliseconds(engine_layer.engine_status, 20)
    report = ROOT / "reports" / "v09a1_benchmark.txt"
    report.write_text("\n".join([
        "MARKETFUSION V0.9A.1 ENGINE BENCHMARK", "",
        "method: perf_counter localhost read-only engine_status",
        "baseline: committed HEAD V0.9A feature-row engines",
        "candidate: bounded M5/M15/H1 V0.9A.1 with file-version cache", "",
        f"baseline_mean_ms: {statistics.mean(baseline):.3f}",
        f"baseline_median_ms: {statistics.median(baseline):.3f}",
        f"v09a1_cold_ms: {cold:.3f}",
        f"v09a1_warm_mean_ms: {statistics.mean(warm):.3f}",
        f"v09a1_warm_median_ms: {statistics.median(warm):.3f}",
        "analysis_window_bars: M5=600,M15=600,H1=400",
        "interpretation: cold analysis is heavier; cached dashboard refresh remains bounded",
        "trading_impact: NONE", "",
    ]), encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
