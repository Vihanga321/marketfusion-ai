"""Lightweight feature/output drift diagnostics with explicit sample gates."""
from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pandas as pd


def reference_hash(frame: pd.DataFrame, features: list[str]) -> str:
    payload = {name: {"mean": float(pd.to_numeric(frame[name], errors="coerce").mean()), "std": float(pd.to_numeric(frame[name], errors="coerce").std()), "missing": float(frame[name].isna().mean())} for name in features}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _psi(reference: pd.Series, recent: pd.Series, bins: int = 10) -> float:
    clean_reference = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(float)
    clean_recent = pd.to_numeric(recent, errors="coerce").dropna().to_numpy(float)
    if not len(clean_reference) or not len(clean_recent):
        return float("nan")
    edges = np.unique(np.quantile(clean_reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.isclose(np.mean(clean_reference), np.mean(clean_recent)) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(clean_reference, bins=edges)[0] / len(clean_reference)
    actual = np.histogram(clean_recent, bins=edges)[0] / len(clean_recent)
    expected, actual = np.clip(expected, 1e-6, None), np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def distribution_drift(
    reference: pd.DataFrame, recent: pd.DataFrame, features: list[str],
    expected_reference_hash: str | None = None, recent_as_of_utc: object | None = None,
) -> pd.DataFrame:
    columns = ["monitor", "feature", "reference_rows", "recent_rows", "psi", "standardized_mean_shift", "missing_rate_shift", "status"]
    missing = [name for name in features if name not in reference or name not in recent]
    if missing:
        return pd.DataFrame([{"monitor": "FEATURE", "feature": name, "reference_rows": len(reference), "recent_rows": len(recent), "status": "INSUFFICIENT_DATA"} for name in missing], columns=columns)
    if expected_reference_hash is not None and reference_hash(reference, features) != expected_reference_hash:
        raise ValueError("Training reference mismatch")
    if recent_as_of_utc is not None and "decision_timestamp_utc" in recent:
        timestamps = pd.to_datetime(recent["decision_timestamp_utc"], utc=True, errors="raise")
        cutoff = pd.Timestamp(recent_as_of_utc)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        if timestamps.gt(cutoff).any():
            raise ValueError("Future feature detected in drift input")
    if len(reference) < 30 or len(recent) < 20:
        return pd.DataFrame([{"monitor": "FEATURE", "feature": name, "reference_rows": len(reference), "recent_rows": len(recent), "status": "INSUFFICIENT_DATA"} for name in features], columns=columns)
    rows = []
    for name in features:
        ref = pd.to_numeric(reference[name], errors="coerce")
        cur = pd.to_numeric(recent[name], errors="coerce")
        std = float(ref.std())
        shift = 0.0 if (not np.isfinite(std) or std == 0) and np.isclose(ref.mean(), cur.mean(), equal_nan=True) else float("inf") if std == 0 else abs(float(cur.mean() - ref.mean())) / std
        psi = _psi(ref, cur)
        missing_shift = abs(float(cur.isna().mean() - ref.isna().mean()))
        score = max(psi if np.isfinite(psi) else 0, shift if np.isfinite(shift) else 10, missing_shift * 4)
        status = "DEGRADED" if score >= 0.5 else "WATCH" if score >= 0.2 else "NORMAL"
        rows.append({"monitor": "FEATURE", "feature": name, "reference_rows": len(reference), "recent_rows": len(recent), "psi": psi, "standardized_mean_shift": shift, "missing_rate_shift": missing_shift, "status": status})
    return pd.DataFrame(rows, columns=columns)
