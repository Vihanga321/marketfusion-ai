"""Pure point-in-time helpers shared by macro feature builders and tests."""

from __future__ import annotations

import pandas as pd


def released_today_asof(release_timestamps: list[pd.Timestamp], decisions: pd.Series) -> pd.Series:
    """Flag a release day only after that day's first release is public.

    Marking an entire UTC day from a future release timestamp would leak the
    release into pre-release market rows. This backward as-of join keeps the
    regime feature governed by the same availability rule as every macro value.
    """
    left = pd.DataFrame({"decision": pd.to_datetime(decisions, utc=True)}).sort_values("decision")
    unique = pd.Series(pd.to_datetime(release_timestamps, utc=True)).dropna().drop_duplicates().sort_values()
    if unique.empty:
        return pd.Series(False, index=left.index, dtype=bool).sort_index()
    right = pd.DataFrame({"latest_release": unique.to_numpy()})
    joined = pd.merge_asof(
        left,
        right,
        left_on="decision",
        right_on="latest_release",
        direction="backward",
        allow_exact_matches=True,
    )
    result = (
        joined["latest_release"].notna()
        & joined["latest_release"].dt.normalize().eq(joined["decision"].dt.normalize())
    )
    return result.sort_index()
