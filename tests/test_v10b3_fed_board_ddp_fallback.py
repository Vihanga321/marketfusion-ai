from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.intelligence.gold_macro_context import CONTEXT_SERIES
from src.intelligence.gold_macro_fed_board_fallback import (
    H15_NOMINAL_PACKAGE_URL,
    H15_REAL_PACKAGE_URL,
    _parse_h15_ddp_csv,
    bootstrap_fed_board_context,
)


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FedBoardSession:
    def __init__(self) -> None:
        self.urls: list[str] = []

    @staticmethod
    def _dates(count: int = 130) -> list[pd.Timestamp]:
        return list(pd.date_range("2025-01-02", periods=count, freq="B", tz="UTC"))

    def _h10(self) -> str:
        rows = "".join(
            f"<tr><td>{stamp.date().isoformat()}</td><td>{100.0 + idx / 10.0:.2f}</td></tr>"
            for idx, stamp in enumerate(self._dates())
        )
        return f"<html><body><table>{rows}</table></body></html>"

    def _nominal_csv(self) -> str:
        lines = [
            "H.15 Selected Interest Rates",
            "Series Description,2-year Treasury,10-year Treasury",
            "Unit,Percent,Percent",
            "Multiplier,1,1",
            "Currency,NA,NA",
            "Time Period,RIFLGFCY02_N.B,RIFLGFCY10_N.B",
        ]
        lines.extend(
            f"{stamp.date().isoformat()},{4.00 + idx / 1000.0:.3f},{4.50 + idx / 1000.0:.3f}"
            for idx, stamp in enumerate(self._dates())
        )
        return "\n".join(lines) + "\n"

    def _real_csv(self) -> str:
        lines = [
            "H.15 Selected Interest Rates",
            "Series Description,10-year inflation indexed Treasury",
            "Unit,Percent",
            "Multiplier,1",
            "Currency,NA",
            "Time Period,RIFLGFCY10_XII_N.B",
        ]
        lines.extend(
            f"{stamp.date().isoformat()},{1.75 + idx / 1000.0:.3f}"
            for idx, stamp in enumerate(self._dates())
        )
        return "\n".join(lines) + "\n"

    def get(self, url: str, **_: object) -> _Response:
        self.urls.append(url)
        if url == H15_NOMINAL_PACKAGE_URL:
            return _Response(self._nominal_csv())
        if url == H15_REAL_PACKAGE_URL:
            return _Response(self._real_csv())
        if "releases/h10/summary" in url:
            return _Response(self._h10())
        raise AssertionError(f"unexpected URL {url}")


class V10B3FedBoardDdpFallbackTests(unittest.TestCase):
    def test_h15_ddp_parser_reads_full_history_after_metadata_rows(self) -> None:
        session = _FedBoardSession()
        text = session._nominal_csv()
        spec = next(item for item in CONTEXT_SERIES if item.key == "yield_2y")
        frame = _parse_h15_ddp_csv(text, spec, "RIFLGFCY02_N.B")
        self.assertEqual(len(frame), 130)
        self.assertEqual(frame.iloc[0]["value"], 4.0)
        self.assertGreater(frame.iloc[-1]["value"], frame.iloc[0]["value"])
        self.assertTrue((frame["available_at_utc"] >= frame["observation_date_utc"]).all())

    def test_bootstrap_fetches_shared_nominal_package_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = _FedBoardSession()
            frame, manifest = bootstrap_fed_board_context(root, session=session)
            self.assertEqual(set(frame["series_id"]), {item.series_id for item in CONTEXT_SERIES})
            self.assertEqual(manifest["unique_provider_requests"], 3)
            self.assertEqual(session.urls.count(H15_NOMINAL_PACKAGE_URL), 1)
            self.assertEqual(session.urls.count(H15_REAL_PACKAGE_URL), 1)
            self.assertEqual(len(manifest["series"]), 4)
            self.assertFalse(manifest["production_model_eligible"])


if __name__ == "__main__":
    unittest.main()
