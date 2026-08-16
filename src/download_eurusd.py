from pathlib import Path

import pandas as pd
import yfinance as yf


SYMBOL = "EURUSD=X"

OUTPUT_FOLDER = Path("data/raw")
OUTPUT_FILE = OUTPUT_FOLDER / "eurusd_daily.csv"


def download_data():
    print("=" * 60)
    print("MARKETFUSION AI")
    print("Downloading EUR/USD historical data...")
    print("=" * 60)

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    df = yf.download(
        SYMBOL,
        period="10y",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("No EUR/USD data was downloaded.")

    # yfinance can return MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    df.columns = [
        str(column).lower().replace(" ", "_")
        for column in df.columns
    ]

    wanted_columns = [
        column
        for column in [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]
        if column in df.columns
    ]

    df = df[wanted_columns]

    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print(f"Rows downloaded: {len(df):,}")
    print(f"First date: {df.iloc[0]['date']}")
    print(f"Last date:  {df.iloc[-1]['date']}")
    print()
    print("First 5 rows:")
    print(df.head())
    print()
    print(f"Saved to: {OUTPUT_FILE}")
    print()
    print("EUR/USD dataset ready ✅")


if __name__ == "__main__":
    download_data()