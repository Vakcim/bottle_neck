from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.storage import LocalStorage
from src.settings import get_settings


def main():
    settings = get_settings()
    storage = LocalStorage(settings.data_path)

    rows = []

    for path in storage.list_candle_files():
        df = pd.read_parquet(path)

        ticker = df["ticker"].iloc[0]
        interval = df["interval"].iloc[0]

        duplicate_count = df.duplicated(subset=["time", "ticker", "interval"]).sum()
        zero_volume_count = (df["volume"] <= 0).sum()

        bad_price_count = (
            (df["open"] <= 0)
            | (df["high"] <= 0)
            | (df["low"] <= 0)
            | (df["close"] <= 0)
            | (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        ).sum()

        rows.append(
            {
                "ticker": ticker,
                "interval": interval,
                "rows": len(df),
                "from": df["time"].min(),
                "to": df["time"].max(),
                "duplicates": int(duplicate_count),
                "zero_volume": int(zero_volume_count),
                "bad_prices": int(bad_price_count),
                "min_close": round(float(df["close"].min()), 4),
                "max_close": round(float(df["close"].max()), 4),
            }
        )

    report = pd.DataFrame(rows).sort_values(["ticker", "interval"])

    print(report.to_string(index=False))

    out_dir = storage.root_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "data_quality_report.csv"
    report.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
