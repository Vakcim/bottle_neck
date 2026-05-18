from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.settings import get_settings


def main():
    settings = get_settings()
    candles_dir = settings.data_path / "candles"

    rows = []

    for path in sorted(candles_dir.glob("*_day.parquet")):
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["time"], utc=True).dt.date
        df["weekday"] = pd.to_datetime(df["date"]).dt.day_name()

        weekend = df[df["weekday"].isin(["Saturday", "Sunday"])]

        rows.append(
            {
                "file": path.name,
                "ticker": df["ticker"].iloc[0],
                "rows": len(df),
                "weekend_rows": len(weekend),
                "first_weekend_dates": ", ".join(map(str, weekend["date"].head(5).tolist())),
            }
        )

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
