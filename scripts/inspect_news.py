from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.settings import get_settings


def main():
    settings = get_settings()
    news_dir = settings.data_path / "news" / "gdelt"

    files = sorted(news_dir.glob("*_gdelt.parquet"))

    if not files:
        print("Новостей пока нет.")
        return

    rows = []

    for path in files:
        df = pd.read_parquet(path)

        rows.append(
            {
                "file": path.name,
                "ticker": df["ticker"].iloc[0] if len(df) else None,
                "rows": len(df),
                "from": df["published_at"].min() if len(df) else None,
                "to": df["published_at"].max() if len(df) else None,
                "domains": df["domain"].nunique() if "domain" in df else None,
                "languages": ", ".join(sorted(df["language"].dropna().unique())) if "language" in df else None,
            }
        )

    report = pd.DataFrame(rows).sort_values("ticker")
    print(report.to_string(index=False))

    out_dir = settings.data_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "news_report.csv"
    report.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
