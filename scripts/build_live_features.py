from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger

from src.settings import get_settings


def main():
    settings = get_settings()

    market_dir = settings.data_path / "features" / "market"
    news_path = settings.data_path / "features" / "news" / "daily_news_features.parquet"

    out_dir = settings.data_path / "live"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    for path in sorted(market_dir.glob("*_day.parquet")):
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["time"], utc=True).dt.floor("D")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No market feature files found in {market_dir}")

    market = pd.concat(frames, ignore_index=True)
    market["date"] = pd.to_datetime(market["date"], utc=True).dt.floor("D")

    if news_path.exists():
        news = pd.read_parquet(news_path)
        news["date"] = pd.to_datetime(news["date"], utc=True).dt.floor("D")
    else:
        news = pd.DataFrame(columns=["ticker", "date"])

    live = market.merge(news, on=["ticker", "date"], how="left")

    news_cols = [
        "news_count_1d",
        "unique_domains_1d",
        "english_news_count_1d",
        "russian_news_count_1d",
        "news_count_3d",
        "unique_domains_3d",
        "news_count_7d",
        "unique_domains_7d",
    ]

    for col in news_cols:
        if col not in live.columns:
            live[col] = 0
        live[col] = live[col].fillna(0)

    # Убираем только строки, где не хватает исторических рыночных признаков.
    required_cols = [
        "return_1",
        "return_3",
        "return_6",
        "return_12",
        "volatility_12",
        "volatility_24",
        "ma_12",
        "ma_24",
        "ma_72",
    ]

    live = live.dropna(subset=required_cols)
    live = live.sort_values(["date", "ticker"]).reset_index(drop=True)

    out_path = out_dir / "live_features_day.parquet"
    live.to_parquet(out_path, index=False)

    print(f"Saved {len(live)} rows to {out_path}")
    print(f"Date range: {live['date'].min()} → {live['date'].max()}")

    print("\nLatest rows by ticker:")
    latest = live.sort_values("date").groupby("ticker").tail(1)
    print(latest[["date", "ticker", "close", "return_1", "close_to_ma_72"]].to_string(index=False))

    logger.info("Done")


if __name__ == "__main__":
    main()
