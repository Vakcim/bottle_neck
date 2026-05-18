from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_all_gdelt_monthly(news_dir: str | Path) -> pd.DataFrame:
    news_dir = Path(news_dir)
    files = sorted(news_dir.glob("*_gdelt_monthly.parquet"))

    if not files:
        return pd.DataFrame()

    frames = []
    for path in files:
        df = pd.read_parquet(path)
        frames.append(df)

    news = pd.concat(frames, ignore_index=True)
    news["published_at"] = pd.to_datetime(news["published_at"], errors="coerce", utc=True)
    news = news.dropna(subset=["published_at", "ticker"])
    news["date"] = news["published_at"].dt.floor("D")

    return news


def build_daily_news_features(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "date",
                "news_count_1d",
                "unique_domains_1d",
                "english_news_count_1d",
                "russian_news_count_1d",
                "news_count_3d",
                "unique_domains_3d",
                "news_count_7d",
                "unique_domains_7d",
            ]
        )

    news = news.copy()
    news["date"] = pd.to_datetime(news["date"], utc=True).dt.floor("D")
    news["language"] = news["language"].fillna("").astype(str)
    news["domain"] = news["domain"].fillna("").astype(str)

    daily = (
        news.groupby(["ticker", "date"])
        .agg(
            news_count_1d=("url", "count"),
            unique_domains_1d=("domain", "nunique"),
            english_news_count_1d=("language", lambda x: (x == "English").sum()),
            russian_news_count_1d=("language", lambda x: (x == "Russian").sum()),
        )
        .reset_index()
    )

    all_frames = []

    for ticker, group in daily.groupby("ticker"):
        group = group.sort_values("date").set_index("date")

        # Полная дневная сетка между первой и последней новостью.
        idx = pd.date_range(group.index.min(), group.index.max(), freq="D", tz="UTC")
        group = group.reindex(idx)
        group.index.name = "date"
        group["ticker"] = ticker

        count_cols = [
            "news_count_1d",
            "unique_domains_1d",
            "english_news_count_1d",
            "russian_news_count_1d",
        ]
        group[count_cols] = group[count_cols].fillna(0)

        group["news_count_3d"] = group["news_count_1d"].rolling(3, min_periods=1).sum()
        group["unique_domains_3d"] = group["unique_domains_1d"].rolling(3, min_periods=1).sum()

        group["news_count_7d"] = group["news_count_1d"].rolling(7, min_periods=1).sum()
        group["unique_domains_7d"] = group["unique_domains_1d"].rolling(7, min_periods=1).sum()

        all_frames.append(group.reset_index())

    features = pd.concat(all_frames, ignore_index=True)
    features = features.sort_values(["ticker", "date"]).reset_index(drop=True)

    return features
