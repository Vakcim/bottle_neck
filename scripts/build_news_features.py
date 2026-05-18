from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from loguru import logger

from src.features.news_features import (
    build_daily_news_features,
    load_all_gdelt_monthly,
)
from src.settings import get_settings


def main():
    settings = get_settings()

    news_dir = settings.data_path / "news" / "gdelt_monthly"
    out_dir = settings.data_path / "features" / "news"
    out_dir.mkdir(parents=True, exist_ok=True)

    news = load_all_gdelt_monthly(news_dir)

    if news.empty:
        print("Нет новостей для расчёта признаков.")
        return

    features = build_daily_news_features(news)

    out_path = out_dir / "daily_news_features.parquet"
    features.to_parquet(out_path, index=False)

    logger.info(f"Saved {len(features)} rows to {out_path}")
    print(features.groupby("ticker").size().to_string())


if __name__ == "__main__":
    main()
