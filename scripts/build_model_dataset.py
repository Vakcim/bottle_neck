from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import typer
from loguru import logger

from src.settings import get_settings


app = typer.Typer(help="Build model dataset from market and news features")


def make_target(
    df: pd.DataFrame,
    horizon_days: int,
    threshold: float,
) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()

    df["future_close"] = df.groupby("ticker")["close"].shift(-horizon_days)
    df["future_return"] = df["future_close"] / df["close"] - 1.0

    df["target"] = 0
    df.loc[df["future_return"] > threshold, "target"] = 1
    df.loc[df["future_return"] < -threshold, "target"] = -1

    return df


@app.command()
def main(
    horizon_days: int = typer.Option(5, help="Prediction horizon in trading days"),
    threshold: float = typer.Option(0.015, help="Return threshold for target class"),
):
    settings = get_settings()

    market_dir = settings.data_path / "features" / "market"
    news_path = settings.data_path / "features" / "news" / "daily_news_features.parquet"

    out_dir = settings.data_path / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    market_files = sorted(market_dir.glob("*_day.parquet"))

    if not market_files:
        raise FileNotFoundError(f"No market feature files found in {market_dir}")

    frames = []

    for path in market_files:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["time"], utc=True).dt.floor("D")
        frames.append(df)

    market = pd.concat(frames, ignore_index=True)
    market["date"] = pd.to_datetime(market["date"], utc=True).dt.floor("D")

    if news_path.exists():
        news = pd.read_parquet(news_path)
        news["date"] = pd.to_datetime(news["date"], utc=True).dt.floor("D")
    else:
        logger.warning("News features not found. Filling news columns with zeros.")
        news = pd.DataFrame(columns=["ticker", "date"])

    dataset = market.merge(news, on=["ticker", "date"], how="left")

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
        if col not in dataset.columns:
            dataset[col] = 0
        dataset[col] = dataset[col].fillna(0)

    dataset = make_target(
        dataset,
        horizon_days=horizon_days,
        threshold=threshold,
    )

    # Убираем строки, где не хватает истории для признаков или будущей цены.
    dataset = dataset.dropna(
        subset=[
            "return_1",
            "return_3",
            "return_6",
            "return_12",
            "volatility_12",
            "volatility_24",
            "ma_12",
            "ma_24",
            "ma_72",
            "future_return",
        ]
    )

    dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)

    out_path = out_dir / f"model_dataset_day_h{horizon_days}_thr{threshold}.parquet"
    dataset.to_parquet(out_path, index=False)

    logger.info(f"Saved {len(dataset)} rows to {out_path}")

    print("\nDataset shape:", dataset.shape)
    print("\nTarget distribution:")
    print(dataset["target"].value_counts().sort_index().to_string())

    print("\nDate range:")
    print(dataset["date"].min(), "→", dataset["date"].max())

    print("\nNews non-zero rows:")
    print((dataset["news_count_7d"] > 0).sum())


if __name__ == "__main__":
    app()
