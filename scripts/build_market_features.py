from __future__ import annotations

import pandas as pd
import typer
from loguru import logger

from src.data.storage import LocalStorage
from src.features.market_features import add_basic_market_features
from src.settings import get_settings


app = typer.Typer(help="Build basic market features from stored candles")


@app.command()
def main():
    settings = get_settings()
    storage = LocalStorage(settings.data_path)

    features_dir = storage.root_dir / "features" / "market"
    features_dir.mkdir(parents=True, exist_ok=True)

    for path in storage.list_candle_files():
        df = pd.read_parquet(path)
        features = add_basic_market_features(df)
        out_path = features_dir / path.name
        features.to_parquet(out_path, index=False)
        logger.info(f"Features written: {out_path}")


if __name__ == "__main__":
    app()
