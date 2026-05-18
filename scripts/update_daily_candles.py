from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import typer
from loguru import logger

from src.connectors.tinkoff_client import Asset, TInvestMarketDataClient
from src.data.storage import LocalStorage
from src.settings import get_settings, load_yaml


app = typer.Typer(help="Update daily candles from T-Invest API")


def get_last_candle_date(storage: LocalStorage, ticker: str, interval: str = "day") -> datetime | None:
    path = storage.candle_path(ticker, interval)

    if not path.exists():
        return None

    df = pd.read_parquet(path, columns=["time"])

    if df.empty:
        return None

    last_time = pd.to_datetime(df["time"], utc=True).max()
    return last_time.to_pydatetime()


@app.command()
def main(
    assets_config: Path = typer.Option(
        Path("config/assets.yaml"),
        help="Path to assets.yaml",
    ),
    settings_config: Path = typer.Option(
        Path("config/settings.yaml"),
        help="Path to settings.yaml",
    ),
    lookback_days: int = typer.Option(
        10,
        help="How many days before last candle to re-download for safety",
    ),
):
    settings = get_settings()
    cfg = load_yaml(settings_config)
    assets_yaml = load_yaml(assets_config)

    request_pause = (
        cfg.get("market_data", {})
        .get("request_pause_seconds", 0.25)
    )

    assets = [
        Asset(
            ticker=item["ticker"],
            class_code=item.get("class_code", "TQBR"),
        )
        for item in assets_yaml.get("assets", [])
    ]

    if not assets:
        raise typer.BadParameter("No assets found in config/assets.yaml")

    storage = LocalStorage(settings.data_path)

    client = TInvestMarketDataClient(
        token=settings.tinvest_token,
        request_pause_seconds=float(request_pause),
    )

    instruments = client.resolve_shares(assets)
    storage.write_instruments([x.__dict__ for x in instruments])

    now_utc = datetime.now(timezone.utc)

    # Берём с запасом до завтрашнего дня, чтобы API точно включил свежие свечи.
    to_dt = now_utc + timedelta(days=1)

    logger.info(f"Update daily candles until {to_dt.isoformat()}")

    for instrument in instruments:
        try:
            last_dt = get_last_candle_date(storage, instrument.ticker, interval="day")

            if last_dt is None:
                from_dt = now_utc - timedelta(days=365 * 3)
                logger.info(f"{instrument.ticker}: no local data, downloading from {from_dt.date()}")
            else:
                from_dt = last_dt - timedelta(days=lookback_days)
                logger.info(
                    f"{instrument.ticker}: last local candle={last_dt.date()}, "
                    f"downloading from {from_dt.date()}"
                )

            df = client.get_candles(
                instrument=instrument,
                from_=from_dt,
                to=to_dt,
                interval="day",
            )

            storage.write_candles(
                df,
                ticker=instrument.ticker,
                interval="day",
            )

        except Exception as exc:
            logger.exception(f"{instrument.ticker}: update failed: {exc}")

    logger.info("Daily candle update done")


if __name__ == "__main__":
    app()
