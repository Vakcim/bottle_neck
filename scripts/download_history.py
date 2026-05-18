from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
from loguru import logger

from src.connectors.tinkoff_client import Asset, TInvestMarketDataClient
from src.data.storage import LocalStorage
from src.settings import get_settings, load_yaml


app = typer.Typer(help="Download historical candles from T-Invest API")


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


@app.command()
def main(
    from_date: str = typer.Option(..., help="Start date, e.g. 2023-01-01"),
    to_date: str = typer.Option(..., help="End date, e.g. 2025-12-31"),
    interval: str = typer.Option("day", help="1min, 5min, 15min, hour, day"),
    assets_config: Path = typer.Option(Path("config/assets.yaml"), help="Path to assets.yaml"),
    settings_config: Path = typer.Option(Path("config/settings.yaml"), help="Path to settings.yaml"),
):
    settings = get_settings()
    cfg = load_yaml(settings_config)
    assets_yaml = load_yaml(assets_config)

    request_pause = (
        cfg.get("market_data", {})
        .get("request_pause_seconds", 0.25)
    )

    assets = [
        Asset(ticker=item["ticker"], class_code=item.get("class_code", "TQBR"))
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

    from_dt = parse_date(from_date)
    to_dt = parse_date(to_date)

    for instrument in instruments:
        try:
            df = client.get_candles(
                instrument=instrument,
                from_=from_dt,
                to=to_dt,
                interval=interval,
            )
            storage.write_candles(df, ticker=instrument.ticker, interval=interval)
        except Exception as exc:
            logger.exception(f"Ошибка загрузки {instrument.ticker}: {exc}")

    logger.info("Готово")


if __name__ == "__main__":
    app()
