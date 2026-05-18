from __future__ import annotations

import typer

from src.data.storage import LocalStorage
from src.settings import get_settings


app = typer.Typer(help="Inspect local candle storage")


@app.command()
def main():
    settings = get_settings()
    storage = LocalStorage(settings.data_path)
    df = storage.inspect_candles()

    if df.empty:
        print("Свечей пока нет.")
        return

    print(df.to_string(index=False))


if __name__ == "__main__":
    app()
