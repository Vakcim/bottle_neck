from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger


class LocalStorage:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.candles_dir = self.root_dir / "candles"
        self.news_dir = self.root_dir / "news"
        self.instruments_dir = self.root_dir / "instruments"

        for path in [self.root_dir, self.candles_dir, self.news_dir, self.instruments_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def candle_path(self, ticker: str, interval: str) -> Path:
        return self.candles_dir / f"{ticker.upper()}_{interval}.parquet"

    def write_candles(self, df: pd.DataFrame, ticker: str, interval: str) -> Path:
        path = self.candle_path(ticker, interval)

        if df.empty:
            logger.warning(f"Нет свечей для записи: {ticker} {interval}")
            return path

        if path.exists():
            old = pd.read_parquet(path)
            merged = pd.concat([old, df], ignore_index=True)
            merged = (
                merged.drop_duplicates(subset=["time", "ticker", "interval"])
                .sort_values("time")
                .reset_index(drop=True)
            )
        else:
            merged = df.sort_values("time").reset_index(drop=True)

        merged.to_parquet(path, index=False)
        logger.info(f"Записано {len(merged)} строк: {path}")
        return path

    def write_instruments(self, instruments: list[dict]) -> Path:
        path = self.instruments_dir / "shares.parquet"
        df = pd.DataFrame(instruments)
        df.to_parquet(path, index=False)
        logger.info(f"Записан справочник инструментов: {path}")
        return path

    def list_candle_files(self) -> list[Path]:
        return sorted(self.candles_dir.glob("*.parquet"))

    def inspect_candles(self) -> pd.DataFrame:
        files = self.list_candle_files()
        if not files:
            return pd.DataFrame(columns=["file", "ticker", "interval", "rows", "min_time", "max_time"])

        rows = []
        for file in files:
            df = pd.read_parquet(file, columns=["time", "ticker", "interval"])
            rows.append(
                {
                    "file": file.name,
                    "ticker": df["ticker"].iloc[0] if len(df) else None,
                    "interval": df["interval"].iloc[0] if len(df) else None,
                    "rows": len(df),
                    "min_time": df["time"].min() if len(df) else None,
                    "max_time": df["time"].max() if len(df) else None,
                }
            )

        return pd.DataFrame(rows)

    def duckdb_query(self, query: str) -> pd.DataFrame:
        con = duckdb.connect()
        return con.execute(query).df()
