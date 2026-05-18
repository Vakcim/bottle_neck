from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
import typer
import yaml
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.settings import get_settings


app = typer.Typer(help="Download GDELT news for configured tickers")


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def gdelt_datetime(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_queries(company_names: list[str]) -> list[str]:
    """
    GDELT плохо переносит большие OR-запросы и короткие слова.
    Поэтому делаем несколько простых запросов по названиям компании.
    """
    queries = []

    for name in company_names:
        name = str(name).strip()
        if len(name) < 4:
            continue

        # MOEX, SBER и прочие короткие тикеры лучше не отправлять как единственный термин.
        if name.isupper() and len(name) <= 4:
            continue

        queries.append(f'"{name}"')

    # Дедупликация с сохранением порядка
    return list(dict.fromkeys(queries))


def make_session(ignore_proxy: bool = False) -> requests.Session:
    session = requests.Session()

    if ignore_proxy:
        session.trust_env = False

    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def fetch_gdelt(
    session: requests.Session,
    query: str,
    from_dt: datetime,
    to_dt: datetime,
    max_records: int = 50,
    timeout: int = 90,
) -> pd.DataFrame:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sort": "datedesc",
        "startdatetime": gdelt_datetime(from_dt),
        "enddatetime": gdelt_datetime(to_dt),
    }

    response = session.get(url, params=params, timeout=timeout)

    if response.status_code == 429:
        logger.warning(f"429 Too Many Requests for query={query}. Sleeping 60s.")
        time.sleep(60)
        response = session.get(url, params=params, timeout=timeout)

    if response.status_code != 200:
        logger.warning(f"GDELT status={response.status_code}, query={query}, text={response.text[:200]}")
        return pd.DataFrame()

    content_type = response.headers.get("content-type", "")
    text_head = response.text[:200]

    if "json" not in content_type.lower() and not response.text.strip().startswith("{"):
        logger.warning(f"GDELT returned non-json for query={query}: {text_head!r}")
        return pd.DataFrame()

    try:
        payload = response.json()
    except Exception:
        logger.warning(f"JSON parse failed for query={query}: {text_head!r}")
        return pd.DataFrame()

    articles = payload.get("articles", [])

    rows = []
    for item in articles:
        rows.append(
            {
                "published_at": item.get("seendate"),
                "title": item.get("title"),
                "url": item.get("url"),
                "domain": item.get("domain"),
                "source_country": item.get("sourcecountry"),
                "language": item.get("language"),
                "query": query,
            }
        )

    return pd.DataFrame(rows)


@app.command()
def main(
    from_date: str = typer.Option(..., help="Start date, e.g. 2025-01-01"),
    to_date: str = typer.Option(..., help="End date, e.g. 2025-12-31"),
    config_path: Path = typer.Option(
        Path("config/ticker_news_map.yaml"),
        help="Path to ticker_news_map.yaml",
    ),
    max_records: int = typer.Option(50, help="Max GDELT records per query"),
    pause_seconds: float = typer.Option(8.0, help="Pause between requests"),
    ignore_proxy: bool = typer.Option(False, help="Ignore system proxy variables"),
):
    settings = get_settings()
    cfg = load_yaml(config_path)

    from_dt = parse_date(from_date)
    to_dt = parse_date(to_date)

    out_dir = settings.data_path / "news" / "gdelt"
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = cfg.get("tickers", {})
    if not tickers:
        raise typer.BadParameter("No tickers found in config/ticker_news_map.yaml")

    session = make_session(ignore_proxy=ignore_proxy)

    for ticker, info in tickers.items():
        company_names = info.get("company_names", [])
        queries = build_queries(company_names)

        if not queries:
            logger.warning(f"{ticker}: no valid queries")
            continue

        frames = []

        for query in queries:
            logger.info(f"{ticker}: {query}")

            try:
                df = fetch_gdelt(
                    session=session,
                    query=query,
                    from_dt=from_dt,
                    to_dt=to_dt,
                    max_records=max_records,
                )

                if not df.empty:
                    df["ticker"] = ticker
                    frames.append(df)
                    logger.info(f"{ticker}: query={query}, rows={len(df)}")
                else:
                    logger.warning(f"{ticker}: query={query}, no rows")

            except Exception as exc:
                logger.warning(f"{ticker}: query={query}, failed: {exc}")

            time.sleep(pause_seconds)

        if not frames:
            logger.warning(f"{ticker}: no news saved")
            continue

        result = pd.concat(frames, ignore_index=True)
        result["published_at"] = pd.to_datetime(result["published_at"], errors="coerce", utc=True)
        result = result.dropna(subset=["published_at"])
        result = result.drop_duplicates(subset=["ticker", "url"]).sort_values("published_at")

        out_path = out_dir / f"{ticker}_gdelt.parquet"
        result.to_parquet(out_path, index=False)

        logger.info(f"{ticker}: saved {len(result)} rows to {out_path}")


if __name__ == "__main__":
    app()