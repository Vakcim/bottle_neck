from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
import typer
import yaml
from dateutil.relativedelta import relativedelta
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.settings import get_settings


app = typer.Typer(help="Download GDELT news by monthly windows")


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def gdelt_datetime(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def month_windows(from_dt: datetime, to_dt: datetime):
    current = from_dt
    while current < to_dt:
        nxt = min(current + relativedelta(months=1), to_dt)
        yield current, nxt
        current = nxt


def build_queries(company_names: list[str]) -> list[str]:
    queries = []
    for name in company_names:
        name = str(name).strip()
        if len(name) < 5:
            continue
        if name.isupper() and len(name) <= 4:
            continue
        queries.append(f'"{name}"')
    return list(dict.fromkeys(queries))


def make_session(ignore_proxy: bool = False) -> requests.Session:
    session = requests.Session()

    if ignore_proxy:
        session.trust_env = False

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=4,
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
    max_records: int,
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
        logger.warning(f"429 Too Many Requests: {query}. Sleeping 90s.")
        time.sleep(90)
        response = session.get(url, params=params, timeout=timeout)

    if response.status_code != 200:
        logger.warning(
            f"GDELT status={response.status_code}, "
            f"query={query}, text={response.text[:160]!r}"
        )
        return pd.DataFrame()

    text = response.text.strip()
    if not text.startswith("{"):
        logger.warning(f"GDELT non-json: query={query}, text={text[:160]!r}")
        return pd.DataFrame()

    try:
        payload = response.json()
    except Exception:
        logger.warning(f"GDELT json parse failed: query={query}, text={text[:160]!r}")
        return pd.DataFrame()

    rows = []
    for item in payload.get("articles", []):
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
    config_path: Path = typer.Option(Path("config/ticker_news_map.yaml")),
    max_records: int = typer.Option(20, help="Max records per query per month"),
    pause_seconds: float = typer.Option(10.0, help="Pause between requests"),
    ignore_proxy: bool = typer.Option(False),
):
    settings = get_settings()
    cfg = load_yaml(config_path)

    from_dt = parse_date(from_date)
    to_dt = parse_date(to_date)

    out_dir = settings.data_path / "news" / "gdelt_monthly"
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = cfg.get("tickers", {})
    session = make_session(ignore_proxy=ignore_proxy)

    for ticker, info in tickers.items():
        queries = build_queries(info.get("company_names", []))

        if not queries:
            logger.warning(f"{ticker}: no valid queries")
            continue

        frames = []

        for window_start, window_end in month_windows(from_dt, to_dt):
            for query in queries:
                logger.info(
                    f"{ticker}: {query}, "
                    f"{window_start.date()} → {window_end.date()}"
                )

                try:
                    df = fetch_gdelt(
                        session=session,
                        query=query,
                        from_dt=window_start,
                        to_dt=window_end,
                        max_records=max_records,
                    )

                    if not df.empty:
                        df["ticker"] = ticker
                        df["window_start"] = window_start
                        df["window_end"] = window_end
                        frames.append(df)
                        logger.info(f"{ticker}: rows={len(df)}")
                    else:
                        logger.warning(f"{ticker}: no rows")

                except Exception as exc:
                    logger.warning(f"{ticker}: failed: {exc}")

                time.sleep(pause_seconds)

        if not frames:
            logger.warning(f"{ticker}: no news saved")
            continue

        result = pd.concat(frames, ignore_index=True)
        result["published_at"] = pd.to_datetime(
            result["published_at"],
            errors="coerce",
            utc=True,
        )
        result = result.dropna(subset=["published_at"])
        result = result.drop_duplicates(subset=["ticker", "url"])
        result = result.sort_values("published_at")

        out_path = out_dir / f"{ticker}_gdelt_monthly.parquet"
        result.to_parquet(out_path, index=False)

        logger.info(f"{ticker}: saved {len(result)} rows to {out_path}")


if __name__ == "__main__":
    app()
