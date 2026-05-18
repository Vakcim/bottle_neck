from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd
import requests
from loguru import logger


class GDELTClient:
    '''
    Минимальная заготовка для новостей.

    В v0.1 используется как scaffold. На следующем этапе добавим:
    - нормальный query builder;
    - сопоставление новостей с тикерами;
    - дедупликацию;
    - тональность;
    - сохранение в parquet.
    '''

    def __init__(self, base_url: str = "https://api.gdeltproject.org/api/v2/doc/doc"):
        self.base_url = base_url

    def search_articles(
        self,
        query: str,
        start: datetime | None = None,
        end: datetime | None = None,
        max_records: int = 250,
    ) -> pd.DataFrame:
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": max_records,
            "sort": "datedesc",
        }

        if start:
            params["startdatetime"] = start.strftime("%Y%m%d%H%M%S")
        if end:
            params["enddatetime"] = end.strftime("%Y%m%d%H%M%S")

        logger.info(f"GDELT query: {query}")

        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()

        payload = response.json()
        articles = payload.get("articles", [])

        rows = []
        for item in articles:
            rows.append(
                {
                    "published_at": item.get("seendate"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "source_country": item.get("sourcecountry"),
                    "domain": item.get("domain"),
                    "language": item.get("language"),
                    "query": query,
                }
            )

        return pd.DataFrame(rows)
