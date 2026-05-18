from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Iterable

import pandas as pd
from loguru import logger
from tinkoff.invest import CandleInterval, Client, InstrumentStatus
from tinkoff.invest.utils import quotation_to_decimal


INTERVAL_MAP: dict[str, CandleInterval] = {
    "1min": CandleInterval.CANDLE_INTERVAL_1_MIN,
    "5min": CandleInterval.CANDLE_INTERVAL_5_MIN,
    "15min": CandleInterval.CANDLE_INTERVAL_15_MIN,
    "hour": CandleInterval.CANDLE_INTERVAL_HOUR,
    "day": CandleInterval.CANDLE_INTERVAL_DAY,
}


@dataclass(frozen=True)
class Asset:
    ticker: str
    class_code: str = "TQBR"


@dataclass(frozen=True)
class InstrumentRef:
    ticker: str
    class_code: str
    figi: str
    uid: str
    name: str
    lot: int
    currency: str


def _decimal(value) -> Decimal:
    return quotation_to_decimal(value)


def _candle_to_dict(candle, instrument: InstrumentRef, interval: str) -> dict:
    return {
        "time": pd.Timestamp(candle.time).tz_convert("UTC"),
        "ticker": instrument.ticker,
        "class_code": instrument.class_code,
        "figi": instrument.figi,
        "uid": instrument.uid,
        "interval": interval,
        "open": float(_decimal(candle.open)),
        "high": float(_decimal(candle.high)),
        "low": float(_decimal(candle.low)),
        "close": float(_decimal(candle.close)),
        "volume": int(candle.volume),
        "is_complete": bool(candle.is_complete),
    }


class TInvestMarketDataClient:
    def __init__(self, token: str, request_pause_seconds: float = 0.25):
        self.token = token
        self.request_pause_seconds = request_pause_seconds

    def resolve_shares(self, assets: Iterable[Asset]) -> list[InstrumentRef]:
        requested = {(a.ticker.upper(), a.class_code.upper()) for a in assets}
        found: list[InstrumentRef] = []

        with Client(self.token) as client:
            response = client.instruments.shares(
                instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
            )

            for share in response.instruments:
                key = (share.ticker.upper(), share.class_code.upper())
                if key in requested:
                    found.append(
                        InstrumentRef(
                            ticker=share.ticker,
                            class_code=share.class_code,
                            figi=share.figi,
                            uid=share.uid,
                            name=share.name,
                            lot=share.lot,
                            currency=share.currency,
                        )
                    )

        missing = requested - {(x.ticker.upper(), x.class_code.upper()) for x in found}
        if missing:
            logger.warning(f"Не найдены инструменты: {sorted(missing)}")

        logger.info(f"Найдено инструментов: {len(found)}")
        return found

    def get_candles(
        self,
        instrument: InstrumentRef,
        from_: datetime,
        to: datetime,
        interval: str,
    ) -> pd.DataFrame:
        if interval not in INTERVAL_MAP:
            raise ValueError(f"Unsupported interval: {interval}. Use one of {list(INTERVAL_MAP)}")

        rows = []

        with Client(self.token) as client:
            logger.info(
                f"Загружаю свечи: {instrument.ticker} {interval} "
                f"from={from_.isoformat()} to={to.isoformat()}"
            )

            candles = client.get_all_candles(
                figi=instrument.figi,
                from_=from_,
                to=to,
                interval=INTERVAL_MAP[interval],
            )

            for candle in candles:
                rows.append(_candle_to_dict(candle, instrument, interval))

        time.sleep(self.request_pause_seconds)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "time", "ticker", "class_code", "figi", "uid", "interval",
                    "open", "high", "low", "close", "volume", "is_complete",
                ]
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("time").reset_index(drop=True)
        return df
