from __future__ import annotations

import pandas as pd


def add_basic_market_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Базовые признаки для следующего этапа.

    На входе свечи одного тикера и одного интервала.
    '''
    if df.empty:
        return df

    out = df.sort_values("time").copy()

    out["return_1"] = out["close"].pct_change(1)
    out["return_3"] = out["close"].pct_change(3)
    out["return_6"] = out["close"].pct_change(6)
    out["return_12"] = out["close"].pct_change(12)

    out["hl_range"] = (out["high"] - out["low"]) / out["close"]
    out["oc_return"] = (out["close"] - out["open"]) / out["open"]

    out["volume_change_1"] = out["volume"].pct_change(1)
    out["volatility_12"] = out["return_1"].rolling(12).std()
    out["volatility_24"] = out["return_1"].rolling(24).std()

    out["ma_12"] = out["close"].rolling(12).mean()
    out["ma_24"] = out["close"].rolling(24).mean()
    out["ma_72"] = out["close"].rolling(72).mean()

    out["close_to_ma_12"] = out["close"] / out["ma_12"] - 1
    out["close_to_ma_24"] = out["close"] / out["ma_24"] - 1
    out["close_to_ma_72"] = out["close"] / out["ma_72"] - 1

    return out
