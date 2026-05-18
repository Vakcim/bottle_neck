from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.settings import get_settings


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def cagr(equity: pd.Series) -> float:
    years = max((equity.index.max() - equity.index.min()).days / 365.25, 1e-9)
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * (252 ** 0.5))


def main():
    settings = get_settings()
    candles_dir = settings.data_path / "candles"

    frames = []

    for path in sorted(candles_dir.glob("*_day.parquet")):
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["time"], utc=True).dt.floor("D")
        df = df[df["date"] >= "2025-01-01"].copy()
        df = df.sort_values("date")
        df["daily_return"] = df["close"].pct_change()
        frames.append(df[["date", "ticker", "close", "daily_return"]])

    data = pd.concat(frames, ignore_index=True)

    rows = []

    for ticker, group in data.groupby("ticker"):
        group = group.sort_values("date").copy()
        equity = (1 + group["daily_return"].fillna(0)).cumprod()
        equity.index = group["date"]

        rows.append(
            {
                "ticker": ticker,
                "total_return": float(equity.iloc[-1] - 1),
                "cagr": cagr(equity),
                "sharpe": sharpe(group["daily_return"]),
                "max_drawdown": max_drawdown(equity),
            }
        )

    # Equal-weight портфель из всех тикеров
    pivot = data.pivot_table(index="date", columns="ticker", values="daily_return")
    equal_weight_returns = pivot.fillna(0).mean(axis=1)
    equal_equity = (1 + equal_weight_returns).cumprod()

    rows.append(
        {
            "ticker": "EQUAL_WEIGHT_ALL",
            "total_return": float(equal_equity.iloc[-1] - 1),
            "cagr": cagr(equal_equity),
            "sharpe": sharpe(equal_weight_returns),
            "max_drawdown": max_drawdown(equal_equity),
        }
    )

    report = pd.DataFrame(rows).sort_values("total_return", ascending=False)

    print(report.to_string(index=False))

    out_path = settings.data_path / "reports" / "buy_hold_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
