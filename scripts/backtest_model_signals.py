from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from loguru import logger

from src.settings import get_settings


FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "hl_range",
    "oc_return",
    "volume_change_1",
    "volatility_12",
    "volatility_24",
    "close_to_ma_12",
    "close_to_ma_24",
    "close_to_ma_72",
    "news_count_1d",
    "unique_domains_1d",
    "english_news_count_1d",
    "russian_news_count_1d",
    "news_count_3d",
    "unique_domains_3d",
    "news_count_7d",
    "unique_domains_7d",
]


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * (periods_per_year ** 0.5))


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def main():
    settings = get_settings()

    dataset_path = (
        settings.data_path
        / "datasets"
        / "model_dataset_day_h5_thr0.015.parquet"
    )
    model_path = settings.data_path / "models" / "baseline_random_forest.joblib"

    df = pd.read_parquet(dataset_path)
    model = joblib.load(model_path)

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    test = df[df["date"] >= "2025-01-01"].copy()

    X_test = test[FEATURE_COLUMNS].fillna(0)

    classes = list(model.named_steps["clf"].classes_)
    proba = model.predict_proba(X_test)

    for idx, cls in enumerate(classes):
        test[f"proba_{cls}"] = proba[:, idx]

    # Для торговли в лонг нас интересует вероятность класса 1.
    thresholds = [0.34, 0.38, 0.42, 0.46, 0.50, 0.55, 0.60]

    rows = []

    for threshold in thresholds:
        trades = test[test["proba_1"] >= threshold].copy()

        if trades.empty:
            rows.append(
                {
                    "threshold": threshold,
                    "trades": 0,
                    "avg_return": None,
                    "median_return": None,
                    "win_rate": None,
                    "total_return_simple": None,
                    "sharpe": None,
                    "max_drawdown": None,
                }
            )
            continue

        # future_return уже посчитан на 5 дней вперёд.
        trades["trade_return"] = trades["future_return"]

        # Грубая комиссия/проскальзывание.
        # Позже заменим на реалистичную модель исполнения.
        fee = 0.001  # 0.1% round-trip estimate
        trades["trade_return_net"] = trades["trade_return"] - fee

        # Упрощённая equity: сделки как независимые последовательные наблюдения.
        # Это не финальный бэктест портфеля, но хороший первый фильтр качества сигнала.
        equity = (1.0 + trades["trade_return_net"]).cumprod()

        rows.append(
            {
                "threshold": threshold,
                "trades": len(trades),
                "avg_return": trades["trade_return_net"].mean(),
                "median_return": trades["trade_return_net"].median(),
                "win_rate": (trades["trade_return_net"] > 0).mean(),
                "total_return_simple": equity.iloc[-1] - 1.0,
                "sharpe": sharpe_ratio(trades["trade_return_net"]),
                "max_drawdown": max_drawdown(equity),
            }
        )

    report = pd.DataFrame(rows)

    print("\nProbability threshold backtest:")
    print(report.to_string(index=False))

    out_dir = settings.data_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "baseline_probability_backtest.csv"
    report.to_csv(out_path, index=False)

    logger.info(f"Saved report to {out_path}")

    print("\nTop 20 strongest long signals:")
    cols = [
        "date",
        "ticker",
        "close",
        "proba_1",
        "future_return",
        "target",
        "news_count_7d",
    ]
    print(
        test.sort_values("proba_1", ascending=False)[cols]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
