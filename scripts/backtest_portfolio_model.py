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


def cagr(equity: pd.Series) -> float:
    equity = equity.dropna()
    if equity.empty:
        return 0.0

    start = equity.index.min()
    end = equity.index.max()
    years = max((end - start).days / 365.25, 1e-9)

    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def main():
    settings = get_settings()

    dataset_path = (
        settings.data_path
        / "datasets"
        / "model_dataset_day_h5_thr0.015.parquet"
    )
    model_path = settings.data_path / "models" / "baseline_random_forest.joblib"

    threshold = 0.55
    hold_days = 5
    max_positions = 3
    fee_round_trip = 0.001

    df = pd.read_parquet(dataset_path)
    model = joblib.load(model_path)

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    test = df[df["date"] >= "2025-01-01"].copy()
    X_test = test[FEATURE_COLUMNS].fillna(0)

    classes = list(model.named_steps["clf"].classes_)
    proba = model.predict_proba(X_test)

    for idx, cls in enumerate(classes):
        test[f"proba_{cls}"] = proba[:, idx]

    # Следующая дата входа и цена выхода через hold_days уже примерно отражены в future_return.
    # Для первой портфельной версии используем future_return как доходность сделки.
    # В следующей версии заменим на точный вход на next close/open.
    signals = test[test["proba_1"] >= threshold].copy()
    signals = signals.sort_values(["date", "proba_1"], ascending=[True, False])

    open_positions = []
    trades = []
    equity_rows = []

    equity = 1.0

    all_dates = sorted(test["date"].unique())

    for current_date in all_dates:
        # 1. Закрываем позиции, срок которых истёк.
        still_open = []

        for pos in open_positions:
            if pos["exit_date"] <= current_date:
                trade_return = pos["future_return"] - fee_round_trip
                pnl = pos["capital"] * trade_return
                equity += pnl

                trades.append(
                    {
                        "entry_date": pos["entry_date"],
                        "exit_date": current_date,
                        "ticker": pos["ticker"],
                        "proba_1": pos["proba_1"],
                        "capital": pos["capital"],
                        "trade_return": trade_return,
                        "pnl": pnl,
                        "equity_after": equity,
                    }
                )
            else:
                still_open.append(pos)

        open_positions = still_open

        # 2. Открываем новые позиции по лучшим сигналам дня.
        free_slots = max_positions - len(open_positions)

        if free_slots > 0:
            day_signals = signals[signals["date"] == current_date].copy()
            day_signals = day_signals.sort_values("proba_1", ascending=False).head(free_slots)

            for _, row in day_signals.iterrows():
                # Не открываем второй раз тот же тикер, если он уже открыт.
                if any(p["ticker"] == row["ticker"] for p in open_positions):
                    continue

                ticker_dates = test[test["ticker"] == row["ticker"]]["date"].sort_values().tolist()

                try:
                    current_idx = ticker_dates.index(current_date)
                    exit_idx = current_idx + hold_days
                    if exit_idx >= len(ticker_dates):
                        continue
                    exit_date = ticker_dates[exit_idx]
                except ValueError:
                    continue

                capital = equity / max_positions

                open_positions.append(
                    {
                        "entry_date": current_date,
                        "exit_date": exit_date,
                        "ticker": row["ticker"],
                        "proba_1": row["proba_1"],
                        "future_return": row["future_return"],
                        "capital": capital,
                    }
                )

        equity_rows.append(
            {
                "date": current_date,
                "equity": equity,
                "open_positions": len(open_positions),
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    equity_df["date"] = pd.to_datetime(equity_df["date"], utc=True)
    equity_df = equity_df.set_index("date").sort_index()
    equity_df["daily_return"] = equity_df["equity"].pct_change().fillna(0)

    trades_df = pd.DataFrame(trades)

    print("\nPortfolio backtest settings:")
    print(f"threshold={threshold}")
    print(f"hold_days={hold_days}")
    print(f"max_positions={max_positions}")
    print(f"fee_round_trip={fee_round_trip}")

    print("\nPortfolio metrics:")
    print(f"Final equity:     {equity_df['equity'].iloc[-1]:.4f}")
    print(f"Total return:     {equity_df['equity'].iloc[-1] - 1:.2%}")
    print(f"CAGR:             {cagr(equity_df['equity']):.2%}")
    print(f"Sharpe:           {sharpe_ratio(equity_df['daily_return']):.3f}")
    print(f"Max drawdown:     {max_drawdown(equity_df['equity']):.2%}")
    print(f"Trades:           {len(trades_df)}")

    if not trades_df.empty:
        print(f"Avg trade return: {trades_df['trade_return'].mean():.2%}")
        print(f"Median return:    {trades_df['trade_return'].median():.2%}")
        print(f"Win rate:         {(trades_df['trade_return'] > 0).mean():.2%}")

        print("\nTrades by ticker:")
        print(
            trades_df.groupby("ticker")
            .agg(
                trades=("ticker", "count"),
                avg_return=("trade_return", "mean"),
                win_rate=("trade_return", lambda x: (x > 0).mean()),
            )
            .sort_values("trades", ascending=False)
            .to_string()
        )

        print("\nLast 20 trades:")
        print(trades_df.tail(20).to_string(index=False))

    out_dir = settings.data_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_path = out_dir / "portfolio_backtest_equity.csv"
    trades_path = out_dir / "portfolio_backtest_trades.csv"

    equity_df.to_csv(equity_path)
    trades_df.to_csv(trades_path, index=False)

    logger.info(f"Saved equity to {equity_path}")
    logger.info(f"Saved trades to {trades_path}")


if __name__ == "__main__":
    main()
