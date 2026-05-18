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


def build_trade_plan(
    row: pd.Series,
    ticker_df: pd.DataFrame,
    hold_days: int,
) -> dict | None:
    ticker_dates = ticker_df["date"].tolist()

    try:
        signal_idx = ticker_dates.index(row["date"])
    except ValueError:
        return None

    entry_idx = signal_idx + 1
    exit_idx = entry_idx + hold_days

    if exit_idx >= len(ticker_df):
        return None

    entry_row = ticker_df.iloc[entry_idx]
    exit_row = ticker_df.iloc[exit_idx]

    entry_price = float(entry_row["close"])
    exit_price = float(exit_row["close"])

    if entry_price <= 0 or exit_price <= 0:
        return None

    gross_return = exit_price / entry_price - 1.0

    return {
        "signal_date": row["date"],
        "entry_date": entry_row["date"],
        "exit_date": exit_row["date"],
        "ticker": row["ticker"],
        "proba_1": row["proba_1"],
        "signal_close": float(row["close"]),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": gross_return,
    }


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

    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    test = df[df["date"] >= "2025-01-01"].copy()
    X_test = test[FEATURE_COLUMNS].fillna(0)

    classes = list(model.named_steps["clf"].classes_)
    proba = model.predict_proba(X_test)

    for idx, cls in enumerate(classes):
        test[f"proba_{cls}"] = proba[:, idx]

    ticker_data = {
        ticker: group.sort_values("date").reset_index(drop=True)
        for ticker, group in test.groupby("ticker")
    }

    raw_signals = test[test["proba_1"] >= threshold].copy()
    raw_signals = raw_signals.sort_values(["date", "proba_1"], ascending=[True, False])

    planned_trades = []

    for _, row in raw_signals.iterrows():
        ticker_df = ticker_data[row["ticker"]]
        plan = build_trade_plan(row, ticker_df, hold_days=hold_days)
        if plan is not None:
            planned_trades.append(plan)

    signals = pd.DataFrame(planned_trades)

    if signals.empty:
        print("No valid signals.")
        return

    signals = signals.sort_values(["signal_date", "proba_1"], ascending=[True, False])

    open_positions = []
    trades = []
    equity_rows = []
    equity = 1.0

    all_dates = sorted(test["date"].unique())

    for current_date in all_dates:
        # Закрываем позиции.
        still_open = []

        for pos in open_positions:
            if pos["exit_date"] <= current_date:
                trade_return = pos["gross_return"] - fee_round_trip
                pnl = pos["capital"] * trade_return
                equity += pnl

                trades.append(
                    {
                        **pos,
                        "trade_return": trade_return,
                        "pnl": pnl,
                        "equity_after": equity,
                    }
                )
            else:
                still_open.append(pos)

        open_positions = still_open

        # Открываем позиции, у которых entry_date сегодня.
        free_slots = max_positions - len(open_positions)

        if free_slots > 0:
            day_entries = signals[signals["entry_date"] == current_date].copy()
            day_entries = day_entries.sort_values("proba_1", ascending=False).head(free_slots)

            for _, row in day_entries.iterrows():
                if any(p["ticker"] == row["ticker"] for p in open_positions):
                    continue

                capital = equity / max_positions

                open_positions.append(
                    {
                        "signal_date": row["signal_date"],
                        "entry_date": row["entry_date"],
                        "exit_date": row["exit_date"],
                        "ticker": row["ticker"],
                        "proba_1": row["proba_1"],
                        "capital": capital,
                        "signal_close": row["signal_close"],
                        "entry_price": row["entry_price"],
                        "exit_price": row["exit_price"],
                        "gross_return": row["gross_return"],
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

    print("\nPortfolio backtest v2 settings:")
    print(f"threshold={threshold}")
    print(f"hold_days={hold_days}")
    print(f"max_positions={max_positions}")
    print(f"fee_round_trip={fee_round_trip}")
    print("entry: next trading close after signal")
    print("exit: close after hold_days trading days")

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
        cols = [
            "signal_date",
            "entry_date",
            "exit_date",
            "ticker",
            "proba_1",
            "signal_close",
            "entry_price",
            "exit_price",
            "trade_return",
            "pnl",
            "equity_after",
        ]
        print(trades_df[cols].tail(20).to_string(index=False))

    out_dir = settings.data_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_path = out_dir / "portfolio_backtest_v2_equity.csv"
    trades_path = out_dir / "portfolio_backtest_v2_trades.csv"

    equity_df.to_csv(equity_path)
    trades_df.to_csv(trades_path, index=False)

    logger.info(f"Saved equity to {equity_path}")
    logger.info(f"Saved trades to {trades_path}")


if __name__ == "__main__":
    main()
