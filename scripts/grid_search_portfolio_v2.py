from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import itertools

import joblib
import pandas as pd

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
    years = max((equity.index.max() - equity.index.min()).days / 365.25, 1e-9)
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def prepare_test_data(df: pd.DataFrame, model) -> pd.DataFrame:
    test = df[df["date"] >= "2025-01-01"].copy()
    X_test = test[FEATURE_COLUMNS].fillna(0)

    classes = list(model.named_steps["clf"].classes_)
    proba = model.predict_proba(X_test)

    for idx, cls in enumerate(classes):
        test[f"proba_{cls}"] = proba[:, idx]

    return test.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_trade_plan(row: pd.Series, ticker_df: pd.DataFrame, hold_days: int) -> dict | None:
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

    return {
        "signal_date": row["date"],
        "entry_date": entry_row["date"],
        "exit_date": exit_row["date"],
        "ticker": row["ticker"],
        "proba_1": row["proba_1"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": exit_price / entry_price - 1.0,
    }


def run_backtest(
    test: pd.DataFrame,
    threshold: float,
    hold_days: int,
    max_positions: int,
    fee_round_trip: float,
    excluded_tickers: set[str],
) -> dict:
    local_test = test[~test["ticker"].isin(excluded_tickers)].copy()

    if local_test.empty:
        return {}

    ticker_data = {
        ticker: group.sort_values("date").reset_index(drop=True)
        for ticker, group in local_test.groupby("ticker")
    }

    raw_signals = local_test[local_test["proba_1"] >= threshold].copy()
    raw_signals = raw_signals.sort_values(["date", "proba_1"], ascending=[True, False])

    planned = []

    for _, row in raw_signals.iterrows():
        plan = build_trade_plan(row, ticker_data[row["ticker"]], hold_days)
        if plan:
            planned.append(plan)

    if not planned:
        return {
            "threshold": threshold,
            "hold_days": hold_days,
            "max_positions": max_positions,
            "excluded": ",".join(sorted(excluded_tickers)) or "none",
            "trades": 0,
            "final_equity": 1.0,
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_trade_return": 0.0,
            "win_rate": 0.0,
        }

    signals = pd.DataFrame(planned)
    signals = signals.sort_values(["signal_date", "proba_1"], ascending=[True, False])

    open_positions = []
    trades = []
    equity_rows = []
    equity = 1.0

    all_dates = sorted(local_test["date"].unique())

    for current_date in all_dates:
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

    if trades_df.empty:
        avg_trade_return = 0.0
        win_rate = 0.0
    else:
        avg_trade_return = float(trades_df["trade_return"].mean())
        win_rate = float((trades_df["trade_return"] > 0).mean())

    return {
        "threshold": threshold,
        "hold_days": hold_days,
        "max_positions": max_positions,
        "excluded": ",".join(sorted(excluded_tickers)) or "none",
        "trades": int(len(trades_df)),
        "final_equity": float(equity_df["equity"].iloc[-1]),
        "total_return": float(equity_df["equity"].iloc[-1] - 1.0),
        "cagr": cagr(equity_df["equity"]),
        "sharpe": sharpe_ratio(equity_df["daily_return"]),
        "max_drawdown": max_drawdown(equity_df["equity"]),
        "avg_trade_return": avg_trade_return,
        "win_rate": win_rate,
    }


def main():
    settings = get_settings()

    dataset_path = settings.data_path / "datasets" / "model_dataset_day_h5_thr0.015.parquet"
    model_path = settings.data_path / "models" / "baseline_random_forest.joblib"

    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    model = joblib.load(model_path)
    test = prepare_test_data(df, model)

    thresholds = [0.50, 0.55, 0.60, 0.65]
    hold_days_list = [3, 5, 7, 10]
    max_positions_list = [1, 2, 3]
    exclusion_sets = [
        set(),
        {"AFLT"},
        {"GAZP"},
        {"AFLT", "GAZP"},
    ]

    fee_round_trip = 0.001

    rows = []

    for threshold, hold_days, max_positions, excluded in itertools.product(
        thresholds,
        hold_days_list,
        max_positions_list,
        exclusion_sets,
    ):
        result = run_backtest(
            test=test,
            threshold=threshold,
            hold_days=hold_days,
            max_positions=max_positions,
            fee_round_trip=fee_round_trip,
            excluded_tickers=excluded,
        )

        if result:
            rows.append(result)

    report = pd.DataFrame(rows)
    report = report.sort_values(
        ["sharpe", "total_return", "trades"],
        ascending=[False, False, False],
    )

    out_dir = settings.data_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "grid_search_portfolio_v2.csv"
    report.to_csv(out_path, index=False)

    print("\nTop 30 parameter sets:")
    print(report.head(30).to_string(index=False))

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
