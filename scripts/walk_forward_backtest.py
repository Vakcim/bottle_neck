from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


@dataclass(frozen=True)
class StrategyConfig:
    threshold: float = 0.50
    hold_days: int = 7
    max_positions: int = 3
    fee_round_trip: float = 0.001
    excluded_tickers: tuple[str, ...] = ("AFLT", "GAZP")


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * (periods_per_year ** 0.5))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def cagr(equity: pd.Series) -> float:
    equity = equity.dropna()
    if equity.empty:
        return 0.0

    years = max((equity.index.max() - equity.index.min()).days / 365.25, 1e-9)
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=20,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def add_probabilities(test: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    test = test.copy()
    X_test = test[FEATURE_COLUMNS].fillna(0)

    classes = list(model.named_steps["clf"].classes_)
    proba = model.predict_proba(X_test)

    for idx, cls in enumerate(classes):
        test[f"proba_{cls}"] = proba[:, idx]

    if "proba_1" not in test.columns:
        test["proba_1"] = 0.0

    return test


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

    return {
        "signal_date": row["date"],
        "entry_date": entry_row["date"],
        "exit_date": exit_row["date"],
        "ticker": row["ticker"],
        "proba_1": float(row["proba_1"]),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": exit_price / entry_price - 1.0,
    }


def run_portfolio_backtest(
    test: pd.DataFrame,
    config: StrategyConfig,
    initial_equity: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    excluded = set(config.excluded_tickers)

    local_test = test[~test["ticker"].isin(excluded)].copy()
    local_test = local_test.sort_values(["ticker", "date"]).reset_index(drop=True)

    if local_test.empty:
        return pd.DataFrame(), pd.DataFrame()

    ticker_data = {
        ticker: group.sort_values("date").reset_index(drop=True)
        for ticker, group in local_test.groupby("ticker")
    }

    raw_signals = local_test[local_test["proba_1"] >= config.threshold].copy()
    raw_signals = raw_signals.sort_values(
        ["date", "proba_1"],
        ascending=[True, False],
    )

    planned_trades = []

    for _, row in raw_signals.iterrows():
        ticker_df = ticker_data.get(row["ticker"])
        if ticker_df is None:
            continue

        plan = build_trade_plan(
            row=row,
            ticker_df=ticker_df,
            hold_days=config.hold_days,
        )

        if plan is not None:
            planned_trades.append(plan)

    if not planned_trades:
        all_dates = sorted(local_test["date"].unique())
        equity_df = pd.DataFrame(
            {
                "date": all_dates,
                "equity": initial_equity,
                "open_positions": 0,
            }
        )
        equity_df["date"] = pd.to_datetime(equity_df["date"], utc=True)
        equity_df = equity_df.set_index("date").sort_index()
        equity_df["daily_return"] = equity_df["equity"].pct_change().fillna(0)
        return equity_df, pd.DataFrame()

    signals = pd.DataFrame(planned_trades)
    signals = signals.sort_values(
        ["signal_date", "proba_1"],
        ascending=[True, False],
    )

    open_positions = []
    trades = []
    equity_rows = []
    equity = initial_equity

    all_dates = sorted(local_test["date"].unique())

    for current_date in all_dates:
        still_open = []

        for pos in open_positions:
            if pos["exit_date"] <= current_date:
                trade_return = pos["gross_return"] - config.fee_round_trip
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

        free_slots = config.max_positions - len(open_positions)

        if free_slots > 0:
            day_entries = signals[signals["entry_date"] == current_date].copy()
            day_entries = day_entries.sort_values("proba_1", ascending=False).head(free_slots)

            for _, row in day_entries.iterrows():
                if any(p["ticker"] == row["ticker"] for p in open_positions):
                    continue

                capital = equity / config.max_positions

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

    return equity_df, trades_df


def summarize_window(
    window_name: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> dict:
    if equity_df.empty:
        return {
            "window": window_name,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "trades": 0,
            "final_equity": 1.0,
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_trade_return": 0.0,
            "win_rate": 0.0,
        }

    if trades_df.empty:
        avg_trade_return = 0.0
        win_rate = 0.0
    else:
        avg_trade_return = float(trades_df["trade_return"].mean())
        win_rate = float((trades_df["trade_return"] > 0).mean())

    return {
        "window": window_name,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "trades": int(len(trades_df)),
        "final_equity": float(equity_df["equity"].iloc[-1]),
        "total_return": float(equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1.0),
        "cagr": cagr(equity_df["equity"]),
        "sharpe": sharpe_ratio(equity_df["daily_return"]),
        "max_drawdown": max_drawdown(equity_df["equity"]),
        "avg_trade_return": avg_trade_return,
        "win_rate": win_rate,
    }


def main():
    settings = get_settings()

    dataset_path = (
        settings.data_path
        / "datasets"
        / "model_dataset_day_h5_thr0.015.parquet"
    )

    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    config = StrategyConfig(
        threshold=0.50,
        hold_days=7,
        max_positions=3,
        fee_round_trip=0.001,
        excluded_tickers=("AFLT", "GAZP"),
    )

    # Расширяющееся окно:
    # train: с начала данных до train_end
    # test: следующий квартал
    windows = [
        ("2024Q1", "2023-04-14", "2023-12-31", "2024-01-01", "2024-04-01"),
        ("2024Q2", "2023-04-14", "2024-03-31", "2024-04-01", "2024-07-01"),
        ("2024Q3", "2023-04-14", "2024-06-30", "2024-07-01", "2024-10-01"),
        ("2024Q4", "2023-04-14", "2024-09-30", "2024-10-01", "2025-01-01"),
        ("2025Q1", "2023-04-14", "2024-12-31", "2025-01-01", "2025-04-01"),
        ("2025Q2", "2023-04-14", "2025-03-31", "2025-04-01", "2025-07-01"),
        ("2025Q3", "2023-04-14", "2025-06-30", "2025-07-01", "2025-10-01"),
        ("2025Q4", "2023-04-14", "2025-09-30", "2025-10-01", "2025-12-25"),
    ]

    all_summaries = []
    all_equity = []
    all_trades = []

    chained_equity = 1.0

    print("\nWalk-forward settings:")
    print(config)

    for window_name, train_start, train_end, test_start, test_end in windows:
        train_start_ts = pd.Timestamp(train_start, tz="UTC")
        train_end_ts = pd.Timestamp(train_end, tz="UTC")
        test_start_ts = pd.Timestamp(test_start, tz="UTC")
        test_end_ts = pd.Timestamp(test_end, tz="UTC")

        train = df[
            (df["date"] >= train_start_ts)
            & (df["date"] <= train_end_ts)
        ].copy()

        test = df[
            (df["date"] >= test_start_ts)
            & (df["date"] < test_end_ts)
        ].copy()

        if train.empty or test.empty:
            logger.warning(f"{window_name}: empty train/test, skipping")
            continue

        X_train = train[FEATURE_COLUMNS].fillna(0)
        y_train = train["target"]

        model = make_model()
        model.fit(X_train, y_train)

        test = add_probabilities(test, model)

        equity_df, trades_df = run_portfolio_backtest(
            test=test,
            config=config,
            initial_equity=chained_equity,
        )

        if not equity_df.empty:
            chained_equity = float(equity_df["equity"].iloc[-1])
            equity_df = equity_df.copy()
            equity_df["window"] = window_name
            all_equity.append(equity_df.reset_index())

        if not trades_df.empty:
            trades_df = trades_df.copy()
            trades_df["window"] = window_name
            all_trades.append(trades_df)

        summary = summarize_window(
            window_name=window_name,
            train_start=train_start_ts,
            train_end=train_end_ts,
            test_start=test_start_ts,
            test_end=test_end_ts,
            equity_df=equity_df,
            trades_df=trades_df,
        )

        all_summaries.append(summary)

        print(
            f"\n{window_name}: "
            f"trades={summary['trades']}, "
            f"return={summary['total_return']:.2%}, "
            f"sharpe={summary['sharpe']:.3f}, "
            f"mdd={summary['max_drawdown']:.2%}, "
            f"win_rate={summary['win_rate']:.2%}"
        )

    summary_df = pd.DataFrame(all_summaries)

    if all_equity:
        equity_all = pd.concat(all_equity, ignore_index=True)
        equity_all["date"] = pd.to_datetime(equity_all["date"], utc=True)
        equity_all = equity_all.sort_values("date")
    else:
        equity_all = pd.DataFrame()

    if all_trades:
        trades_all = pd.concat(all_trades, ignore_index=True)
    else:
        trades_all = pd.DataFrame()

    print("\nWalk-forward summary:")
    print(summary_df.to_string(index=False))

    if not equity_all.empty:
        equity_series = equity_all.drop_duplicates("date").set_index("date")["equity"].sort_index()
        daily_return = equity_series.pct_change().fillna(0)

        print("\nChained walk-forward metrics:")
        print(f"Final equity: {equity_series.iloc[-1]:.4f}")
        print(f"Total return: {equity_series.iloc[-1] - 1:.2%}")
        print(f"CAGR:         {cagr(equity_series):.2%}")
        print(f"Sharpe:       {sharpe_ratio(daily_return):.3f}")
        print(f"MaxDD:        {max_drawdown(equity_series):.2%}")
        print(f"Trades:       {len(trades_all)}")

        if not trades_all.empty:
            print(f"Avg trade:    {trades_all['trade_return'].mean():.2%}")
            print(f"Win rate:     {(trades_all['trade_return'] > 0).mean():.2%}")

            print("\nTrades by ticker:")
            print(
                trades_all.groupby("ticker")
                .agg(
                    trades=("ticker", "count"),
                    avg_return=("trade_return", "mean"),
                    win_rate=("trade_return", lambda x: (x > 0).mean()),
                )
                .sort_values("trades", ascending=False)
                .to_string()
            )

    out_dir = settings.data_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "walk_forward_summary.csv"
    equity_path = out_dir / "walk_forward_equity.csv"
    trades_path = out_dir / "walk_forward_trades.csv"

    summary_df.to_csv(summary_path, index=False)

    if not equity_all.empty:
        equity_all.to_csv(equity_path, index=False)

    if not trades_all.empty:
        trades_all.to_csv(trades_path, index=False)

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {equity_path}")
    print(f"Saved: {trades_path}")


if __name__ == "__main__":
    main()
