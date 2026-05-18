from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
from loguru import logger

from src.settings import get_settings


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_latest_signal_file(signals_dir: Path, strategy_name: str) -> Path | None:
    files = sorted(signals_dir.glob(f"signals_*_{strategy_name}.csv"))
    if not files:
        return None
    return files[-1]


def load_or_create_positions(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=["signal_date", "entry_date", "planned_exit_date"])

    return pd.DataFrame(
        columns=[
            "status",
            "ticker",
            "signal_date",
            "entry_date",
            "planned_exit_date",
            "entry_price",
            "shares",
            "capital",
            "proba_1",
        ]
    )


def load_or_create_trades(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=["signal_date", "entry_date", "exit_date"])

    return pd.DataFrame(
        columns=[
            "ticker",
            "signal_date",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "shares",
            "capital",
            "proba_1",
            "trade_return",
            "pnl",
        ]
    )


def load_or_create_equity(path: Path, initial_equity: float) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])

    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp.now(tz="UTC").floor("D"),
                "cash": initial_equity,
                "positions_value": 0.0,
                "equity": initial_equity,
            }
        ]
    )


def get_ticker_history(live: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return live[live["ticker"] == ticker].sort_values("date").reset_index(drop=True)


def get_close_on_or_before(live: pd.DataFrame, ticker: str, date: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    hist = live[(live["ticker"] == ticker) & (live["date"] <= date)].sort_values("date")

    if hist.empty:
        return None

    row = hist.iloc[-1]
    return row["date"], float(row["close"])


def get_entry_and_exit_dates(
    live: pd.DataFrame,
    ticker: str,
    signal_date: pd.Timestamp,
    hold_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp, float] | None:
    hist = get_ticker_history(live, ticker)
    dates = hist["date"].tolist()

    try:
        signal_idx = dates.index(signal_date)
    except ValueError:
        return None

    entry_idx = signal_idx + 1
    exit_idx = entry_idx + hold_days

    if entry_idx >= len(hist):
        return None

    entry_row = hist.iloc[entry_idx]
    entry_date = entry_row["date"]
    entry_price = float(entry_row["close"])

    if exit_idx < len(hist):
        planned_exit_date = hist.iloc[exit_idx]["date"]
    else:
        planned_exit_date = hist.iloc[-1]["date"]

    return entry_date, planned_exit_date, entry_price


def main():
    settings = get_settings()

    strategy_path = Path("config/strategy_candidate_v1.yaml")
    cfg = load_yaml(strategy_path)

    strategy_cfg = cfg["strategy"]
    strategy_name = strategy_cfg["name"]
    hold_days = int(strategy_cfg["hold_days"])
    max_positions = int(strategy_cfg["max_positions"])
    threshold = float(strategy_cfg["threshold"])
    fee_round_trip = float(strategy_cfg.get("fee_round_trip", 0.001))
    excluded_tickers = set(strategy_cfg.get("excluded_tickers", []))

    initial_equity = 100_000.0

    live_path = settings.data_path / "live" / "live_features_day.parquet"
    signals_dir = settings.data_path / "signals"
    paper_dir = settings.data_path / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    positions_path = paper_dir / "positions.csv"
    trades_path = paper_dir / "trades.csv"
    equity_path = paper_dir / "equity.csv"
    report_path = paper_dir / "paper_report.csv"

    live = pd.read_parquet(live_path)
    live["date"] = pd.to_datetime(live["date"], utc=True)
    live = live.sort_values(["date", "ticker"]).reset_index(drop=True)

    latest_live_date = live["date"].max()

    positions = load_or_create_positions(positions_path)
    trades = load_or_create_trades(trades_path)
    equity_history = load_or_create_equity(equity_path, initial_equity)

    latest_signal_file = find_latest_signal_file(signals_dir, strategy_name)

    if latest_signal_file is None:
        print("No signal files found.")
        return

    signals = pd.read_csv(latest_signal_file)
    signals["date"] = pd.to_datetime(signals["date"], utc=True)

    # Файл может содержать топ вероятностей даже без сигналов.
    signals = signals[
        (signals["proba_1"] >= threshold)
        & (~signals["ticker"].isin(excluded_tickers))
    ].copy()

    if "status" not in positions.columns:
        positions["status"] = "open"

    open_positions = positions[positions["status"] == "open"].copy()

    # 1. Закрываем позиции, если наступила planned_exit_date.
    closed_rows = []

    for idx, pos in open_positions.iterrows():
        ticker = pos["ticker"]
        planned_exit_date = pd.Timestamp(pos["planned_exit_date"])

        if latest_live_date < planned_exit_date:
            continue

        close_info = get_close_on_or_before(live, ticker, latest_live_date)

        if close_info is None:
            continue

        exit_date, exit_price = close_info

        entry_price = float(pos["entry_price"])
        shares = float(pos["shares"])
        capital = float(pos["capital"])

        trade_return = exit_price / entry_price - 1.0 - fee_round_trip
        pnl = capital * trade_return

        closed_rows.append(
            {
                "ticker": ticker,
                "signal_date": pos["signal_date"],
                "entry_date": pos["entry_date"],
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": shares,
                "capital": capital,
                "proba_1": float(pos["proba_1"]),
                "trade_return": trade_return,
                "pnl": pnl,
            }
        )

        positions.loc[idx, "status"] = "closed"

    if closed_rows:
        trades = pd.concat([trades, pd.DataFrame(closed_rows)], ignore_index=True)

    open_positions = positions[positions["status"] == "open"].copy()

    # 2. Считаем текущую equity.
    realized_pnl = trades["pnl"].sum() if not trades.empty else 0.0

    open_value = 0.0
    open_unrealized_pnl = 0.0

    for _, pos in open_positions.iterrows():
        ticker = pos["ticker"]
        close_info = get_close_on_or_before(live, ticker, latest_live_date)

        if close_info is None:
            continue

        _, current_price = close_info

        entry_price = float(pos["entry_price"])
        capital = float(pos["capital"])

        current_return = current_price / entry_price - 1.0
        current_value = capital * (1.0 + current_return)

        open_value += current_value
        open_unrealized_pnl += current_value - capital

    equity = initial_equity + realized_pnl + open_unrealized_pnl
    used_capital = open_positions["capital"].sum() if not open_positions.empty else 0.0
    cash = equity - used_capital

    # 3. Открываем новые позиции по сигналам, если есть свободные слоты.
    free_slots = max_positions - len(open_positions)

    opened_rows = []

    if free_slots > 0 and not signals.empty:
        signals = signals.sort_values("proba_1", ascending=False).head(free_slots)

        already_open = set(open_positions["ticker"].tolist())

        for _, sig in signals.iterrows():
            ticker = sig["ticker"]

            if ticker in already_open:
                continue

            signal_date = pd.Timestamp(sig["date"])

            plan = get_entry_and_exit_dates(
                live=live,
                ticker=ticker,
                signal_date=signal_date,
                hold_days=hold_days,
            )

            if plan is None:
                continue

            entry_date, planned_exit_date, entry_price = plan

            # В paper-режиме открываем только если entry_date уже известна в live_features.
            if entry_date > latest_live_date:
                continue

            capital = equity / max_positions
            shares = capital / entry_price

            opened_rows.append(
                {
                    "status": "open",
                    "ticker": ticker,
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "planned_exit_date": planned_exit_date,
                    "entry_price": entry_price,
                    "shares": shares,
                    "capital": capital,
                    "proba_1": float(sig["proba_1"]),
                }
            )

    if opened_rows:
        positions = pd.concat([positions, pd.DataFrame(opened_rows)], ignore_index=True)

    # 4. Пересчитываем equity после возможного открытия.
    open_positions = positions[positions["status"] == "open"].copy()

    realized_pnl = trades["pnl"].sum() if not trades.empty else 0.0
    open_value = 0.0
    open_unrealized_pnl = 0.0

    for _, pos in open_positions.iterrows():
        ticker = pos["ticker"]
        close_info = get_close_on_or_before(live, ticker, latest_live_date)

        if close_info is None:
            continue

        _, current_price = close_info

        entry_price = float(pos["entry_price"])
        capital = float(pos["capital"])

        current_return = current_price / entry_price - 1.0
        current_value = capital * (1.0 + current_return)

        open_value += current_value
        open_unrealized_pnl += current_value - capital

    equity = initial_equity + realized_pnl + open_unrealized_pnl
    used_capital = open_positions["capital"].sum() if not open_positions.empty else 0.0
    cash = equity - used_capital

    new_equity_row = pd.DataFrame(
        [
            {
                "date": latest_live_date,
                "cash": cash,
                "positions_value": open_value,
                "equity": equity,
            }
        ]
    )

    equity_history = pd.concat([equity_history, new_equity_row], ignore_index=True)
    equity_history = equity_history.drop_duplicates(subset=["date"], keep="last")
    equity_history = equity_history.sort_values("date")

    # 5. Сохраняем всё.
    positions.to_csv(positions_path, index=False)
    trades.to_csv(trades_path, index=False)
    equity_history.to_csv(equity_path, index=False)

    report = {
        "run_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "latest_live_date": latest_live_date,
        "strategy": strategy_name,
        "threshold": threshold,
        "hold_days": hold_days,
        "max_positions": max_positions,
        "open_positions": len(open_positions),
        "closed_trades_total": len(trades),
        "opened_today": len(opened_rows),
        "closed_today": len(closed_rows),
        "realized_pnl": realized_pnl,
        "open_unrealized_pnl": open_unrealized_pnl,
        "equity": equity,
        "total_return": equity / initial_equity - 1.0,
    }

    pd.DataFrame([report]).to_csv(report_path, index=False)

    print("\nPaper portfolio report:")
    for k, v in report.items():
        print(f"{k}: {v}")

    print("\nOpened today:")
    if opened_rows:
        print(pd.DataFrame(opened_rows).to_string(index=False))
    else:
        print("None")

    print("\nClosed today:")
    if closed_rows:
        print(pd.DataFrame(closed_rows).to_string(index=False))
    else:
        print("None")

    print("\nOpen positions:")
    open_positions = positions[positions["status"] == "open"].copy()
    if open_positions.empty:
        print("None")
    else:
        print(open_positions.to_string(index=False))

    print(f"\nSaved positions: {positions_path}")
    print(f"Saved trades:    {trades_path}")
    print(f"Saved equity:    {equity_path}")
    print(f"Saved report:    {report_path}")

    logger.info("Done")


if __name__ == "__main__":
    main()
