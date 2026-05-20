from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
from dotenv import load_dotenv
from tinkoff.invest import Client

from src.settings import get_settings


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def quotation_to_float(q) -> float:
    if q is None:
        return 0.0
    return float(q.units) + float(q.nano) / 1_000_000_000


def money_to_float(m) -> float:
    if m is None:
        return 0.0
    return float(m.units) + float(m.nano) / 1_000_000_000


def load_instruments(settings) -> pd.DataFrame:
    path = settings.data_path / "instruments" / "shares.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def get_ticker_for_figi(instruments: pd.DataFrame, figi: str) -> str | None:
    row = instruments[instruments["figi"] == figi]
    if row.empty:
        return None
    return str(row.iloc[0]["ticker"])


def load_or_create_tracker(path: Path) -> pd.DataFrame:
    columns = [
        "status",
        "ticker",
        "figi",
        "instrument_uid",
        "first_seen_at",
        "last_seen_at",
        "quantity",
        "quantity_lots",
        "average_position_price",
        "current_price",
        "expected_yield",
        "take_profit_price",
        "stop_loss_price",
        "planned_exit_date",
        "hold_days",
        "take_profit_pct",
        "stop_loss_pct",
        "source",
    ]

    if path.exists():
        df = pd.read_csv(
            path,
            parse_dates=[
                "first_seen_at",
                "last_seen_at",
                "planned_exit_date",
            ],
        )

        for col in columns:
            if col not in df.columns:
                df[col] = None

        return df[columns]

    return pd.DataFrame(columns=columns)


def get_latest_trading_dates(settings, ticker: str) -> list[pd.Timestamp]:
    live_path = settings.data_path / "live" / "live_features_day.parquet"
    if not live_path.exists():
        return []

    live = pd.read_parquet(live_path, columns=["date", "ticker"])
    live["date"] = pd.to_datetime(live["date"], utc=True)

    dates = (
        live[live["ticker"] == ticker]["date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return dates


def calc_planned_exit_date(
    settings,
    ticker: str,
    first_seen_at: pd.Timestamp,
    hold_days: int,
) -> pd.Timestamp | None:
    dates = get_latest_trading_dates(settings, ticker)

    if not dates:
        return None

    entry_idx = None
    for i, d in enumerate(dates):
        if d >= first_seen_at.floor("D"):
            entry_idx = i
            break

    if entry_idx is None:
        return None

    exit_idx = entry_idx + hold_days

    if exit_idx >= len(dates):
        return dates[-1]

    return dates[exit_idx]


def main():
    load_dotenv()

    token = os.getenv("TINVEST_TOKEN")
    account_id = os.getenv("TINVEST_SANDBOX_ACCOUNT_ID")

    if not token:
        raise RuntimeError("TINVEST_TOKEN is not set in .env")
    if not account_id:
        raise RuntimeError("TINVEST_SANDBOX_ACCOUNT_ID is not set in .env")

    settings = get_settings()
    cfg = load_yaml(Path("config/strategy_candidate_v1.yaml"))

    strategy_cfg = cfg["strategy"]
    execution_cfg = cfg.get("execution", {})

    hold_days = int(strategy_cfg["hold_days"])
    take_profit_pct = float(execution_cfg.get("take_profit_pct", 0.035))
    stop_loss_pct = float(execution_cfg.get("stop_loss_pct", 0.020))

    sandbox_dir = settings.data_path / "sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    tracker_path = sandbox_dir / "sandbox_positions_tracker.csv"
    report_path = sandbox_dir / "sandbox_position_tracker_report.csv"

    instruments = load_instruments(settings)
    tracker = load_or_create_tracker(tracker_path)

    now = pd.Timestamp.now(tz="UTC")

    current_rows = []

    with Client(token) as client:
        portfolio = client.sandbox.get_sandbox_portfolio(account_id=account_id)

        for pos in portfolio.positions:
            if pos.instrument_type != "share":
                continue

            quantity = quotation_to_float(pos.quantity)
            if quantity == 0:
                continue

            ticker = get_ticker_for_figi(instruments, pos.figi)
            if not ticker:
                continue

            avg_price = money_to_float(pos.average_position_price)
            current_price = money_to_float(pos.current_price)

            current_rows.append(
                {
                    "ticker": ticker,
                    "figi": pos.figi,
                    "instrument_uid": pos.instrument_uid,
                    "quantity": quantity,
                    "quantity_lots": quotation_to_float(pos.quantity_lots),
                    "average_position_price": avg_price,
                    "current_price": current_price,
                    "expected_yield": quotation_to_float(pos.expected_yield),
                }
            )

    current = pd.DataFrame(current_rows)

    if current.empty:
        if not tracker.empty and "status" in tracker.columns:
            tracker.loc[tracker["status"] == "open", "status"] = "closed_or_missing"
            tracker.loc[tracker["last_seen_at"].isna(), "last_seen_at"] = now

        tracker.to_csv(tracker_path, index=False)

        report = pd.DataFrame(
            [
                {
                    "run_at": now,
                    "open_positions": 0,
                    "tracked_rows": len(tracker),
                    "new_rows": 0,
                    "updated_rows": 0,
                    "closed_or_missing": int((tracker["status"] == "closed_or_missing").sum()) if not tracker.empty else 0,
                }
            ]
        )
        report.to_csv(report_path, index=False)

        print("No sandbox share positions.")
        print(f"Saved tracker: {tracker_path}")
        return

    new_rows = []
    updated_count = 0

    if tracker.empty:
        existing_open_keys = set()
    else:
        existing_open = tracker[tracker["status"] == "open"].copy()
        existing_open_keys = set(zip(existing_open["ticker"], existing_open["figi"]))

    current_keys = set(zip(current["ticker"], current["figi"]))

    if not tracker.empty:
        for idx, row in tracker[tracker["status"] == "open"].iterrows():
            key = (row["ticker"], row["figi"])

            if key not in current_keys:
                tracker.loc[idx, "status"] = "closed_or_missing"
                tracker.loc[idx, "last_seen_at"] = now
                continue

            cur = current[
                (current["ticker"] == row["ticker"])
                & (current["figi"] == row["figi"])
            ].iloc[0]

            avg_price = float(cur["average_position_price"])

            tracker.loc[idx, "last_seen_at"] = now
            tracker.loc[idx, "quantity"] = cur["quantity"]
            tracker.loc[idx, "quantity_lots"] = cur["quantity_lots"]
            tracker.loc[idx, "average_position_price"] = avg_price
            tracker.loc[idx, "current_price"] = cur["current_price"]
            tracker.loc[idx, "expected_yield"] = cur["expected_yield"]

            # Если старые строки были без TP/SL — дозаполняем.
            if pd.isna(tracker.loc[idx, "take_profit_price"]):
                tracker.loc[idx, "take_profit_price"] = avg_price * (1.0 + take_profit_pct)

            if pd.isna(tracker.loc[idx, "stop_loss_price"]):
                tracker.loc[idx, "stop_loss_price"] = avg_price * (1.0 - stop_loss_pct)

            if pd.isna(tracker.loc[idx, "take_profit_pct"]):
                tracker.loc[idx, "take_profit_pct"] = take_profit_pct

            if pd.isna(tracker.loc[idx, "stop_loss_pct"]):
                tracker.loc[idx, "stop_loss_pct"] = stop_loss_pct

            updated_count += 1

    for _, cur in current.iterrows():
        key = (cur["ticker"], cur["figi"])

        if key in existing_open_keys:
            continue

        first_seen_at = now.floor("D")
        planned_exit_date = calc_planned_exit_date(
            settings=settings,
            ticker=cur["ticker"],
            first_seen_at=first_seen_at,
            hold_days=hold_days,
        )

        avg_price = float(cur["average_position_price"])

        new_rows.append(
            {
                "status": "open",
                "ticker": cur["ticker"],
                "figi": cur["figi"],
                "instrument_uid": cur["instrument_uid"],
                "first_seen_at": first_seen_at,
                "last_seen_at": now,
                "quantity": cur["quantity"],
                "quantity_lots": cur["quantity_lots"],
                "average_position_price": avg_price,
                "current_price": cur["current_price"],
                "expected_yield": cur["expected_yield"],
                "take_profit_price": avg_price * (1.0 + take_profit_pct),
                "stop_loss_price": avg_price * (1.0 - stop_loss_pct),
                "planned_exit_date": planned_exit_date,
                "hold_days": hold_days,
                "take_profit_pct": take_profit_pct,
                "stop_loss_pct": stop_loss_pct,
                "source": "sandbox_portfolio",
            }
        )

    if new_rows:
        tracker = pd.concat([tracker, pd.DataFrame(new_rows)], ignore_index=True)

    tracker.to_csv(tracker_path, index=False)

    report = pd.DataFrame(
        [
            {
                "run_at": now,
                "open_positions": int((tracker["status"] == "open").sum()),
                "tracked_rows": len(tracker),
                "new_rows": len(new_rows),
                "updated_rows": updated_count,
                "closed_or_missing": int((tracker["status"] == "closed_or_missing").sum()),
            }
        ]
    )
    report.to_csv(report_path, index=False)

    print("\nSandbox position tracker report:")
    print(report.to_string(index=False))

    print("\nCurrent sandbox share positions:")
    print(current.to_string(index=False))

    print("\nTracker open positions:")
    open_tracker = tracker[tracker["status"] == "open"].copy()
    if open_tracker.empty:
        print("None")
    else:
        cols = [
            "status",
            "ticker",
            "quantity",
            "quantity_lots",
            "average_position_price",
            "current_price",
            "expected_yield",
            "take_profit_price",
            "stop_loss_price",
            "first_seen_at",
            "planned_exit_date",
        ]
        print(open_tracker[cols].to_string(index=False))

    print(f"\nSaved tracker: {tracker_path}")
    print(f"Saved report:  {report_path}")


if __name__ == "__main__":
    main()
