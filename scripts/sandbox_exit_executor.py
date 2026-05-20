from __future__ import annotations

import os
import sys
import uuid
from decimal import Decimal
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import load_dotenv
from tinkoff.invest import Client, OrderDirection, OrderType
from tinkoff.invest.utils import decimal_to_quotation

from src.settings import get_settings


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


def get_figi_for_ticker(instruments: pd.DataFrame, ticker: str) -> str:
    row = instruments[instruments["ticker"] == ticker]
    if row.empty:
        raise RuntimeError(f"FIGI not found for ticker={ticker}")
    return str(row.iloc[0]["figi"])


def load_execution_cfg() -> dict:
    import yaml

    path = Path("config/strategy_candidate_v1.yaml")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    return cfg.get("execution", {})


def get_latest_live_date(settings) -> pd.Timestamp:
    live_path = settings.data_path / "live" / "live_features_day.parquet"
    if not live_path.exists():
        raise FileNotFoundError(live_path)

    live = pd.read_parquet(live_path, columns=["date"])
    live["date"] = pd.to_datetime(live["date"], utc=True)

    return live["date"].max()


def get_latest_close(settings, ticker: str) -> float:
    live_path = settings.data_path / "live" / "live_features_day.parquet"
    if not live_path.exists():
        raise FileNotFoundError(live_path)

    live = pd.read_parquet(live_path)
    live["date"] = pd.to_datetime(live["date"], utc=True)

    row = live[live["ticker"] == ticker].sort_values("date").tail(1)

    if row.empty:
        raise RuntimeError(f"No live close for ticker={ticker}")

    return float(row.iloc[0]["close"])


def get_sandbox_share_positions(
    client: Client,
    account_id: str,
    instruments: pd.DataFrame,
) -> dict[str, dict]:
    portfolio = client.sandbox.get_sandbox_portfolio(account_id=account_id)

    positions: dict[str, dict] = {}

    for pos in portfolio.positions:
        if pos.instrument_type != "share":
            continue

        quantity = quotation_to_float(pos.quantity)
        quantity_lots = quotation_to_float(pos.quantity_lots)

        if quantity <= 0 or quantity_lots <= 0:
            continue

        ticker = get_ticker_for_figi(instruments, pos.figi)
        if not ticker:
            continue

        positions[ticker] = {
            "ticker": ticker,
            "figi": pos.figi,
            "instrument_uid": pos.instrument_uid,
            "quantity": quantity,
            "quantity_lots": quantity_lots,
            "average_position_price": money_to_float(pos.average_position_price),
            "current_price": money_to_float(pos.current_price),
            "expected_yield": quotation_to_float(pos.expected_yield),
        }

    return positions


def get_active_sell_order_tickers(
    client: Client,
    account_id: str,
    instruments: pd.DataFrame,
) -> set[str]:
    orders = client.sandbox.get_sandbox_orders(account_id=account_id).orders

    tickers: set[str] = set()

    for order in orders:
        direction = str(order.direction)
        if "SELL" not in direction:
            continue

        ticker = get_ticker_for_figi(instruments, order.figi)
        if ticker:
            tickers.add(ticker)

    return tickers


def decide_exit_reason(
    latest_close: float,
    take_profit_price: float | None,
    stop_loss_price: float | None,
    planned_exit_date: pd.Timestamp | None,
    latest_live_date: pd.Timestamp,
) -> str | None:
    if take_profit_price is not None and latest_close >= take_profit_price:
        return "take_profit"

    if stop_loss_price is not None and latest_close <= stop_loss_price:
        return "stop_loss"

    if planned_exit_date is not None and planned_exit_date <= latest_live_date:
        return "time_exit"

    return None


def main(place_orders: bool = False, order_type: str = "limit"):
    load_dotenv()

    token = os.getenv("TINVEST_TOKEN")
    account_id = os.getenv("TINVEST_SANDBOX_ACCOUNT_ID")

    if not token:
        raise RuntimeError("TINVEST_TOKEN is not set in .env")
    if not account_id:
        raise RuntimeError("TINVEST_SANDBOX_ACCOUNT_ID is not set in .env")

    settings = get_settings()
    instruments = load_instruments(settings)
    execution_cfg = load_execution_cfg()

    sell_limit_offset_pct = Decimal(str(execution_cfg.get("sell_limit_offset_pct", 0.005)))

    sandbox_dir = settings.data_path / "sandbox"
    tracker_path = sandbox_dir / "sandbox_positions_tracker.csv"
    out_path = sandbox_dir / "sandbox_exit_orders_log.csv"

    if not tracker_path.exists():
        raise FileNotFoundError(tracker_path)

    tracker = pd.read_csv(
        tracker_path,
        parse_dates=["first_seen_at", "last_seen_at", "planned_exit_date"],
    )

    if tracker.empty:
        print("Tracker is empty. No exits.")
        return

    latest_live_date = get_latest_live_date(settings)

    open_tracker = tracker[tracker["status"] == "open"].copy()

    if open_tracker.empty:
        print("No open tracked positions. No exits.")
        return

    exit_candidates = []

    for _, row in open_tracker.iterrows():
        ticker = str(row["ticker"])
        latest_close = get_latest_close(settings, ticker)

        take_profit_price = None
        stop_loss_price = None
        planned_exit_date = None

        if "take_profit_price" in row and not pd.isna(row["take_profit_price"]):
            take_profit_price = float(row["take_profit_price"])

        if "stop_loss_price" in row and not pd.isna(row["stop_loss_price"]):
            stop_loss_price = float(row["stop_loss_price"])

        if "planned_exit_date" in row and not pd.isna(row["planned_exit_date"]):
            planned_exit_date = pd.Timestamp(row["planned_exit_date"])
            if planned_exit_date.tzinfo is None:
                planned_exit_date = planned_exit_date.tz_localize("UTC")
            else:
                planned_exit_date = planned_exit_date.tz_convert("UTC")

        exit_reason = decide_exit_reason(
            latest_close=latest_close,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            planned_exit_date=planned_exit_date,
            latest_live_date=latest_live_date,
        )

        if exit_reason is None:
            continue

        item = row.to_dict()
        item["latest_close"] = latest_close
        item["exit_reason"] = exit_reason
        exit_candidates.append(item)

    if not exit_candidates:
        print("No positions due for TP/SL/time exit.")
        print(f"Latest live date: {latest_live_date}")
        return

    due = pd.DataFrame(exit_candidates)

    exit_logs = []

    with Client(token) as client:
        sandbox_positions = get_sandbox_share_positions(
            client=client,
            account_id=account_id,
            instruments=instruments,
        )

        active_sell_tickers = get_active_sell_order_tickers(
            client=client,
            account_id=account_id,
            instruments=instruments,
        )

        print("\nSandbox exit state:")
        print(f"Latest live date: {latest_live_date}")
        print(f"Exit candidates: {len(due)}")
        print(f"Sandbox position tickers: {sorted(sandbox_positions.keys())}")
        print(f"Active sell order tickers: {sorted(active_sell_tickers)}")

        for _, row in due.iterrows():
            ticker = str(row["ticker"])
            figi = get_figi_for_ticker(instruments, ticker)

            latest_close = Decimal(str(row["latest_close"]))

            record = {
                "run_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "ticker": ticker,
                "figi": figi,
                "exit_reason": row["exit_reason"],
                "planned_exit_date": row.get("planned_exit_date"),
                "latest_live_date": latest_live_date,
                "take_profit_price": row.get("take_profit_price"),
                "stop_loss_price": row.get("stop_loss_price"),
                "place_orders": place_orders,
                "order_type": order_type,
                "skipped": False,
                "skipped_reason": "",
                "quantity_lots": None,
                "latest_close": float(latest_close),
                "limit_price": None,
                "order_id": str(uuid.uuid4()),
                "sandbox_response": "",
            }

            if ticker not in sandbox_positions:
                record["skipped"] = True
                record["skipped_reason"] = "no_sandbox_position"
                record["sandbox_response"] = "skipped"
                exit_logs.append(record)
                print("\nSkipped exit:")
                print(record)
                continue

            if ticker in active_sell_tickers:
                record["skipped"] = True
                record["skipped_reason"] = "active_sell_order_exists"
                record["sandbox_response"] = "skipped"
                exit_logs.append(record)
                print("\nSkipped exit:")
                print(record)
                continue

            pos = sandbox_positions[ticker]
            quantity_lots = int(pos["quantity_lots"])

            if quantity_lots <= 0:
                record["skipped"] = True
                record["skipped_reason"] = "zero_quantity_lots"
                record["sandbox_response"] = "skipped"
                exit_logs.append(record)
                print("\nSkipped exit:")
                print(record)
                continue

            # Для SELL limit ставим немного ниже последней цены,
            # чтобы повысить шанс исполнения.
            limit_price = latest_close * (Decimal("1.0") - sell_limit_offset_pct)

            record["quantity_lots"] = quantity_lots
            record["limit_price"] = float(limit_price)

            print("\nPlanned sandbox exit order:")
            print(record)

            if place_orders:
                if order_type == "market":
                    response = client.sandbox.post_sandbox_order(
                        figi=figi,
                        quantity=quantity_lots,
                        price=None,
                        direction=OrderDirection.ORDER_DIRECTION_SELL,
                        account_id=account_id,
                        order_type=OrderType.ORDER_TYPE_MARKET,
                        order_id=record["order_id"],
                    )
                else:
                    response = client.sandbox.post_sandbox_order(
                        figi=figi,
                        quantity=quantity_lots,
                        price=decimal_to_quotation(limit_price),
                        direction=OrderDirection.ORDER_DIRECTION_SELL,
                        account_id=account_id,
                        order_type=OrderType.ORDER_TYPE_LIMIT,
                        order_id=record["order_id"],
                    )

                record["sandbox_response"] = str(response)
                print("Exit order sent:")
                print(response)

                active_sell_tickers.add(ticker)

                # Не помечаем tracker как closed сразу.
                # Позиция будет закрыта только после фактического исчезновения из sandbox portfolio.
            else:
                record["sandbox_response"] = "dry_run"

            exit_logs.append(record)

    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    result = pd.concat([existing, pd.DataFrame(exit_logs)], ignore_index=True)
    result.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--place-orders",
        action="store_true",
        help="Actually send sandbox exit orders. Without this flag, dry-run only.",
    )
    parser.add_argument(
        "--order-type",
        choices=["limit", "market"],
        default="limit",
        help="Exit order type.",
    )

    args = parser.parse_args()

    main(
        place_orders=args.place_orders,
        order_type=args.order_type,
    )
