from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
from dotenv import load_dotenv
from pandas.errors import EmptyDataError
from tinkoff.invest import Client, OrderDirection, OrderType
from tinkoff.invest.utils import decimal_to_quotation

from src.settings import get_settings


EXECUTOR_VERSION = "sandbox_executor_order_intent_v1"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def append_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    existing = read_csv_or_empty(path)
    result = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    result.to_csv(path, index=False)


def find_latest_signal_file(signals_dir: Path, strategy_name: str) -> Path | None:
    files = sorted(signals_dir.glob(f"signals_*_{strategy_name}.csv"))
    if not files:
        return None
    return files[-1]


def load_instruments(settings) -> pd.DataFrame:
    path = settings.data_path / "instruments" / "shares.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def get_figi_for_ticker(instruments: pd.DataFrame, ticker: str) -> str:
    row = instruments[instruments["ticker"] == ticker]
    if row.empty:
        raise RuntimeError(f"FIGI not found for ticker={ticker}")
    return str(row.iloc[0]["figi"])


def get_ticker_for_figi(instruments: pd.DataFrame, figi: str) -> str | None:
    row = instruments[instruments["figi"] == figi]
    if row.empty:
        return None
    return str(row.iloc[0]["ticker"])


def get_lot_for_ticker(instruments: pd.DataFrame, ticker: str) -> int:
    row = instruments[instruments["ticker"] == ticker]
    if row.empty:
        raise RuntimeError(f"Lot not found for ticker={ticker}")
    return int(row.iloc[0]["lot"])


def quotation_to_float(q) -> float:
    if q is None:
        return 0.0
    return float(q.units) + float(q.nano) / 1_000_000_000


def get_latest_signals(
    settings,
    strategy_name: str,
    threshold: float,
    excluded_tickers: set[str],
) -> pd.DataFrame:
    signals_dir = settings.data_path / "signals"
    latest_file = find_latest_signal_file(signals_dir, strategy_name)

    if latest_file is None:
        raise RuntimeError(f"No signal files found in {signals_dir}")

    signals = pd.read_csv(latest_file)
    signals["date"] = pd.to_datetime(signals["date"], utc=True)

    signals = signals[
        (signals["proba_1"] >= threshold)
        & (~signals["ticker"].isin(excluded_tickers))
    ].copy()

    signals = signals.sort_values("proba_1", ascending=False)

    return signals


def get_active_order_tickers(client: Client, account_id: str, instruments: pd.DataFrame) -> set[str]:
    orders = client.sandbox.get_sandbox_orders(account_id=account_id).orders

    tickers: set[str] = set()

    for order in orders:
        ticker = get_ticker_for_figi(instruments, order.figi)
        if ticker:
            tickers.add(ticker)

    return tickers


def get_position_tickers(client: Client, account_id: str, instruments: pd.DataFrame) -> set[str]:
    portfolio = client.sandbox.get_sandbox_portfolio(account_id=account_id)

    tickers: set[str] = set()

    for pos in portfolio.positions:
        # RUB000UTSTOM — рубли, не акция.
        if pos.instrument_type != "share":
            continue

        quantity = quotation_to_float(pos.quantity)
        if quantity == 0:
            continue

        ticker = get_ticker_for_figi(instruments, pos.figi)
        if ticker:
            tickers.add(ticker)

    return tickers


def normalize_order_intents_columns(intents: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "intent_id",
        "created_at",
        "mode",
        "source",
        "side",
        "ticker",
        "figi",
        "lots",
        "estimated_price",
        "limit_price",
        "take_profit_price",
        "stop_loss_price",
        "planned_exit_date",
        "max_loss_rub",
        "expected_order_value",
        "reason_code",
        "status",
        "model_version",
        "strategy_version",
        "planner_version",
        "linked_signal_id",
        "linked_event_id",
        "broker_order_id",
        "submitted_at",
        "sandbox_response",
        "executor_version",
        "last_error",
    ]

    result = intents.copy()

    for col in required_columns:
        if col not in result.columns:
            result[col] = ""

    return result


def load_planned_sandbox_buy_intents(settings) -> tuple[pd.DataFrame, Path]:
    intents_path = settings.data_path / "orders" / "order_intents.csv"
    intents = read_csv_or_empty(intents_path)

    if intents.empty:
        return normalize_order_intents_columns(intents), intents_path

    intents = normalize_order_intents_columns(intents)

    mask = (
        intents["mode"].astype(str).str.lower().eq("sandbox")
        & intents["side"].astype(str).str.upper().eq("BUY")
        & intents["status"].astype(str).str.lower().eq("planned")
    )

    return intents.loc[mask].copy(), intents_path


def parse_positive_int(value, field_name: str) -> int:
    if pd.isna(value):
        raise ValueError(f"{field_name} is missing")

    lots = int(float(value))

    if lots < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")

    return lots


def parse_positive_decimal(value, field_name: str) -> Decimal:
    if pd.isna(value):
        raise ValueError(f"{field_name} is missing")

    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name} is invalid: {value}") from exc

    if result <= 0:
        raise ValueError(f"{field_name} must be > 0, got {value}")

    return result


def update_intent_row(
    all_intents: pd.DataFrame,
    intent_id: str,
    updates: dict,
) -> pd.DataFrame:
    mask = all_intents["intent_id"].astype(str).eq(str(intent_id))

    if not mask.any():
        return all_intents

    for key, value in updates.items():
        if key not in all_intents.columns:
            all_intents[key] = ""
        all_intents.loc[mask, key] = value

    return all_intents


def run_order_intent_mode(place_orders: bool = False) -> None:
    load_dotenv()

    token = os.getenv("TINVEST_TOKEN")
    account_id = os.getenv("TINVEST_SANDBOX_ACCOUNT_ID")

    if not token:
        raise RuntimeError("TINVEST_TOKEN is not set in .env")

    if not account_id:
        raise RuntimeError("TINVEST_SANDBOX_ACCOUNT_ID is not set in .env")

    settings = get_settings()
    instruments = load_instruments(settings)

    intents_path = settings.data_path / "orders" / "order_intents.csv"
    all_intents = normalize_order_intents_columns(read_csv_or_empty(intents_path))
    planned_intents, _ = load_planned_sandbox_buy_intents(settings)

    sandbox_dir = settings.data_path / "sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    log_path = sandbox_dir / "sandbox_orders_log.csv"

    if planned_intents.empty:
        print("No planned sandbox BUY OrderIntent rows found.")
        print(f"Checked: {intents_path}")
        print("Run first: python scripts/plan_daily_order_intents.py --mode sandbox")
        return

    orders_log: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()

    with Client(token) as client:
        active_order_tickers = get_active_order_tickers(
            client=client,
            account_id=account_id,
            instruments=instruments,
        )
        position_tickers = get_position_tickers(
            client=client,
            account_id=account_id,
            instruments=instruments,
        )

        print("\nCurrent sandbox state:")
        print(f"Active order tickers: {sorted(active_order_tickers)}")
        print(f"Position tickers: {sorted(position_tickers)}")

        for _, intent in planned_intents.iterrows():
            intent_id = str(intent["intent_id"])
            ticker = str(intent["ticker"])
            figi = str(intent["figi"]).strip()

            if not figi or figi.lower() == "nan":
                figi = get_figi_for_ticker(instruments, ticker)

            order_id = str(uuid.uuid4())

            order_record = {
                "run_at": now,
                "executor_version": EXECUTOR_VERSION,
                "intent_id": intent_id,
                "ticker": ticker,
                "figi": figi,
                "side": "BUY",
                "lots": intent.get("lots", ""),
                "limit_price": intent.get("limit_price", ""),
                "estimated_price": intent.get("estimated_price", ""),
                "take_profit_price": intent.get("take_profit_price", ""),
                "stop_loss_price": intent.get("stop_loss_price", ""),
                "planned_exit_date": intent.get("planned_exit_date", ""),
                "place_orders": place_orders,
                "broker_order_id": "",
                "client_order_id": order_id,
                "status_before": intent.get("status", ""),
                "status_after": intent.get("status", ""),
                "skipped": False,
                "skipped_reason": "",
                "sandbox_response": "",
                "error": "",
            }

            try:
                quantity_lots = parse_positive_int(intent["lots"], "lots")
                limit_price = parse_positive_decimal(intent["limit_price"], "limit_price")
            except Exception as exc:
                message = str(exc)
                order_record["skipped"] = True
                order_record["skipped_reason"] = "ERROR_CONFIG_INVALID"
                order_record["status_after"] = "rejected"
                order_record["error"] = message
                order_record["sandbox_response"] = "rejected_before_submit"
                orders_log.append(order_record)

                all_intents = update_intent_row(
                    all_intents,
                    intent_id,
                    {
                        "status": "rejected",
                        "reason_code": "ERROR_CONFIG_INVALID",
                        "last_error": message,
                        "executor_version": EXECUTOR_VERSION,
                    },
                )

                print("\nRejected sandbox intent before submit:")
                print(order_record)
                continue

            if ticker in active_order_tickers:
                message = "active order already exists at execution time"
                order_record["skipped"] = True
                order_record["skipped_reason"] = "SKIP_ACTIVE_ORDER"
                order_record["status_after"] = "skipped"
                order_record["sandbox_response"] = "skipped_before_submit"
                order_record["error"] = message
                orders_log.append(order_record)

                all_intents = update_intent_row(
                    all_intents,
                    intent_id,
                    {
                        "status": "skipped",
                        "reason_code": "SKIP_ACTIVE_ORDER",
                        "last_error": message,
                        "executor_version": EXECUTOR_VERSION,
                    },
                )

                print("\nSkipped sandbox intent:")
                print(order_record)
                continue

            if ticker in position_tickers:
                message = "position already exists at execution time"
                order_record["skipped"] = True
                order_record["skipped_reason"] = "SKIP_ALREADY_POSITION"
                order_record["status_after"] = "skipped"
                order_record["sandbox_response"] = "skipped_before_submit"
                order_record["error"] = message
                orders_log.append(order_record)

                all_intents = update_intent_row(
                    all_intents,
                    intent_id,
                    {
                        "status": "skipped",
                        "reason_code": "SKIP_ALREADY_POSITION",
                        "last_error": message,
                        "executor_version": EXECUTOR_VERSION,
                    },
                )

                print("\nSkipped sandbox intent:")
                print(order_record)
                continue

            print("\nApproved sandbox OrderIntent:")
            print(order_record)

            if place_orders:
                try:
                    response = client.sandbox.post_sandbox_order(
                        figi=figi,
                        quantity=quantity_lots,
                        price=decimal_to_quotation(limit_price),
                        direction=OrderDirection.ORDER_DIRECTION_BUY,
                        account_id=account_id,
                        order_type=OrderType.ORDER_TYPE_LIMIT,
                        order_id=order_id,
                    )

                    broker_order_id = str(getattr(response, "order_id", order_id))

                    order_record["broker_order_id"] = broker_order_id
                    order_record["status_after"] = "submitted"
                    order_record["sandbox_response"] = str(response)

                    all_intents = update_intent_row(
                        all_intents,
                        intent_id,
                        {
                            "status": "submitted",
                            "broker_order_id": broker_order_id,
                            "submitted_at": now,
                            "sandbox_response": str(response),
                            "executor_version": EXECUTOR_VERSION,
                            "last_error": "",
                        },
                    )

                    active_order_tickers.add(ticker)

                    print("Order sent:")
                    print(response)

                except Exception as exc:
                    message = str(exc)
                    order_record["status_after"] = "rejected"
                    order_record["sandbox_response"] = "submit_error"
                    order_record["error"] = message

                    all_intents = update_intent_row(
                        all_intents,
                        intent_id,
                        {
                            "status": "rejected",
                            "reason_code": "ERROR_TINVEST_API",
                            "last_error": message,
                            "executor_version": EXECUTOR_VERSION,
                        },
                    )

                    print("\nSandbox submit error:")
                    print(order_record)
            else:
                order_record["status_after"] = "planned"
                order_record["sandbox_response"] = "dry_run"
                print("Dry-run only. Order was not sent.")

            orders_log.append(order_record)

        portfolio_after = client.sandbox.get_sandbox_portfolio(account_id=account_id)

        print("\nSandbox portfolio after:")
        print(portfolio_after)

    if place_orders:
        intents_path.parent.mkdir(parents=True, exist_ok=True)
        all_intents.to_csv(intents_path, index=False)
        print(f"\nUpdated intents: {intents_path}")
    else:
        print("\nDry-run mode: order_intents.csv was not modified.")

    append_csv(log_path, orders_log)
    print(f"Saved sandbox order log: {log_path}")


def run_legacy_signals_mode(place_orders: bool = False):
    """Legacy mode kept as an explicit fallback.

    This path preserves the old behavior:
    - read latest signals directly
    - fixed 1 lot
    - limit_price = close * 0.995
    - skip active orders / existing positions
    """

    load_dotenv()

    token = os.getenv("TINVEST_TOKEN")
    account_id = os.getenv("TINVEST_SANDBOX_ACCOUNT_ID")

    if not token:
        raise RuntimeError("TINVEST_TOKEN is not set in .env")

    if not account_id:
        raise RuntimeError("TINVEST_SANDBOX_ACCOUNT_ID is not set in .env")

    settings = get_settings()

    strategy_cfg = load_yaml(Path("config/strategy_candidate_v1.yaml"))["strategy"]

    strategy_name = strategy_cfg["name"]
    threshold = float(strategy_cfg["threshold"])
    max_positions = int(strategy_cfg["max_positions"])
    excluded_tickers = set(strategy_cfg.get("excluded_tickers", []))

    instruments = load_instruments(settings)

    signals = get_latest_signals(
        settings=settings,
        strategy_name=strategy_name,
        threshold=threshold,
        excluded_tickers=excluded_tickers,
    )

    out_dir = settings.data_path / "sandbox"
    out_dir.mkdir(parents=True, exist_ok=True)

    if signals.empty:
        print("No sandbox orders: no signals above threshold.")
        return

    orders_log = []

    with Client(token) as client:
        active_order_tickers = get_active_order_tickers(
            client=client,
            account_id=account_id,
            instruments=instruments,
        )
        position_tickers = get_position_tickers(
            client=client,
            account_id=account_id,
            instruments=instruments,
        )

        print("\nCurrent sandbox state:")
        print(f"Active order tickers: {sorted(active_order_tickers)}")
        print(f"Position tickers: {sorted(position_tickers)}")

        signals = signals.head(max_positions)

        for _, sig in signals.iterrows():
            ticker = str(sig["ticker"])
            close = Decimal(str(sig["close"]))
            proba_1 = float(sig["proba_1"])

            figi = get_figi_for_ticker(instruments, ticker)
            lot = get_lot_for_ticker(instruments, ticker)

            quantity_lots = 1
            limit_price = close * Decimal("0.995")
            order_id = str(uuid.uuid4())

            order_record = {
                "ticker": ticker,
                "figi": figi,
                "lot": lot,
                "quantity_lots": quantity_lots,
                "last_close": float(close),
                "limit_price": float(limit_price),
                "proba_1": proba_1,
                "place_orders": place_orders,
                "order_id": order_id,
                "skipped": False,
                "skipped_reason": "",
                "sandbox_response": "",
                "executor_version": "legacy_signals_mode",
            }

            if ticker in active_order_tickers:
                order_record["skipped"] = True
                order_record["skipped_reason"] = "active_order_exists"
                order_record["sandbox_response"] = "skipped"
                orders_log.append(order_record)

                print("\nSkipped sandbox order:")
                print(order_record)
                continue

            if ticker in position_tickers:
                order_record["skipped"] = True
                order_record["skipped_reason"] = "position_exists"
                order_record["sandbox_response"] = "skipped"
                orders_log.append(order_record)

                print("\nSkipped sandbox order:")
                print(order_record)
                continue

            print("\nPlanned sandbox order:")
            print(order_record)

            if place_orders:
                response = client.sandbox.post_sandbox_order(
                    figi=figi,
                    quantity=quantity_lots,
                    price=decimal_to_quotation(limit_price),
                    direction=OrderDirection.ORDER_DIRECTION_BUY,
                    account_id=account_id,
                    order_type=OrderType.ORDER_TYPE_LIMIT,
                    order_id=order_id,
                )

                order_record["sandbox_response"] = str(response)
                print("Order sent:")
                print(response)

                active_order_tickers.add(ticker)
            else:
                order_record["sandbox_response"] = "dry_run"

            orders_log.append(order_record)

        portfolio_after = client.sandbox.get_sandbox_portfolio(account_id=account_id)

        print("\nSandbox portfolio after:")
        print(portfolio_after)

    out_path = out_dir / "sandbox_orders_log.csv"
    append_csv(out_path, orders_log)

    print(f"\nSaved: {out_path}")


def main(place_orders: bool = False, legacy_signals: bool = False):
    if legacy_signals:
        print("Running legacy signals mode. Prefer OrderIntent mode for normal operation.")
        run_legacy_signals_mode(place_orders=place_orders)
        return

    run_order_intent_mode(place_orders=place_orders)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--place-orders",
        action="store_true",
        help="Actually send sandbox orders. Without this flag, dry-run only.",
    )
    parser.add_argument(
        "--legacy-signals",
        action="store_true",
        help="Fallback to the old behavior: read signals directly and submit fixed 1-lot orders.",
    )

    args = parser.parse_args()

    main(place_orders=args.place_orders, legacy_signals=args.legacy_signals)
