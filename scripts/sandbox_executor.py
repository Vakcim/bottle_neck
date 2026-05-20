from __future__ import annotations

import os
import sys
import uuid
from decimal import Decimal
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
from dotenv import load_dotenv
from tinkoff.invest import Client, OrderDirection, OrderType
from tinkoff.invest.utils import decimal_to_quotation

from src.settings import get_settings


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def main(place_orders: bool = False):
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

    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    result = pd.concat([existing, pd.DataFrame(orders_log)], ignore_index=True)
    result.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--place-orders",
        action="store_true",
        help="Actually send sandbox orders. Without this flag, dry-run only.",
    )

    args = parser.parse_args()

    main(place_orders=args.place_orders)
