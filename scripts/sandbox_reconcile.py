from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
from dotenv import load_dotenv
from tinkoff.invest import Client

from src.settings import get_settings


def quotation_to_float(q) -> float:
    if q is None:
        return 0.0
    return float(q.units) + float(q.nano) / 1_000_000_000


def money_to_float(m) -> float:
    if m is None:
        return 0.0
    return float(m.units) + float(m.nano) / 1_000_000_000


def send_telegram_message(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram env vars are not set. Skipping sandbox alert.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=30,
    )

    response.raise_for_status()


def order_to_dict(order, now: datetime) -> dict:
    order_date = order.order_date
    age_hours = None

    if order_date is not None:
        age_hours = (now - order_date).total_seconds() / 3600

    return {
        "order_id": order.order_id,
        "figi": order.figi,
        "instrument_uid": order.instrument_uid,
        "direction": str(order.direction),
        "status": str(order.execution_report_status),
        "lots_requested": int(order.lots_requested),
        "lots_executed": int(order.lots_executed),
        "initial_order_price": money_to_float(order.initial_order_price),
        "executed_order_price": money_to_float(order.executed_order_price),
        "total_order_amount": money_to_float(order.total_order_amount),
        "initial_commission": money_to_float(order.initial_commission),
        "executed_commission": money_to_float(order.executed_commission),
        "currency": order.currency,
        "order_type": str(order.order_type),
        "order_date": order_date,
        "age_hours": age_hours,
        "order_request_id": order.order_request_id,
    }


def position_to_dict(pos) -> dict:
    return {
        "figi": pos.figi,
        "instrument_uid": pos.instrument_uid,
        "instrument_type": pos.instrument_type,
        "quantity": quotation_to_float(pos.quantity),
        "quantity_lots": quotation_to_float(pos.quantity_lots),
        "average_position_price": money_to_float(pos.average_position_price),
        "current_price": money_to_float(pos.current_price),
        "expected_yield": quotation_to_float(pos.expected_yield),
        "blocked": bool(pos.blocked),
    }


def main(
    cancel_older_than_hours: float = 20.0,
    do_cancel: bool = False,
    send_report: bool = False,
):
    load_dotenv()

    token = os.getenv("TINVEST_TOKEN")
    account_id = os.getenv("TINVEST_SANDBOX_ACCOUNT_ID")

    if not token:
        raise RuntimeError("TINVEST_TOKEN is not set in .env")
    if not account_id:
        raise RuntimeError("TINVEST_SANDBOX_ACCOUNT_ID is not set in .env")

    settings = get_settings()
    out_dir = settings.data_path / "sandbox"
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)

    orders_rows = []
    positions_rows = []
    cancelled_rows = []

    with Client(token) as client:
        orders = client.sandbox.get_sandbox_orders(account_id=account_id).orders
        portfolio = client.sandbox.get_sandbox_portfolio(account_id=account_id)

        for order in orders:
            row = order_to_dict(order, now)
            orders_rows.append(row)

            age_hours = row["age_hours"]

            should_cancel = (
                age_hours is not None
                and age_hours >= cancel_older_than_hours
                and int(row["lots_executed"]) == 0
            )

            if should_cancel:
                cancel_record = {
                    "order_id": row["order_id"],
                    "figi": row["figi"],
                    "age_hours": age_hours,
                    "cancelled": False,
                    "error": "",
                }

                if do_cancel:
                    try:
                        client.sandbox.cancel_sandbox_order(
                            account_id=account_id,
                            order_id=row["order_id"],
                        )
                        cancel_record["cancelled"] = True
                    except Exception as exc:
                        cancel_record["error"] = str(exc)

                cancelled_rows.append(cancel_record)

        for pos in portfolio.positions:
            positions_rows.append(position_to_dict(pos))

        portfolio_summary = {
            "run_at": now.isoformat(),
            "account_id": account_id,
            "total_amount_portfolio": money_to_float(portfolio.total_amount_portfolio),
            "total_amount_currencies": money_to_float(portfolio.total_amount_currencies),
            "total_amount_shares": money_to_float(portfolio.total_amount_shares),
            "expected_yield": quotation_to_float(portfolio.expected_yield),
            "active_orders": len(orders_rows),
            "positions": len(positions_rows),
            "cancel_candidates": len(cancelled_rows),
            "cancelled": sum(1 for x in cancelled_rows if x.get("cancelled")),
            "do_cancel": do_cancel,
            "cancel_older_than_hours": cancel_older_than_hours,
        }

    orders_df = pd.DataFrame(orders_rows)
    positions_df = pd.DataFrame(positions_rows)
    cancelled_df = pd.DataFrame(cancelled_rows)
    summary_df = pd.DataFrame([portfolio_summary])

    orders_path = out_dir / "sandbox_active_orders.csv"
    positions_path = out_dir / "sandbox_positions.csv"
    cancelled_path = out_dir / "sandbox_cancelled_orders.csv"
    summary_path = out_dir / "sandbox_reconcile_summary.csv"

    orders_df.to_csv(orders_path, index=False)
    positions_df.to_csv(positions_path, index=False)
    cancelled_df.to_csv(cancelled_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSandbox reconcile summary:")
    for k, v in portfolio_summary.items():
        print(f"{k}: {v}")

    print(f"\nSaved: {orders_path}")
    print(f"Saved: {positions_path}")
    print(f"Saved: {cancelled_path}")
    print(f"Saved: {summary_path}")

    if not orders_df.empty:
        print("\nActive orders:")
        print(
            orders_df[
                [
                    "order_id",
                    "figi",
                    "direction",
                    "status",
                    "lots_requested",
                    "lots_executed",
                    "initial_order_price",
                    "age_hours",
                ]
            ].to_string(index=False)
        )
    else:
        print("\nActive orders: none")

    if not positions_df.empty:
        print("\nPositions:")
        print(
            positions_df[
                [
                    "figi",
                    "instrument_type",
                    "quantity",
                    "quantity_lots",
                    "average_position_price",
                    "current_price",
                    "expected_yield",
                ]
            ].to_string(index=False)
        )
    else:
        print("\nPositions: none")

    if send_report:
        lines = []
        lines.append("🧾 <b>T-Invest Sandbox Reconcile</b>")
        lines.append(f"Portfolio: <b>{portfolio_summary['total_amount_portfolio']:.2f} ₽</b>")
        lines.append(f"Cash: <b>{portfolio_summary['total_amount_currencies']:.2f} ₽</b>")
        lines.append(f"Shares: <b>{portfolio_summary['total_amount_shares']:.2f} ₽</b>")
        lines.append(f"Active orders: <b>{portfolio_summary['active_orders']}</b>")
        lines.append(f"Positions: <b>{portfolio_summary['positions']}</b>")
        lines.append(f"Cancel candidates: <b>{portfolio_summary['cancel_candidates']}</b>")
        lines.append(f"Cancelled: <b>{portfolio_summary['cancelled']}</b>")
        lines.append(f"do_cancel: <b>{do_cancel}</b>")

        if cancelled_rows:
            lines.append("")
            lines.append("<b>Cancel candidates:</b>")
            for row in cancelled_rows[:5]:
                lines.append(
                    f"{row['figi']} age={row['age_hours']:.1f}h "
                    f"cancelled={row['cancelled']}"
                )

        send_telegram_message("\n".join(lines))
        print("\nTelegram sandbox report sent.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cancel-older-than-hours",
        type=float,
        default=20.0,
        help="Cancel unfilled sandbox orders older than this many hours.",
    )
    parser.add_argument(
        "--do-cancel",
        action="store_true",
        help="Actually cancel old orders. Without this flag, dry-run only.",
    )
    parser.add_argument(
        "--send-report",
        action="store_true",
        help="Send Telegram report.",
    )

    args = parser.parse_args()

    main(
        cancel_older_than_hours=args.cancel_older_than_hours,
        do_cancel=args.do_cancel,
        send_report=args.send_report,
    )
