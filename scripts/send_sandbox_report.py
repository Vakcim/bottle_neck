from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
from dotenv import load_dotenv

from src.settings import get_settings


def fmt_money(value: float) -> str:
    return f"{value:,.2f}".replace(",", " ")


def send_telegram_message(text: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not set")

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


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_str(value, default: str = "") -> str:
    try:
        if pd.isna(value):
            return default
        return str(value)
    except Exception:
        return default


def main():
    load_dotenv()

    settings = get_settings()
    sandbox_dir = settings.data_path / "sandbox"

    reconcile_summary_path = sandbox_dir / "sandbox_reconcile_summary.csv"
    active_orders_path = sandbox_dir / "sandbox_active_orders.csv"
    positions_path = sandbox_dir / "sandbox_positions.csv"
    tracker_path = sandbox_dir / "sandbox_positions_tracker.csv"
    exit_orders_path = sandbox_dir / "sandbox_exit_orders_log.csv"
    orders_log_path = sandbox_dir / "sandbox_orders_log.csv"

    reconcile = read_csv_or_empty(reconcile_summary_path)
    active_orders = read_csv_or_empty(active_orders_path)
    positions = read_csv_or_empty(positions_path)
    tracker = read_csv_or_empty(tracker_path)
    exit_orders = read_csv_or_empty(exit_orders_path)
    orders_log = read_csv_or_empty(orders_log_path)

    lines: list[str] = []

    lines.append("🧾 <b>T-Invest Sandbox Lifecycle</b>")

    if not reconcile.empty:
        row = reconcile.iloc[-1].to_dict()

        total_portfolio = safe_float(row.get("total_amount_portfolio"))
        cash = safe_float(row.get("total_amount_currencies"))
        shares = safe_float(row.get("total_amount_shares"))
        expected_yield = safe_float(row.get("expected_yield"))
        active_orders_count = int(safe_float(row.get("active_orders")))
        positions_count = int(safe_float(row.get("positions")))
        cancel_candidates = int(safe_float(row.get("cancel_candidates")))
        cancelled = int(safe_float(row.get("cancelled")))

        lines.append(f"Run at: <b>{safe_str(row.get('run_at'), 'unknown')}</b>")
        lines.append("")
        lines.append(f"Portfolio: <b>{fmt_money(total_portfolio)} ₽</b>")
        lines.append(f"Cash: <b>{fmt_money(cash)} ₽</b>")
        lines.append(f"Shares: <b>{fmt_money(shares)} ₽</b>")
        lines.append(f"Expected yield: <b>{fmt_money(expected_yield)} ₽</b>")
        lines.append("")
        lines.append(f"Active orders: <b>{active_orders_count}</b>")
        lines.append(f"Positions: <b>{positions_count}</b>")
        lines.append(f"Cancel candidates: <b>{cancel_candidates}</b>")
        lines.append(f"Cancelled: <b>{cancelled}</b>")
    else:
        lines.append("⚠️ Reconcile summary not found.")

    # Активные заявки
    lines.append("")
    if active_orders.empty:
        lines.append("📭 <b>Active sandbox orders:</b> none")
    else:
        lines.append("📌 <b>Active sandbox orders:</b>")
        for _, order in active_orders.head(10).iterrows():
            figi = safe_str(order.get("figi"), "?")
            status = safe_str(order.get("status"), "?")
            lots_requested = safe_str(order.get("lots_requested"), "?")
            lots_executed = safe_str(order.get("lots_executed"), "?")
            price = safe_str(order.get("initial_order_price"), "?")

            age_text = ""
            age = order.get("age_hours", "")
            try:
                age_text = f", age={float(age):.1f}h"
            except Exception:
                pass

            lines.append(
                f"{figi}: {status}, lots {lots_executed}/{lots_requested}, "
                f"price={price}{age_text}"
            )

    # Позиции в sandbox-портфеле
    lines.append("")
    share_positions = positions.copy()
    if not share_positions.empty and "instrument_type" in share_positions.columns:
        share_positions = share_positions[share_positions["instrument_type"] == "share"]

    if share_positions.empty:
        lines.append("💼 <b>Sandbox share positions:</b> none")
    else:
        lines.append("💼 <b>Sandbox share positions:</b>")
        for _, pos in share_positions.head(10).iterrows():
            figi = safe_str(pos.get("figi"), "?")
            qty_lots = safe_str(pos.get("quantity_lots"), "?")
            avg_price = safe_str(pos.get("average_position_price"), "?")
            current_price = safe_str(pos.get("current_price"), "?")
            expected = safe_str(pos.get("expected_yield"), "?")

            lines.append(
                f"{figi}: lots={qty_lots}, avg={avg_price}, "
                f"current={current_price}, PnL={expected}"
            )

    # Локальный tracker с TP/SL
    lines.append("")
    if tracker.empty:
        lines.append("🗂 <b>Tracked positions:</b> none")
    else:
        open_tracker = tracker[tracker.get("status", "") == "open"].copy()
        if open_tracker.empty:
            lines.append("🗂 <b>Tracked open positions:</b> none")
        else:
            lines.append("🗂 <b>Tracked open positions:</b>")
            for _, row in open_tracker.head(10).iterrows():
                ticker = safe_str(row.get("ticker"), "?")
                qty_lots = safe_str(row.get("quantity_lots"), "?")
                avg_price = safe_float(row.get("average_position_price"))
                current_price = safe_float(row.get("current_price"))
                tp = safe_float(row.get("take_profit_price"))
                sl = safe_float(row.get("stop_loss_price"))
                planned_exit = safe_str(row.get("planned_exit_date"), "?")
                expected = safe_float(row.get("expected_yield"))

                lines.append(
                    f"{ticker}: lots={qty_lots}, avg={avg_price:.2f}, "
                    f"cur={current_price:.2f}, TP={tp:.2f}, SL={sl:.2f}, "
                    f"exit={planned_exit}, PnL={expected:.2f}"
                )

    # Последние buy-заявки
    lines.append("")
    if orders_log.empty:
        lines.append("🟢 <b>Last sandbox buy logs:</b> none")
    else:
        lines.append("🟢 <b>Last sandbox buy logs:</b>")
        recent = orders_log.tail(5)
        for _, row in recent.iterrows():
            ticker = safe_str(row.get("ticker"), "?")
            skipped = safe_str(row.get("skipped"), "False")
            reason = safe_str(row.get("skipped_reason"), "")
            place_orders = safe_str(row.get("place_orders"), "")
            proba = safe_float(row.get("proba_1"))
            price = safe_float(row.get("limit_price"))

            extra = f" reason={reason}" if reason else ""

            lines.append(
                f"{ticker}: proba={proba:.3f}, buy_limit={price:.2f}, "
                f"place={place_orders}, skipped={skipped}{extra}"
            )

    # Последние exit-заявки с exit_reason
    lines.append("")
    if exit_orders.empty:
        lines.append("🔴 <b>Last sandbox exit logs:</b> none")
    else:
        lines.append("🔴 <b>Last sandbox exit logs:</b>")
        recent = exit_orders.tail(5)
        for _, row in recent.iterrows():
            ticker = safe_str(row.get("ticker"), "?")
            exit_reason = safe_str(row.get("exit_reason"), "?")
            skipped = safe_str(row.get("skipped"), "False")
            reason = safe_str(row.get("skipped_reason"), "")
            place_orders = safe_str(row.get("place_orders"), "")
            latest_close = safe_float(row.get("latest_close"))
            limit_price = safe_float(row.get("limit_price"))
            tp = safe_float(row.get("take_profit_price"))
            sl = safe_float(row.get("stop_loss_price"))
            planned_exit = safe_str(row.get("planned_exit_date"), "")

            extra = f" reason={reason}" if reason else ""

            lines.append(
                f"{ticker}: {exit_reason}, close={latest_close:.2f}, "
                f"sell_limit={limit_price:.2f}, TP={tp:.2f}, SL={sl:.2f}, "
                f"time_exit={planned_exit}, place={place_orders}, skipped={skipped}{extra}"
            )

    text = "\n".join(lines)

    send_telegram_message(text)
    print("Sandbox Telegram report sent.")


if __name__ == "__main__":
    main()
