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


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_latest_signal(signals_dir: Path, strategy_name: str) -> pd.DataFrame:
    files = sorted(signals_dir.glob(f"signals_*_{strategy_name}.csv"))
    if not files:
        return pd.DataFrame()

    return pd.read_csv(files[-1])


def send_telegram_message(token: str, chat_id: str, text: str) -> None:
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


def main():
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env"
        )

    settings = get_settings()

    strategy_name = "candidate_v1"

    paper_report_path = settings.data_path / "paper" / "paper_report.csv"
    positions_path = settings.data_path / "paper" / "positions.csv"
    signals_dir = settings.data_path / "signals"

    if not paper_report_path.exists():
        raise FileNotFoundError(paper_report_path)

    report = pd.read_csv(paper_report_path).iloc[-1].to_dict()

    latest_signal = load_latest_signal(signals_dir, strategy_name)

    latest_live_date = str(report.get("latest_live_date", "unknown"))
    equity = float(report.get("equity", 0.0))
    total_return = float(report.get("total_return", 0.0))
    open_positions_count = int(report.get("open_positions", 0))
    opened_today = int(report.get("opened_today", 0))
    closed_today = int(report.get("closed_today", 0))

    lines = []

    lines.append("🤖 <b>T-Invest Paper Bot</b>")
    lines.append(f"Стратегия: <b>{strategy_name}</b>")
    lines.append(f"Дата данных: <b>{latest_live_date}</b>")
    lines.append("")
    lines.append(f"Equity: <b>{fmt_money(equity)} ₽</b>")
    lines.append(f"Total return: <b>{fmt_pct(total_return)}</b>")
    lines.append(f"Открытых позиций: <b>{open_positions_count}</b>")
    lines.append(f"Открыто сегодня: <b>{opened_today}</b>")
    lines.append(f"Закрыто сегодня: <b>{closed_today}</b>")

    if not latest_signal.empty and "proba_1" in latest_signal.columns:
        latest_signal = latest_signal.sort_values("proba_1", ascending=False)

        signals = latest_signal[latest_signal["proba_1"] >= 0.50].copy()

        lines.append("")
        if signals.empty:
            lines.append("📭 <b>Сигналов сегодня нет</b>")
        else:
            lines.append("📈 <b>Сигналы:</b>")
            for _, row in signals.head(3).iterrows():
                lines.append(
                    f"{row['ticker']}: proba_1={float(row['proba_1']):.3f}, "
                    f"close={float(row['close']):.2f}"
                )

        lines.append("")
        lines.append("🔎 <b>Top probabilities:</b>")
        for _, row in latest_signal.head(5).iterrows():
            lines.append(
                f"{row['ticker']}: {float(row['proba_1']):.3f}"
            )

    if positions_path.exists():
        positions = pd.read_csv(positions_path)
        if not positions.empty and "status" in positions.columns:
            open_positions = positions[positions["status"] == "open"].copy()

            if not open_positions.empty:
                lines.append("")
                lines.append("📌 <b>Open positions:</b>")
                for _, pos in open_positions.iterrows():
                    lines.append(
                        f"{pos['ticker']}: entry={float(pos['entry_price']):.2f}, "
                        f"capital={fmt_money(float(pos['capital']))}"
                    )

    text = "\n".join(lines)

    send_telegram_message(token, chat_id, text)

    print("Telegram report sent.")


if __name__ == "__main__":
    main()
