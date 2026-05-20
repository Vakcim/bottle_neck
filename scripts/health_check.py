from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
from dotenv import load_dotenv

from src.settings import get_settings


def send_telegram_message(text: str) -> None:
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram env vars are not set. Skipping alert.")
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


def check_file_exists(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"Missing file: {path}")


def check_latest_live_date(live_path: Path, errors: list[str], warnings: list[str]) -> pd.Timestamp | None:
    if not live_path.exists():
        errors.append(f"Missing live features: {live_path}")
        return None

    live = pd.read_parquet(live_path)

    if live.empty:
        errors.append("live_features_day.parquet is empty")
        return None

    live["date"] = pd.to_datetime(live["date"], utc=True)
    latest_date = live["date"].max()

    now_utc = pd.Timestamp.now(tz="UTC").floor("D")
    age_days = (now_utc - latest_date).days

    if age_days > 5:
        errors.append(f"Live data is stale: latest={latest_date}, age_days={age_days}")
    elif age_days > 2:
        warnings.append(f"Live data may be stale: latest={latest_date}, age_days={age_days}")

    return latest_date


def check_signals(signals_dir: Path, latest_live_date: pd.Timestamp | None, errors: list[str], warnings: list[str]) -> None:
    files = sorted(signals_dir.glob("signals_*_candidate_v1.csv"))

    if not files:
        errors.append(f"No signal files found in {signals_dir}")
        return

    latest_file = files[-1]

    try:
        df = pd.read_csv(latest_file)
    except Exception as exc:
        errors.append(f"Cannot read latest signal file {latest_file}: {exc}")
        return

    if df.empty:
        warnings.append(f"Latest signal file is empty: {latest_file}")
        return

    if "date" not in df.columns:
        errors.append(f"Latest signal file has no date column: {latest_file}")
        return

    df["date"] = pd.to_datetime(df["date"], utc=True)
    signal_date = df["date"].max()

    if latest_live_date is not None and signal_date != latest_live_date:
        warnings.append(
            f"Signal date differs from live date: signal={signal_date}, live={latest_live_date}"
        )


def check_paper_report(report_path: Path, errors: list[str], warnings: list[str]) -> None:
    if not report_path.exists():
        errors.append(f"Missing paper report: {report_path}")
        return

    try:
        report = pd.read_csv(report_path)
    except Exception as exc:
        errors.append(f"Cannot read paper report {report_path}: {exc}")
        return

    if report.empty:
        errors.append("paper_report.csv is empty")
        return

    row = report.iloc[-1].to_dict()

    equity = float(row.get("equity", 0))
    open_positions = int(row.get("open_positions", 0))

    if equity <= 0:
        errors.append(f"Invalid equity: {equity}")

    if open_positions < 0:
        errors.append(f"Invalid open_positions: {open_positions}")

    if equity < 90_000:
        warnings.append(f"Paper equity below 90k: {equity:.2f}")


def check_log(log_path: Path, warnings: list[str]) -> None:
    if not log_path.exists():
        warnings.append(f"Log file not found yet: {log_path}")
        return

    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        warnings.append(f"Cannot read log file: {exc}")
        return

    tail = text[-5000:]

    bad_markers = [
        "Traceback",
        "ERROR",
        "ModuleNotFoundError",
        "HTTPError",
        "Bad Request",
        "No module named",
    ]

    found = [marker for marker in bad_markers if marker in tail]

    if found:
        warnings.append(f"Suspicious markers in log tail: {', '.join(found)}")


def main():
    settings = get_settings()

    data_dir = settings.data_path

    live_path = data_dir / "live" / "live_features_day.parquet"
    signals_dir = data_dir / "signals"
    paper_dir = data_dir / "paper"
    report_path = paper_dir / "paper_report.csv"
    positions_path = paper_dir / "positions.csv"
    trades_path = paper_dir / "trades.csv"
    equity_path = paper_dir / "equity.csv"
    log_path = data_dir / "logs" / "daily_pipeline.log"

    errors: list[str] = []
    warnings: list[str] = []

    latest_live_date = check_latest_live_date(live_path, errors, warnings)

    check_signals(signals_dir, latest_live_date, errors, warnings)
    check_paper_report(report_path, errors, warnings)

    for path in [positions_path, trades_path, equity_path]:
        check_file_exists(path, errors)

    check_log(log_path, warnings)

    if errors:
        status = "🚨 <b>Bot health check: ERROR</b>"
    elif warnings:
        status = "⚠️ <b>Bot health check: WARNING</b>"
    else:
        status = "✅ <b>Bot health check: OK</b>"

    lines = [status]

    if latest_live_date is not None:
        lines.append(f"Latest live date: <b>{latest_live_date}</b>")

    if errors:
        lines.append("")
        lines.append("<b>Errors:</b>")
        for item in errors:
            lines.append(f"• {item}")

    if warnings:
        lines.append("")
        lines.append("<b>Warnings:</b>")
        for item in warnings:
            lines.append(f"• {item}")

    if not errors and not warnings:
        lines.append("All required files exist. No suspicious log markers found.")

    text = "\n".join(lines)

    print(text.replace("<b>", "").replace("</b>", ""))

    # Отправляем Telegram только если есть проблема.
    # Если хочешь ежедневный OK-отчёт тоже — поменяем условие.
    if errors or warnings:
        send_telegram_message(text)


if __name__ == "__main__":
    main()
