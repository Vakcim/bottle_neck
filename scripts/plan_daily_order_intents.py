from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
from loguru import logger
from pandas.errors import EmptyDataError

from src.execution.order_intent import ExecutionMode
from src.execution.order_planner import OrderPlanner
from src.execution.order_storage import append_records_csv
from src.settings import get_settings


DEFAULT_INITIAL_EQUITY = 100_000.0
FINAL_ORDER_STATUSES = {"filled", "cancelled", "rejected", "expired"}
ACTIVE_ORDER_STATUSES = {"planned", "submitted", "new", "partiallyfill", "active"}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_latest_signal_file(signals_dir: Path, strategy_name: str) -> Path | None:
    files = sorted(signals_dir.glob(f"signals_*_{strategy_name}.csv"))
    if not files:
        return None
    return files[-1]


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def load_portfolio_state(data_path: Path) -> dict[str, float]:
    paper_dir = data_path / "paper"
    equity_path = paper_dir / "equity.csv"
    positions_path = paper_dir / "positions.csv"

    if equity_path.exists():
        equity = pd.read_csv(equity_path)
        if not equity.empty:
            row = equity.iloc[-1]
            portfolio_value = float(row.get("equity", DEFAULT_INITIAL_EQUITY))
            available_cash = float(row.get("cash", portfolio_value))
            return {"portfolio_value": portfolio_value, "available_cash": available_cash}

    # Fallback for a fresh repository before paper_portfolio_tracker has run.
    positions = read_csv_or_empty(positions_path)
    used_capital = 0.0
    if not positions.empty and "status" in positions.columns and "capital" in positions.columns:
        open_positions = positions[positions["status"].astype(str).str.lower() == "open"]
        used_capital = float(open_positions["capital"].fillna(0).sum())
    available_cash = max(DEFAULT_INITIAL_EQUITY - used_capital, 0.0)
    return {"portfolio_value": DEFAULT_INITIAL_EQUITY, "available_cash": available_cash}


def load_positions(data_path: Path) -> list[dict[str, Any]]:
    positions_path = data_path / "paper" / "positions.csv"
    positions = read_csv_or_empty(positions_path)
    if positions.empty:
        return []
    if "status" not in positions.columns:
        positions["status"] = "open"
    return positions.to_dict("records")


def load_active_orders(data_path: Path) -> list[dict[str, Any]]:
    intents_path = data_path / "orders" / "order_intents.csv"
    intents = read_csv_or_empty(intents_path)
    if intents.empty or "status" not in intents.columns:
        return []
    statuses = intents["status"].astype(str).str.lower()
    active = intents[statuses.isin(ACTIVE_ORDER_STATUSES)].copy()
    return active.to_dict("records")


def add_signal_ids(signals: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    signals = signals.copy()
    if "signal_id" not in signals.columns:
        date_part = signals["date"].astype(str).str.slice(0, 10) if "date" in signals.columns else "unknown"
        signals["signal_id"] = [
            f"{strategy_name}:{date}:{ticker}"
            for date, ticker in zip(date_part, signals["ticker"].astype(str).str.upper())
        ]
    return signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan daily OrderIntent records from candidate signals.")
    parser.add_argument("--mode", choices=[ExecutionMode.PAPER.value, ExecutionMode.SANDBOX.value], default=ExecutionMode.PAPER.value)
    parser.add_argument("--strategy-config", default="config/strategy_candidate_v1.yaml")
    parser.add_argument("--risk-config", default="config/risk.yaml")
    parser.add_argument("--output-dir", default=None, help="Defaults to <data_dir>/orders")
    args = parser.parse_args()

    settings = get_settings()
    data_path = settings.data_path

    cfg = load_yaml(Path(args.strategy_config))
    strategy_cfg = cfg.get("strategy", {})
    execution_cfg = cfg.get("execution", {})
    strategy_name = str(strategy_cfg.get("name", "candidate_v1"))

    risk_root = load_yaml(Path(args.risk_config))
    risk_cfg = risk_root.get("risk", risk_root)

    signals_dir = data_path / "signals"
    latest_signal_file = find_latest_signal_file(signals_dir, strategy_name)
    if latest_signal_file is None:
        print(f"No signal files found in {signals_dir} for strategy={strategy_name}")
        return

    signals = pd.read_csv(latest_signal_file)
    if signals.empty:
        print(f"Signal file is empty: {latest_signal_file}")
        return
    signals = add_signal_ids(signals, strategy_name)

    portfolio_state = load_portfolio_state(data_path)
    positions = load_positions(data_path)
    active_orders = load_active_orders(data_path)

    planner = OrderPlanner(
        strategy_config=strategy_cfg,
        risk_config=risk_cfg,
        execution_config=execution_cfg,
        mode=args.mode,
    )

    intents = []
    skipped = []
    for _, row in signals.iterrows():
        signal = row.to_dict()
        result = planner.plan_daily_buy(
            signal=signal,
            portfolio_state=portfolio_state,
            active_orders=active_orders,
            positions=positions,
            instrument_metadata={"lot_size": 1, "figi": signal.get("figi")},
        )
        if result.intent is not None:
            intents.append(result.intent)
            # Prevent multiple intents for the same ticker inside one run.
            active_orders.append(result.intent.to_dict())
        elif result.skipped is not None:
            skipped.append(result.skipped)

    output_dir = Path(args.output_dir) if args.output_dir else data_path / "orders"
    intents_path = output_dir / "order_intents.csv"
    skipped_path = output_dir / "skipped_decisions.csv"

    if intents:
        append_records_csv(intents_path, intents)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not intents_path.exists():
            pd.DataFrame().to_csv(intents_path, index=False)

    if skipped:
        append_records_csv(skipped_path, skipped)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not skipped_path.exists():
            pd.DataFrame().to_csv(skipped_path, index=False)

    print("\nDaily OrderIntent planning report")
    print(f"signal_file: {latest_signal_file}")
    print(f"mode: {args.mode}")
    print(f"portfolio_value: {portfolio_state['portfolio_value']:.2f}")
    print(f"available_cash: {portfolio_state['available_cash']:.2f}")
    print(f"signals_seen: {len(signals)}")
    print(f"planned_intents: {len(intents)}")
    print(f"skipped_decisions: {len(skipped)}")
    print(f"saved_intents: {intents_path}")
    print(f"saved_skipped: {skipped_path}")

    if intents:
        print("\nPlanned intents:")
        cols = ["ticker", "side", "lots", "limit_price", "take_profit_price", "stop_loss_price", "planned_exit_date", "reason_code"]
        print(pd.DataFrame([x.to_dict() for x in intents])[cols].to_string(index=False))

    if skipped:
        print("\nSkipped decisions:")
        skip_df = pd.DataFrame([x.to_dict() for x in skipped])
        cols = ["ticker", "reason_code", "message"]
        print(skip_df[cols].to_string(index=False))

    logger.info("Daily OrderIntent planning done")


if __name__ == "__main__":
    main()
