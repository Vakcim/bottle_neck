"""OrderPlanner: converts signals/events into OrderIntent or skipped decisions.

V1 is intentionally rule-based. It does not submit orders and it does not call
T-Invest. This makes it safe to run before any sandbox executor integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from src.execution.order_intent import (
    ExecutionMode,
    IntentSource,
    OrderIntent,
    OrderSide,
    SkippedDecision,
)
from src.execution.reason_codes import ReasonCode
from src.risk.sizing import SizingInput, calculate_risk_based_lots, normalize_pct


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _ticker_set(rows: Iterable[dict[str, Any]], status_values: set[str] | None = None) -> set[str]:
    out: set[str] = set()
    for row in rows or []:
        status = str(row.get("status", "")).lower()
        if status_values is not None and status not in status_values:
            continue
        ticker = row.get("ticker")
        if ticker:
            out.add(str(ticker).upper().strip())
    return out


def add_weekdays(start: datetime, days: int) -> datetime:
    cur = start
    remaining = max(int(days), 0)
    while remaining > 0:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            remaining -= 1
    return cur


@dataclass(slots=True)
class PlannerResult:
    intent: OrderIntent | None
    skipped: SkippedDecision | None

    @property
    def approved(self) -> bool:
        return self.intent is not None


class OrderPlanner:
    planner_version = "order_planner_v1"

    def __init__(
        self,
        strategy_config: dict[str, Any],
        risk_config: dict[str, Any] | None = None,
        execution_config: dict[str, Any] | None = None,
        mode: str = ExecutionMode.PAPER.value,
    ) -> None:
        self.strategy_config = strategy_config or {}
        self.risk_config = risk_config or {}
        self.execution_config = execution_config or {}
        self.mode = mode
        if self.mode == ExecutionMode.LIVE.value:
            raise ValueError("live mode is disabled for OrderPlanner v1")

    def _skip(
        self,
        *,
        ticker: str,
        reason_code: ReasonCode,
        message: str,
        signal: dict[str, Any] | None = None,
    ) -> PlannerResult:
        signal = signal or {}
        return PlannerResult(
            intent=None,
            skipped=SkippedDecision(
                ticker=ticker,
                reason_code=reason_code.value,
                message=message,
                mode=self.mode,
                source=IntentSource.DAILY_ALPHA.value,
                linked_signal_id=signal.get("signal_id"),
                proba_1=_as_float(signal.get("proba_1"), None),
                estimated_price=_as_float(_get(signal, "close", "last_close", "price"), None),
                strategy_version=str(self.strategy_config.get("name", "candidate_v1")),
                planner_version=self.planner_version,
            ),
        )

    def plan_daily_buy(
        self,
        *,
        signal: dict[str, Any],
        portfolio_state: dict[str, Any],
        active_orders: Iterable[dict[str, Any]] | None,
        positions: Iterable[dict[str, Any]] | None,
        instrument_metadata: dict[str, Any] | None = None,
    ) -> PlannerResult:
        instrument_metadata = instrument_metadata or {}
        ticker = str(signal.get("ticker", "")).upper().strip()
        if not ticker:
            return self._skip(
                ticker="UNKNOWN",
                reason_code=ReasonCode.SKIP_STALE_DATA,
                message="signal has no ticker",
                signal=signal,
            )

        threshold = _as_float(self.strategy_config.get("threshold"), 0.50)
        proba_1 = _as_float(signal.get("proba_1"), 0.0)
        if proba_1 < threshold:
            return self._skip(
                ticker=ticker,
                reason_code=ReasonCode.SKIP_LOW_PROBA,
                message=f"proba_1={proba_1:.4f} < threshold={threshold:.4f}",
                signal=signal,
            )

        excluded = {str(x).upper().strip() for x in self.strategy_config.get("excluded_tickers", [])}
        if ticker in excluded:
            return self._skip(
                ticker=ticker,
                reason_code=ReasonCode.SKIP_EXCLUDED_TICKER,
                message="ticker is excluded by strategy config",
                signal=signal,
            )

        active_tickers = _ticker_set(
            active_orders or [],
            status_values={"planned", "submitted", "new", "partiallyfill", "active"},
        )
        if ticker in active_tickers:
            return self._skip(
                ticker=ticker,
                reason_code=ReasonCode.SKIP_ACTIVE_ORDER,
                message="active/planned order already exists for ticker",
                signal=signal,
            )

        open_positions = list(positions or [])
        open_position_tickers = _ticker_set(open_positions, status_values={"open", "closing"})
        if ticker in open_position_tickers:
            return self._skip(
                ticker=ticker,
                reason_code=ReasonCode.SKIP_ALREADY_POSITION,
                message="open/closing position already exists for ticker",
                signal=signal,
            )

        max_positions = _as_int(
            self.risk_config.get("max_positions", self.strategy_config.get("max_positions")),
            _as_int(self.strategy_config.get("max_positions"), 3),
        )
        if len(open_position_tickers) >= max_positions:
            return self._skip(
                ticker=ticker,
                reason_code=ReasonCode.SKIP_MAX_POSITIONS,
                message=f"open positions={len(open_position_tickers)} >= max_positions={max_positions}",
                signal=signal,
            )

        allow_short = bool(self.risk_config.get("allow_short", False))
        allow_leverage = bool(self.risk_config.get("allow_leverage", False))
        if allow_short or allow_leverage:
            return self._skip(
                ticker=ticker,
                reason_code=ReasonCode.SKIP_RISK_LIMIT,
                message="config attempts to allow short/leverage; disabled for this system",
                signal=signal,
            )

        estimated_price = _as_float(_get(signal, "close", "last_close", "price"), 0.0)
        if estimated_price <= 0:
            return self._skip(
                ticker=ticker,
                reason_code=ReasonCode.SKIP_STALE_PRICE,
                message="missing or non-positive close/price in signal",
                signal=signal,
            )

        buy_limit_offset_pct = normalize_pct(
            self.execution_config.get("buy_limit_offset_pct", 0.005), 0.005
        )
        take_profit_pct = normalize_pct(self.execution_config.get("take_profit_pct", 0.035), 0.035)
        stop_loss_pct = normalize_pct(self.execution_config.get("stop_loss_pct", 0.020), 0.020)

        limit_price = estimated_price * (1.0 - buy_limit_offset_pct)
        take_profit_price = limit_price * (1.0 + take_profit_pct)
        stop_loss_price = limit_price * (1.0 - stop_loss_pct)

        portfolio_value = _as_float(portfolio_state.get("portfolio_value"), 0.0)
        available_cash = _as_float(portfolio_state.get("available_cash"), 0.0)
        if portfolio_value <= 0 or available_cash <= 0:
            return self._skip(
                ticker=ticker,
                reason_code=ReasonCode.SKIP_INSUFFICIENT_CASH,
                message="portfolio_value or available_cash is not positive",
                signal=signal,
            )

        lot_size = _as_int(instrument_metadata.get("lot_size"), 1) or 1
        sizing = calculate_risk_based_lots(
            SizingInput(
                portfolio_value=portfolio_value,
                available_cash=available_cash,
                entry_price=limit_price,
                stop_loss_price=stop_loss_price,
                lot_size=lot_size,
                risk_per_trade_pct=self.risk_config.get(
                    "risk_per_trade_pct",
                    self.risk_config.get("max_trade_risk_pct", 0.005),
                ),
                max_position_value_pct=self.risk_config.get(
                    "max_position_value_pct",
                    self.risk_config.get("max_position_per_asset_pct", 0.20),
                ),
                cash_buffer_pct=self.risk_config.get("cash_buffer_pct", 0.05),
            )
        )
        if sizing.lots < 1:
            return self._skip(
                ticker=ticker,
                reason_code=ReasonCode.SKIP_SIZE_TOO_SMALL,
                message=f"risk sizing returned zero lots: {sizing.reason}",
                signal=signal,
            )

        signal_date_raw = signal.get("date")
        try:
            signal_dt = datetime.fromisoformat(str(signal_date_raw).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            signal_dt = datetime.now(timezone.utc)
        hold_days = _as_int(self.strategy_config.get("hold_days"), 7)
        planned_exit_date = add_weekdays(signal_dt, hold_days).date().isoformat()

        intent = OrderIntent(
            mode=self.mode,
            source=IntentSource.DAILY_ALPHA.value,
            side=OrderSide.BUY.value,
            ticker=ticker,
            figi=instrument_metadata.get("figi"),
            lots=sizing.lots,
            estimated_price=estimated_price,
            limit_price=limit_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            planned_exit_date=planned_exit_date,
            max_loss_rub=sizing.max_loss_rub,
            expected_order_value=sizing.expected_order_value,
            reason_code=ReasonCode.BUY_DAILY_SIGNAL.value,
            model_version=str(self.strategy_config.get("model_version", "random_forest_baseline")),
            strategy_version=str(self.strategy_config.get("name", "candidate_v1")),
            planner_version=self.planner_version,
            linked_signal_id=signal.get("signal_id"),
        )
        return PlannerResult(intent=intent, skipped=None)
