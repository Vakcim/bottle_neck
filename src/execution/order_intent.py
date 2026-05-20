from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from .reason_codes import ReasonCode


class OrderMode(StrEnum):
    PAPER = "paper"
    SANDBOX = "sandbox"
    LIVE = "live"


class OrderSource(StrEnum):
    DAILY_ALPHA = "daily_alpha"
    NEWS_EVENT = "news_event"
    EXIT = "exit"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderIntentStatus(StrEnum):
    PLANNED = "planned"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SKIPPED = "skipped"


@dataclass(slots=True)
class OrderIntent:
    mode: OrderMode
    source: OrderSource
    side: OrderSide
    ticker: str
    figi: str
    lots: int
    estimated_price: float
    limit_price: float
    reason_code: ReasonCode

    intent_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    take_profit_price: float | None = None
    stop_loss_price: float | None = None
    planned_exit_date: str | None = None

    max_loss_rub: float | None = None
    expected_order_value: float | None = None

    status: OrderIntentStatus = OrderIntentStatus.PLANNED

    model_version: str | None = None
    strategy_version: str | None = None
    planner_version: str | None = None

    linked_signal_id: str | None = None
    linked_event_id: str | None = None

    broker_order_id: str | None = None
    broker_response: str | None = None

    skipped_reason: ReasonCode | None = None
    error_message: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_skipped(self, reason: ReasonCode, message: str | None = None) -> None:
        self.status = OrderIntentStatus.SKIPPED
        self.skipped_reason = reason
        self.reason_code = reason
        self.error_message = message

    def mark_submitted(self, broker_order_id: str | None = None, broker_response: str | None = None) -> None:
        self.status = OrderIntentStatus.SUBMITTED
        self.broker_order_id = broker_order_id
        self.broker_response = broker_response

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        for key, value in list(data.items()):
            if isinstance(value, StrEnum):
                data[key] = value.value
            elif isinstance(value, datetime):
                data[key] = value.isoformat()

        return data

    @classmethod
    def skipped(
        cls,
        *,
        mode: OrderMode,
        source: OrderSource,
        side: OrderSide,
        ticker: str,
        reason_code: ReasonCode,
        message: str | None = None,
        figi: str = "",
        estimated_price: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> "OrderIntent":
        intent = cls(
            mode=mode,
            source=source,
            side=side,
            ticker=ticker,
            figi=figi,
            lots=0,
            estimated_price=estimated_price,
            limit_price=0.0,
            reason_code=reason_code,
            status=OrderIntentStatus.SKIPPED,
            skipped_reason=reason_code,
            error_message=message,
            metadata=metadata or {},
        )
        return intent
