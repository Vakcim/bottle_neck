"""Order intent schema used before any paper/sandbox/live execution.

The executor must execute approved intents; it must not decide what to trade,
how many lots to trade, or where to place TP/SL.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


class ExecutionMode(StrEnum):
    PAPER = "paper"
    SANDBOX = "sandbox"
    LIVE = "live"


class IntentSource(StrEnum):
    DAILY_ALPHA = "daily_alpha"
    NEWS_EVENT = "news_event"
    EXIT = "exit"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class IntentStatus(StrEnum):
    PLANNED = "planned"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SKIPPED = "skipped"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class OrderIntent:
    mode: str
    source: str
    side: str
    ticker: str
    figi: str | None
    lots: int
    estimated_price: float
    limit_price: float
    take_profit_price: float | None
    stop_loss_price: float | None
    planned_exit_date: str | None
    max_loss_rub: float | None
    expected_order_value: float
    reason_code: str
    status: str = IntentStatus.PLANNED.value
    intent_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now_iso)
    model_version: str | None = None
    strategy_version: str | None = None
    planner_version: str = "order_planner_v1"
    linked_signal_id: str | None = None
    linked_event_id: str | None = None
    broker_order_id: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        self.ticker = str(self.ticker).upper().strip()
        self.side = str(self.side)
        self.mode = str(self.mode)
        self.source = str(self.source)
        self.status = str(self.status)
        self.reason_code = str(self.reason_code)
        self.lots = int(self.lots)
        if self.lots < 0:
            raise ValueError("lots must be non-negative")
        if self.mode == ExecutionMode.LIVE.value:
            raise ValueError("live OrderIntent creation is disabled for the current project phase")
        if self.side == OrderSide.SELL.value and self.lots <= 0:
            raise ValueError("SELL OrderIntent must have lots > 0")
        if self.side == OrderSide.BUY.value and self.lots <= 0:
            raise ValueError("BUY OrderIntent must have lots > 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrderIntent":
        return cls(**data)


@dataclass(slots=True)
class SkippedDecision:
    ticker: str
    reason_code: str
    message: str
    mode: str = ExecutionMode.PAPER.value
    source: str = IntentSource.DAILY_ALPHA.value
    created_at: str = field(default_factory=utc_now_iso)
    linked_signal_id: str | None = None
    linked_event_id: str | None = None
    proba_1: float | None = None
    estimated_price: float | None = None
    strategy_version: str | None = None
    planner_version: str = "order_planner_v1"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ticker"] = str(data["ticker"]).upper().strip()
        return data
