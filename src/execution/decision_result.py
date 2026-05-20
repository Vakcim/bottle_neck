from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .order_intent import OrderIntent
from .reason_codes import ReasonCode


@dataclass(slots=True)
class DecisionResult:
    accepted: bool
    reason_code: ReasonCode
    ticker: str

    decision_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    intent: OrderIntent | None = None
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["reason_code"] = self.reason_code.value
        data["created_at"] = self.created_at.isoformat()

        if self.intent is not None:
            data["intent"] = self.intent.to_dict()

        return data
