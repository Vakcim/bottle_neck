"""Conservative risk-based position sizing."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite


@dataclass(slots=True)
class SizingInput:
    portfolio_value: float
    available_cash: float
    entry_price: float
    stop_loss_price: float
    lot_size: int = 1
    risk_per_trade_pct: float = 0.005
    max_position_value_pct: float = 0.20
    cash_buffer_pct: float = 0.05
    multiplier: float = 1.0


@dataclass(slots=True)
class SizingResult:
    lots: int
    shares: int
    expected_order_value: float
    max_loss_rub: float
    reason: str | None = None


def normalize_pct(value: float, default: float) -> float:
    """Accept both fractional and percentage-style config values.

    Examples:
    - 0.005 stays 0.005
    - 0.5 stays 0.5
    - 20 becomes 0.20
    """
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if not isfinite(x) or x < 0:
        return default
    if x > 1.0:
        return x / 100.0
    return x


def calculate_risk_based_lots(inp: SizingInput) -> SizingResult:
    lot_size = max(int(inp.lot_size or 1), 1)
    portfolio_value = float(inp.portfolio_value)
    available_cash = float(inp.available_cash)
    entry_price = float(inp.entry_price)
    stop_loss_price = float(inp.stop_loss_price)

    if portfolio_value <= 0 or available_cash <= 0:
        return SizingResult(0, 0, 0.0, 0.0, "non_positive_portfolio_or_cash")
    if entry_price <= 0 or stop_loss_price <= 0:
        return SizingResult(0, 0, 0.0, 0.0, "non_positive_price")
    if stop_loss_price >= entry_price:
        return SizingResult(0, 0, 0.0, 0.0, "invalid_stop_loss")

    risk_per_trade_pct = normalize_pct(inp.risk_per_trade_pct, 0.005)
    max_position_value_pct = normalize_pct(inp.max_position_value_pct, 0.20)
    cash_buffer_pct = normalize_pct(inp.cash_buffer_pct, 0.05)
    multiplier = max(float(inp.multiplier or 1.0), 0.0)

    cash_after_buffer = max(available_cash - portfolio_value * cash_buffer_pct, 0.0)
    risk_budget = portfolio_value * risk_per_trade_pct * multiplier
    risk_per_share = entry_price - stop_loss_price

    shares_by_risk = floor(risk_budget / risk_per_share)
    shares_by_cash = floor(cash_after_buffer / entry_price)
    shares_by_position_limit = floor((portfolio_value * max_position_value_pct) / entry_price)

    shares = min(shares_by_risk, shares_by_cash, shares_by_position_limit)
    lots = floor(shares / lot_size)
    shares = lots * lot_size

    expected_order_value = shares * entry_price
    max_loss_rub = shares * risk_per_share

    if lots < 1:
        return SizingResult(0, 0, 0.0, 0.0, "size_too_small")

    return SizingResult(lots, shares, expected_order_value, max_loss_rub, None)
