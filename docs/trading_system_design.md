# Trading System Design

Status: architecture design document for the current safe trading-bot refactor.

Repository target path:

```text
docs/trading_system_design.md
```

This document intentionally does not introduce live trading. The system is limited to paper mode and T-Invest sandbox mode. No leverage, no short selling, no real-money orders.

---

## 1. Design Goals

The immediate goal is not to add a more complex ML model. The goal is to make the trading system safe, observable, and testable.

Priority order:

1. Document the current pipeline.
2. Introduce a central `OrderIntent`.
3. Introduce `OrderPlanner`.
4. Move risk checks and sizing out of executors.
5. Make sandbox executors execute approved intents only.
6. Add explicit skip/exit reason codes.
7. Persist TP/SL/planned exit in position state.
8. Add `ExitPlanner` first in dry-run mode.
9. Add news/event loop only after the order/risk lifecycle is stable.

Core principle:

```text
Alpha model may be wrong.
Risk layer must not fail catastrophically.
```

---

## 2. Hard Constraints

The system must enforce:

```text
No real orders.
Only paper and sandbox execution.
No leverage.
No short selling.
No trading with insufficient data.
No trading when daily proba_1 is below threshold.
No duplicate active orders.
No buying if an open position already exists for the ticker.
No buying if an active order already exists for the ticker.
No exceeding max_positions.
No event trade after excessive intraday move.
No event trade with wide spread.
All important actions must be logged.
All important actions must be visible in Telegram reports.
All skipped decisions must be recorded with reason_code.
```

---

## 3. Target Architecture

```text
Daily Market Alpha Model
        ↓
Intraday News/Event Model
        ↓
Signal Fusion Layer
        ↓
Decision Layer
        ↓
Risk Layer
        ↓
Order Planner
        ↓
Paper/Sandbox Executor
        ↓
Reconciliation
        ↓
Position Tracker
        ↓
Exit Planner
        ↓
Telegram Reports / Monitoring
```

Separation of responsibilities:

```text
Model: produces scores, not orders.
Decision Layer: decides whether a candidate may become an order candidate.
Risk Layer: checks safety, cash, exposure, leverage/short constraints.
OrderPlanner: computes lots, limit price, TP, SL, planned exit, reason codes.
Executor: submits already approved OrderIntent only.
Reconciliation: syncs broker order state.
Position Tracker: maintains persistent position state.
ExitPlanner: decides hold/exit for open positions.
Reports: show accepted, skipped, submitted, filled, cancelled, and exit decisions.
```

---

## 4. Current Public Repository Map

The public GitHub repository currently exposes the following top-level structure:

```text
config/
scripts/
src/
.env.example
.gitignore
README.md
pyproject.toml
```

The public README describes the project as a T-Invest data-layer v0.1 with candle download, Parquet/DuckDB storage, config files, GDELT placeholder, and CLI scripts. It also states that the public README version does not buy or sell and is a data layer only.

Important public scripts currently visible:

```text
scripts/update_daily_candles.py
scripts/build_market_features.py
scripts/build_live_features.py
scripts/build_model_dataset.py
scripts/generate_candidate_signals.py
scripts/paper_portfolio_tracker.py
scripts/send_telegram_report.py
scripts/run_daily_pipeline.sh
scripts/backtest_model_signals.py
scripts/backtest_portfolio_model.py
scripts/backtest_portfolio_model_v2.py
scripts/grid_search_portfolio_v2.py
scripts/walk_forward_backtest.py
scripts/download_gdelt_news.py
scripts/build_news_features.py
```

Important public source modules currently visible:

```text
src/settings.py
src/data/storage.py
src/connectors/
src/features/
```

Gap versus the described server state:

```text
The public repository does not currently show the sandbox executor, sandbox reconcile,
sandbox position tracker, sandbox exit executor, lifecycle report, or health-check scripts
mentioned in the operational description. Those may exist only on the server and should
be inspected before implementing the executor integration steps.
```

---

## 5. Current Public Daily Pipeline

The public `scripts/run_daily_pipeline.sh` executes:

```text
1. update_daily_candles.py
2. build_market_features.py
3. build_live_features.py
4. build_model_dataset.py --horizon-days 5 --threshold 0.015
5. generate_candidate_signals.py
6. paper_portfolio_tracker.py
7. send_telegram_report.py
```

Current public data flow:

```text
T-Invest candles
    ↓
data/candles/*.parquet
    ↓
market features
    ↓
data/features/market/*_day.parquet
    ↓
live features
    ↓
data/live/live_features_day.parquet
    ↓
model dataset
    ↓
data/datasets/model_dataset_day_h5_thr0.015.parquet
    ↓
candidate signals
    ↓
data/signals/signals_<date>_candidate_v1.csv
    ↓
paper positions/trades/equity/report
    ↓
data/paper/positions.csv
data/paper/trades.csv
data/paper/equity.csv
data/paper/paper_report.csv
    ↓
Telegram report
```

---

## 6. Current Public Script Responsibilities

### `scripts/generate_candidate_signals.py`

Current role:

```text
Loads config/strategy_candidate_v1.yaml.
Loads model dataset.
Loads live features.
Trains RandomForest on the full dataset.
Scores latest live rows.
Creates proba columns.
Filters by proba_1 >= threshold.
Excludes configured tickers.
Keeps top max_positions rows.
Writes CSV into data/signals/.
```

Current inputs:

```text
config/strategy_candidate_v1.yaml
data/datasets/model_dataset_day_h5_thr0.015.parquet
data/live/live_features_day.parquet
```

Current outputs:

```text
data/signals/signals_<latest_date>_candidate_v1.csv
```

Current limitations:

```text
No active order check.
No open position check.
No OrderIntent.
No risk-based sizing.
No planned order lifecycle.
No persistent skip reason table.
Model is trained inline inside the signal script.
```

### `scripts/paper_portfolio_tracker.py`

Current role:

```text
Loads latest candidate signals.
Filters by threshold and excluded tickers.
Closes paper positions when planned_exit_date is reached.
Calculates equity.
Opens paper positions if slots are free.
Uses equal capital allocation: equity / max_positions.
Persists paper positions, trades, equity, and report.
```

Current inputs:

```text
config/strategy_candidate_v1.yaml
data/live/live_features_day.parquet
data/signals/signals_*_candidate_v1.csv
data/paper/positions.csv, if exists
data/paper/trades.csv, if exists
data/paper/equity.csv, if exists
```

Current outputs:

```text
data/paper/positions.csv
data/paper/trades.csv
data/paper/equity.csv
data/paper/paper_report.csv
```

Current limitations:

```text
No TP/SL.
Only time exit.
No OrderIntent.
No reason codes.
No risk-based sizing.
No active broker/sandbox order state.
Position state lacks linked_intent_id, source, TP, SL, strategy version.
```

### `scripts/send_telegram_report.py`

Current role:

```text
Reads latest paper report.
Reads latest signals.
Reads current open paper positions.
Sends a Telegram summary.
```

Current inputs:

```text
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
data/paper/paper_report.csv
data/paper/positions.csv
data/signals/signals_*_candidate_v1.csv
```

Current limitations:

```text
Does not show skipped decisions.
Does not show planned/submitted/filled/cancelled order lifecycle.
Does not show TP/SL or exit reasons.
Does not show sandbox lifecycle state in the public version.
```

### `scripts/build_live_features.py`

Current role:

```text
Loads market features.
Optionally merges daily news features if they exist.
Fills missing news-count columns with zero.
Writes live feature table.
```

Current inputs:

```text
data/features/market/*_day.parquet
data/features/news/daily_news_features.parquet, optional
```

Current outputs:

```text
data/live/live_features_day.parquet
```

### `scripts/build_model_dataset.py`

Current role:

```text
Loads market features.
Optionally merges daily news features if they exist.
Creates future_return and class target for a fixed horizon and threshold.
Writes model dataset.
```

Current inputs:

```text
data/features/market/*_day.parquet
data/features/news/daily_news_features.parquet, optional
```

Current outputs:

```text
data/datasets/model_dataset_day_h<horizon>_thr<threshold>.parquet
```

---

## 7. Target Storage Layout for the Refactor

Add only the minimal new storage paths first:

```text
data/orders/
  order_intents.csv
  skipped_decisions.csv
  sandbox_submissions.csv

data/portfolio/
  positions.csv

data/reports/
  lifecycle_report.csv
```

Do not remove existing public paths immediately:

```text
data/paper/positions.csv
data/paper/trades.csv
data/paper/equity.csv
data/paper/paper_report.csv
```

Migration rule:

```text
Keep legacy paper files working.
Introduce new state files alongside them.
Switch reports gradually.
```

---

## 8. OrderIntent

`OrderIntent` is the central object shared by paper, sandbox, future live, backtest, reconciliation, and reports.

Required fields:

```text
intent_id
created_at
mode
source
side
ticker
figi
lots
estimated_price
limit_price
take_profit_price
stop_loss_price
planned_exit_date
max_loss_rub
expected_order_value
reason_code
status
model_version
strategy_version
planner_version
linked_signal_id
linked_event_id
```

Allowed `mode`:

```text
paper
sandbox
live
```

For now, `live` must not be used by any executor.

Allowed `source`:

```text
daily_alpha
news_event
exit
```

Allowed `side`:

```text
BUY
SELL
```

Allowed `status`:

```text
planned
submitted
filled
cancelled
rejected
expired
skipped
```

Implementation v1:

```text
src/execution/order_intent.py
```

Use the style that best fits the repository. Recommended first version:

```text
Python dataclass + to_dict/from_dict helpers
```

Pydantic may be added later if config/schema validation becomes heavier.

---

## 9. Reason Codes

Use a single source of truth:

```text
src/execution/reason_codes.py
```

Reason codes:

```text
BUY_DAILY_SIGNAL
BUY_NEWS_EVENT

SKIP_LOW_PROBA
SKIP_EXCLUDED_TICKER
SKIP_ALREADY_POSITION
SKIP_ACTIVE_ORDER
SKIP_MAX_POSITIONS
SKIP_INSUFFICIENT_CASH
SKIP_LOW_LIQUIDITY
SKIP_WIDE_SPREAD
SKIP_STALE_PRICE
SKIP_STALE_DATA
SKIP_CHASE_RISK
SKIP_LOW_CONFIDENCE_NEWS
SKIP_LOW_SOURCE_RELIABILITY
SKIP_NEGATIVE_NEWS
SKIP_MARKET_NOT_CONFIRMED
SKIP_RISK_LIMIT
SKIP_SIZE_TOO_SMALL

SELL_TAKE_PROFIT
SELL_STOP_LOSS
SELL_TIME_EXIT
SELL_NEGATIVE_NEWS
HOLD_NO_EXIT_CONDITION

ERROR_TINVEST_API
ERROR_POSITION_RECONCILE
ERROR_CONFIG_INVALID
ERROR_MODEL_MISSING
```

Every candidate must produce one of:

```text
approved OrderIntent
skipped decision with reason_code
error decision with reason_code
```

---

## 10. OrderPlanner

Implementation target:

```text
src/execution/order_planner.py
```

Inputs:

```text
signal/event
portfolio_state
active_orders
positions
instrument_metadata
latest_price
risk_config
execution_config
```

Outputs:

```text
OrderIntent
or skipped decision with reason_code
```

For daily mode:

```text
Reject if proba_1 < threshold.
Reject if ticker excluded.
Reject if active order exists.
Reject if open position exists.
Reject if max_positions reached.
Reject if insufficient cash.
Reject if size is too small.
Otherwise create BUY OrderIntent.
```

For event mode:

```text
Reject if news confidence too low.
Reject if severity too low.
Reject if source reliability too low.
Reject if event is stale.
Reject if no fresh price.
Reject if spread too wide.
Reject if intraday move exceeds max_chase_pct.
Reject if active order exists.
Reject if position already exists.
Reject if max_positions reached.
Reject if risk checks fail.
Otherwise create BUY OrderIntent with smaller sizing.
```

---

## 11. Risk-Based Sizing

Target config:

```yaml
risk:
  max_positions: 3
  risk_per_trade_pct: 0.005
  max_position_value_pct: 0.20
  cash_buffer_pct: 0.05
  allow_short: false
  allow_leverage: false
```

Sizing formula:

```text
risk_per_trade_rub = portfolio_value * risk_per_trade_pct

risk_per_share = entry_price - stop_loss_price

shares_by_risk = risk_per_trade_rub / risk_per_share
shares_by_cash = available_cash_after_buffer / entry_price

shares = min(shares_by_risk, shares_by_cash)

position_value_limit = portfolio_value * max_position_value_pct
shares_by_position_limit = position_value_limit / entry_price

shares = min(shares, shares_by_position_limit)

lots = floor(shares / lot_size)
```

Reject with `SKIP_SIZE_TOO_SMALL` if:

```text
lots < 1
```

Event trade adjustment:

```text
risk_per_trade_rub *= news_position_size_multiplier
```

Recommended first module:

```text
src/risk/sizing.py
```

---

## 12. Price Planning

Daily BUY:

```text
buy_limit_price = last_close * (1 - buy_limit_offset_pct)
```

Event BUY:

```text
buy_limit_price = current_price * (1 - event_pullback_offset_pct)
```

Daily TP/SL:

```text
take_profit_price = entry_price * (1 + take_profit_pct)
stop_loss_price   = entry_price * (1 - stop_loss_pct)
planned_exit_date = after hold_days trading candles
```

Event TP/SL:

```text
take_profit_price = entry_price * (1 + event_take_profit_pct)
stop_loss_price   = entry_price * (1 - event_stop_loss_pct)
planned_exit_date = after event_hold_days trading candles
```

Config example:

```yaml
execution:
  buy_limit_offset_pct: 0.005
  take_profit_pct: 0.035
  stop_loss_pct: 0.020
  sell_limit_offset_pct: 0.005
  event_pullback_offset_pct: 0.007
  max_chase_pct: 0.03
  max_spread_pct: 0.01

event_execution:
  event_take_profit_pct: 0.025
  event_stop_loss_pct: 0.020
  event_hold_days: 2
  news_position_size_multiplier: 0.5
```

---

## 13. Position State

Target position fields:

```text
position_id
ticker
figi
entry_date
entry_price
quantity
lots
source
take_profit_price
stop_loss_price
planned_exit_date
entry_order_id
linked_intent_id
linked_signal_id
linked_event_id
strategy_version
status
```

Allowed status:

```text
open
closing
closed
```

The current public paper position file should be extended gradually instead of being replaced abruptly.

---

## 14. ExitPlanner

Implementation target:

```text
src/execution/exit_planner.py
```

Input:

```text
position
current_price
latest_live_date
optional negative_news_event
```

Decision logic v1:

```text
if current_price >= take_profit_price:
    SELL_TAKE_PROFIT
elif current_price <= stop_loss_price:
    SELL_STOP_LOSS
elif latest_live_date >= planned_exit_date:
    SELL_TIME_EXIT
else:
    HOLD_NO_EXIT_CONDITION
```

Negative news exit, later:

```text
if negative_news_event and severity high and confidence high:
    SELL_NEGATIVE_NEWS
```

Initial integration:

```text
dry-run only
no real sell orders
sandbox sell only after dry-run reports are stable
```

---

## 15. Sandbox Executor Contract

Executor must not decide what to buy.

Executor responsibility:

```text
Load approved OrderIntent.
Submit to T-Invest sandbox.
Save broker_order_id.
Save submission/failure status.
Log result.
Send Telegram report.
```

Executor must not:

```text
Select tickers.
Compute lots.
Compute limit price.
Compute TP/SL.
Ignore active orders.
Ignore existing positions.
Create live orders.
```

---

## 16. Reconciliation Contract

Reconciliation should:

```text
Load submitted intents.
Fetch sandbox order state.
Update statuses:
  submitted
  filled
  cancelled
  rejected
  expired
Cancel stale active orders according to config.
Persist broker status fields.
Report lifecycle changes.
```

---

## 17. Daily Loop

Current public daily loop remains the entrypoint.

Target daily loop:

```text
update_daily_candles
build_market_features
build_live_features
build_model_dataset
generate_candidate_signals
plan_daily_order_intents
paper/sandbox executor
reconcile
position tracker
exit planner dry-run
telegram lifecycle report
```

Do not break existing cron. Add new scripts beside existing ones:

```text
scripts/plan_daily_order_intents.py
scripts/run_exit_planner.py
scripts/run_lifecycle_report.py
```

---

## 18. Intraday News/Event Loop

Do not make news buy directly.

Target event flow:

```text
NEWS RECEIVED
    ↓
Deduplicate news
    ↓
Check source reliability
    ↓
Extract entities / tickers / sectors
    ↓
Classify event
    ↓
Estimate sentiment / severity / confidence / horizon
    ↓
If confidence low: LOG_ONLY / NO_TRADE
    ↓
If event stale: NO_TRADE
    ↓
Map to candidate tickers
    ↓
For each ticker:
    Load latest market state
    Check price, spread, liquidity, volatility
    Check chase risk
    Check active orders and positions
    Check max_positions
    Combine with market state
    Run Risk Layer
    Create sandbox-only BUY OrderIntent if allowed
```

News/event output schema:

```json
{
  "event_id": "...",
  "published_at": "...",
  "collected_at": "...",
  "source": "...",
  "source_reliability": 0.0,
  "event_type": "sanctions_relief | sanctions_risk | dividend | earnings | buyback | m&a | lawsuit | regulation | production | geopolitical | other",
  "affected_scope": "ticker | sector | market",
  "affected_tickers": ["SBER", "LKOH"],
  "sentiment": "positive | negative | neutral | mixed",
  "severity": 0.0,
  "confidence": 0.0,
  "horizon": "intraday | days | weeks",
  "summary": "short explanation"
}
```

---

## 19. News Storage

Target fields:

```text
news_id
url
source
title
body/summary
published_at
collected_at
language
dedup_hash
mentioned_entities
validated_tickers
event_type
sentiment
severity
confidence
source_reliability
llm_model_version
prompt_version
created_at
```

Mandatory timestamps:

```text
published_at
collected_at
```

Without both timestamps, a news item must not be used for trading decisions.

---

## 20. Look-Ahead Bias Rules

Daily model:

```text
Use only market/news data available before decision_time.
decision_time = 20:30 Europe/Moscow
```

Intraday event loop:

```text
Use only news where collected_at <= current_decision_time.
Use only prices available at or before current_decision_time.
```

Forbidden:

```text
Future prices.
Future candles.
News collected after the decision.
Rewritten articles without original timestamp.
```

---

## 21. Validation Plan

Do not trust strong backtest results without additional checks.

Required checks:

```text
walk-forward
purged/embargo split
transaction costs
spread
slippage
limit order fill model
no-trade baseline
random baseline
buy-and-hold benchmark
simple momentum baseline
feature importance stability
calibration curve
Brier score
performance by year
performance by ticker
performance by sector
stress test for worse fills
```

Specific checks:

```text
Did 1–3 trades produce most of the result?
Does the result survive higher costs and worse fills?
Was threshold/TP/SL/hold_days overfit?
Does performance degrade on unseen periods?
```

---

## 22. Minimal File Change Plan

Do not mass-refactor. Add small modules first.

### Step A: Add documentation

```text
docs/trading_system_design.md
```

### Step B: Add execution primitives

```text
src/execution/__init__.py
src/execution/order_intent.py
src/execution/reason_codes.py
src/execution/order_planner.py
```

### Step C: Add risk sizing

```text
src/risk/__init__.py
src/risk/sizing.py
src/risk/checks.py
```

### Step D: Add lifecycle storage helpers

```text
src/execution/order_storage.py
```

### Step E: Add planning script without touching executor

```text
scripts/plan_daily_order_intents.py
```

This script should:

```text
read latest signals
read current paper/sandbox positions
read active orders if available
read instrument metadata
read risk/execution config
call OrderPlanner
save planned intents and skipped decisions
print summary
```

### Step F: Connect sandbox BUY executor

Only after Step E works:

```text
modify existing sandbox BUY executor
```

Executor should:

```text
read approved planned OrderIntent
submit sandbox order
write broker_order_id
update status to submitted/rejected
```

### Step G: Extend position state

```text
src/portfolio/state.py
```

Fields to add:

```text
source
linked_intent_id
take_profit_price
stop_loss_price
planned_exit_date
strategy_version
```

### Step H: Add ExitPlanner dry-run

```text
src/execution/exit_planner.py
scripts/run_exit_planner.py
```

Dry-run report fields:

```text
ticker
position_id
current_price
take_profit_price
stop_loss_price
planned_exit_date
decision
reason_code
```

---

## 23. Verification Commands

After adding the design doc:

```bash
test -f docs/trading_system_design.md
python -m compileall src scripts
```

After adding OrderIntent:

```bash
python - <<'PY'
from src.execution.order_intent import OrderIntent
print(OrderIntent)
PY
```

After adding OrderPlanner:

```bash
python - <<'PY'
from src.execution.order_planner import OrderPlanner
print(OrderPlanner)
PY
```

After adding planning script:

```bash
python scripts/plan_daily_order_intents.py
ls -la data/orders
```

After integrating sandbox executor:

```bash
python scripts/plan_daily_order_intents.py
python scripts/run_sandbox_buy_executor.py --dry-run
```

Only if dry-run is correct:

```bash
python scripts/run_sandbox_buy_executor.py
```

After adding ExitPlanner:

```bash
python scripts/run_exit_planner.py --dry-run
```

---

## 24. Telegram Report Target

The lifecycle report should show:

```text
Signals:
  ticker, proba_1, close, source

Accepted intents:
  ticker, side, lots, limit_price, TP, SL, planned_exit_date, reason_code

Skipped decisions:
  ticker, reason_code, short explanation

Orders:
  planned, submitted, filled, cancelled, rejected, expired

Positions:
  ticker, entry, current, TP, SL, planned_exit_date, hold/exit decision

Errors:
  reason_code, file/script, message
```

---

## 25. Deferred Work

Do not implement until OrderPlanner/Risk/ExitPlanner are stable:

```text
Meta-model
Second ML model
LLM-based direct trading decisions
Telegram/social trading signals
Live trading
Leverage
Short selling
```

News/event pipeline v1 may be added after lifecycle stabilization, but only as:

```text
storage
entity linking
classification
event candidates
Telegram-only report
```

Trading from event candidates must remain sandbox-only and must pass OrderPlanner and Risk Layer.

