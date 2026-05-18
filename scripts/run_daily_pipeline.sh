#!/usr/bin/env bash
set -e

export PYTHONPATH="$(pwd)"

echo "1/7 Update daily candles"
python scripts/update_daily_candles.py

echo "2/7 Build market features"
python scripts/build_market_features.py

echo "3/7 Build live features"
python scripts/build_live_features.py

echo "4/7 Build model dataset"
python scripts/build_model_dataset.py --horizon-days 5 --threshold 0.015

echo "5/7 Generate candidate signals"
python scripts/generate_candidate_signals.py

echo "6/7 Update paper portfolio"
python scripts/paper_portfolio_tracker.py

echo "7/7 Send Telegram report"
python scripts/send_telegram_report.py

echo "Done"