#!/usr/bin/env bash
set -e

cd /root/bottle_neck
source /root/bottle_neck/.venv/bin/activate
export PYTHONPATH="/root/bottle_neck"

echo "1/8 Update daily candles"
python scripts/update_daily_candles.py

echo "2/8 Build market features"
python scripts/build_market_features.py

echo "3/8 Build live features"
python scripts/build_live_features.py

echo "4/8 Build model dataset"
python scripts/build_model_dataset.py --horizon-days 5 --threshold 0.015

echo "5/8 Generate candidate signals"
python scripts/generate_candidate_signals.py

echo "6/8 Update paper portfolio"
python scripts/paper_portfolio_tracker.py

echo "7/8 Send Telegram report"
python scripts/send_telegram_report.py

echo "8/8 Health check"
python scripts/health_check.py

echo "Done"
