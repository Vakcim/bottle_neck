from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.settings import get_settings


FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "hl_range",
    "oc_return",
    "volume_change_1",
    "volatility_12",
    "volatility_24",
    "close_to_ma_12",
    "close_to_ma_24",
    "close_to_ma_72",
    "news_count_1d",
    "unique_domains_1d",
    "english_news_count_1d",
    "russian_news_count_1d",
    "news_count_3d",
    "unique_domains_3d",
    "news_count_7d",
    "unique_domains_7d",
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_model(model_cfg: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=int(model_cfg.get("n_estimators", 300)),
                    max_depth=int(model_cfg.get("max_depth", 6)),
                    min_samples_leaf=int(model_cfg.get("min_samples_leaf", 20)),
                    random_state=int(model_cfg.get("random_state", 42)),
                    n_jobs=-1,
                    class_weight=model_cfg.get("class_weight", "balanced"),
                ),
            ),
        ]
    )


def main():
    settings = get_settings()

    strategy_path = Path("config/strategy_candidate_v1.yaml")
    cfg = load_yaml(strategy_path)

    strategy_cfg = cfg["strategy"]
    model_cfg = cfg["model"]

    threshold = float(strategy_cfg["threshold"])
    hold_days = int(strategy_cfg["hold_days"])
    max_positions = int(strategy_cfg["max_positions"])
    excluded_tickers = set(strategy_cfg.get("excluded_tickers", []))

    train_dataset_path = (
        settings.data_path
        / "datasets"
        / "model_dataset_day_h5_thr0.015.parquet"
    )

    live_features_path = (
        settings.data_path
        / "live"
        / "live_features_day.parquet"
    )

    train_df = pd.read_parquet(train_dataset_path)
    train_df["date"] = pd.to_datetime(train_df["date"], utc=True)
    train_df = train_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    live_df = pd.read_parquet(live_features_path)
    live_df["date"] = pd.to_datetime(live_df["date"], utc=True)
    live_df = live_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    train_df = train_df[~train_df["ticker"].isin(excluded_tickers)].copy()
    live_df = live_df[~live_df["ticker"].isin(excluded_tickers)].copy()

    X_train = train_df[FEATURE_COLUMNS].fillna(0)
    y_train = train_df["target"]

    model = make_model(model_cfg)
    model.fit(X_train, y_train)

    latest_date = live_df["date"].max()
    latest = live_df[live_df["date"] == latest_date].copy()

    X_latest = latest[FEATURE_COLUMNS].fillna(0)

    classes = list(model.named_steps["clf"].classes_)
    proba = model.predict_proba(X_latest)

    for idx, cls in enumerate(classes):
        latest[f"proba_{cls}"] = proba[:, idx]

    if "proba_1" not in latest.columns:
        latest["proba_1"] = 0.0

    latest = latest.sort_values("proba_1", ascending=False)

    signals = latest[latest["proba_1"] >= threshold].copy()
    signals = signals.head(max_positions)

    out_dir = settings.data_path / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"signals_{latest_date.date()}_{strategy_cfg['name']}.csv"

    cols = [
        "date",
        "ticker",
        "close",
        "proba_1",
        "proba_0",
        "proba_-1",
        "news_count_7d",
        "return_1",
        "return_3",
        "return_6",
        "return_12",
        "close_to_ma_72",
        "volatility_12",
    ]

    available_cols = [c for c in cols if c in latest.columns]

    print("\nCandidate strategy:")
    print(strategy_cfg)

    print(f"\nTrain dataset date range: {train_df['date'].min()} → {train_df['date'].max()}")
    print(f"Live features date range: {live_df['date'].min()} → {live_df['date'].max()}")
    print(f"Latest available live date: {latest_date}")

    print(f"\nThreshold: {threshold}")
    print(f"Hold days: {hold_days}")
    print(f"Max positions: {max_positions}")
    print(f"Excluded: {sorted(excluded_tickers)}")

    print("\nTop probabilities:")
    print(latest[available_cols].head(10).to_string(index=False))

    if signals.empty:
        print("\nNo signals today.")
        latest[available_cols].to_csv(out_path, index=False)
    else:
        print("\nSignals:")
        print(signals[available_cols].to_string(index=False))
        signals[available_cols].to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")

    logger.info("Done")


if __name__ == "__main__":
    main()