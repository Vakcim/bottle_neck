from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.settings import get_settings


FEATURE_COLUMNS = [
    # market features
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

    # news features
    "news_count_1d",
    "unique_domains_1d",
    "english_news_count_1d",
    "russian_news_count_1d",
    "news_count_3d",
    "unique_domains_3d",
    "news_count_7d",
    "unique_domains_7d",
]


def main():
    settings = get_settings()

    dataset_path = (
        settings.data_path
        / "datasets"
        / "model_dataset_day_h5_thr0.015.parquet"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    df = pd.read_parquet(dataset_path)
    df = df.sort_values("date").reset_index(drop=True)

    # Временное разделение, без перемешивания.
    # Это важно: нельзя обучаться на будущем и тестировать на прошлом.
    train = df[df["date"] < "2025-01-01"].copy()
    test = df[df["date"] >= "2025-01-01"].copy()

    X_train = train[FEATURE_COLUMNS].fillna(0)
    y_train = train["target"]

    X_test = test[FEATURE_COLUMNS].fillna(0)
    y_test = test["target"]

    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"Test shape: {X_test.shape}")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=20,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, pred, digits=4))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, pred, labels=[-1, 0, 1]))

    clf = model.named_steps["clf"]
    importances = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": clf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nFeature importances:")
    print(importances.to_string(index=False))

    out_dir = settings.data_path / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    import joblib

    model_path = out_dir / "baseline_random_forest.joblib"
    joblib.dump(model, model_path)

    logger.info(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
