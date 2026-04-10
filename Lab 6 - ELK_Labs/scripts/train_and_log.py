#!/usr/bin/env python3
"""
Train the Lab 3 financial XGBoost (gblinear) model and append JSON Lines to logs/training.jsonl
for ingestion by Logstash → Elasticsearch → Kibana.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
REPO_ROOT = BASE_DIR.parent
LAB3 = REPO_ROOT / "Lab 3 - Fast_API"

sys.path.insert(0, str(LAB3))

from src.data import load_data, split_data  # noqa: E402
from src.features import engineer_target_and_features  # noqa: E402

LOG_DIR = BASE_DIR / "logs"
TRAIN_LOG = LOG_DIR / "training.jsonl"
MODEL_DIR = LAB3 / "model"
DATASET = "ml.training"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def emit(event: dict) -> None:
    event.setdefault("timestamp", utc_now_iso())
    event.setdefault("dataset", DATASET)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False)
    with TRAIN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    emit({"level": "INFO", "message": "training_pipeline_started", "ml": {"component": "lab6-elk"}})
    try:
        if not (LAB3 / "data" / "financial_data.csv").is_file():
            raise FileNotFoundError(
                f"Missing {LAB3 / 'data' / 'financial_data.csv'}. "
                "Run from the MLOps Labs repo; Lab 3 data must be present."
            )

        df_raw = load_data()
        emit(
            {
                "level": "INFO",
                "message": "data_loaded",
                "ml": {"rows": int(len(df_raw))},
            }
        )

        X, y = engineer_target_and_features(df_raw, is_training=True)
        X_train, X_test, y_train, y_test = split_data(X, y)
        emit(
            {
                "level": "INFO",
                "message": "data_split",
                "ml": {
                    "train_rows": int(len(X_train)),
                    "test_rows": int(len(X_test)),
                },
            }
        )

        emit({"level": "INFO", "message": "model_training_started", "ml": {"booster": "gblinear"}})

        model = XGBClassifier(
            booster="gblinear",
            random_state=12,
            learning_rate=0.1,
            reg_lambda=1.0,
            n_estimators=200,
        )
        model.fit(X_train, y_train)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / "financial_linear_model.pkl")

        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        emit(
            {
                "level": "INFO",
                "message": "model_training_completed",
                "ml": {"test_accuracy": acc},
            }
        )
        emit({"level": "INFO", "message": "training_pipeline_finished", "ml": {"status": "success"}})
    except Exception as exc:
        emit(
            {
                "level": "ERROR",
                "message": "training_pipeline_failed",
                "ml": {"error": str(exc)},
            }
        )
        raise


if __name__ == "__main__":
    main()
