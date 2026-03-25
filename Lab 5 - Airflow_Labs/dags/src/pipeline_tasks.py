"""
Pipeline callables for the breast cancer training DAG.

Loads the Wisconsin breast cancer dataset, preprocesses with StandardScaler,
trains logistic regression and random forest, compares test accuracy, and writes
artifacts under /opt/airflow/working_data and /opt/airflow/model (Docker volume mounts).
"""
from __future__ import annotations

import os
import pickle

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paths inside containers; host folders working_data/ and model/ are mounted here.
WORKING_DIR = "/opt/airflow/working_data"
MODEL_DIR = "/opt/airflow/model"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RAW_NAME = "raw.pkl"
PREPROCESSED_NAME = "preprocessed.pkl"
BEST_MODEL = "best_model.pkl"


def ingest_dataset() -> str:
    """
    Load sklearn breast cancer frame, pickle to working_data/raw.pkl.

    Returns:
        str: Absolute path to the saved pickle file (for XCom / downstream tasks).
    """
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame
    path = os.path.join(WORKING_DIR, RAW_NAME)
    with open(path, "wb") as f:
        pickle.dump(df, f)
    print(f"Ingested {len(df)} rows to {path}")
    return path


def preprocess_data(raw_path: str) -> str:
    """
    Stratified train/test split, fit StandardScaler on train, persist arrays.

    Args:
        raw_path: Path to raw.pkl from ingest_dataset.

    Returns:
        str: Path to preprocessed.pkl containing arrays and scaler.
    """
    with open(raw_path, "rb") as f:
        df = pickle.load(f)
    target_col = "target"
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    y = df[target_col].values

    # 75% train / 25% test, stratified by label
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    out_path = os.path.join(WORKING_DIR, PREPROCESSED_NAME)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "X_train": X_train_s,
                "X_test": X_test_s,
                "y_train": y_train,
                "y_test": y_test,
                "scaler": scaler,
                "feature_names": feature_cols,
            },
            f,
        )
    print(f"Preprocessed data written to {out_path}")
    return out_path


def train_logistic_regression(preprocessed_path: str) -> dict:
    """
    Fit LogisticRegression, score on test set, save model to model/logistic_regression.pkl.

    Returns:
        dict: Keys name, accuracy, path (for XCom and compare_and_select_best).
    """
    with open(preprocessed_path, "rb") as f:
        bundle = pickle.load(f)

    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(bundle["X_train"], bundle["y_train"])
    acc = float(model.score(bundle["X_test"], bundle["y_test"]))
    print(f"LogisticRegression test accuracy: {acc:.4f}")

    path = os.path.join(MODEL_DIR, "logistic_regression.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    return {"name": "logistic_regression", "accuracy": acc, "path": path}


def train_random_forest(preprocessed_path: str) -> dict:
    """
    Fit RandomForestClassifier, score on test set, save model to model/random_forest.pkl.

    Returns:
        dict: Keys name, accuracy, path (for XCom and compare_and_select_best).
    """
    with open(preprocessed_path, "rb") as f:
        bundle = pickle.load(f)

    model = RandomForestClassifier(
        n_estimators=120, max_depth=8, random_state=42, n_jobs=-1
    )
    model.fit(bundle["X_train"], bundle["y_train"])
    acc = float(model.score(bundle["X_test"], bundle["y_test"]))
    print(f"RandomForest test accuracy: {acc:.4f}")

    path = os.path.join(MODEL_DIR, "random_forest.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)

    return {"name": "random_forest", "accuracy": acc, "path": path}


def compare_and_select_best(lr_result: dict, rf_result: dict) -> dict:
    """
    Pick the model with higher test accuracy and copy it to model/best_model.pkl.

    Args:
        lr_result: XCom dict from train_logistic_regression.
        rf_result: XCom dict from train_random_forest.

    Returns:
        dict: best_name, best_accuracy, per-model accuracies, best_model_path.
    """
    winner = lr_result if lr_result["accuracy"] >= rf_result["accuracy"] else rf_result
    best_path = os.path.join(MODEL_DIR, BEST_MODEL)
    with open(winner["path"], "rb") as src:
        model = pickle.load(src)
    with open(best_path, "wb") as dst:
        pickle.dump(model, dst)

    summary = {
        "best_name": winner["name"],
        "best_accuracy": winner["accuracy"],
        "logistic_accuracy": lr_result["accuracy"],
        "forest_accuracy": rf_result["accuracy"],
        "best_model_path": best_path,
    }
    print(
        f"Best model: {summary['best_name']} "
        f"({summary['best_accuracy']:.4f}) vs LR {summary['logistic_accuracy']:.4f} "
        f"/ RF {summary['forest_accuracy']:.4f}"
    )
    return summary


def meets_quality_gate(comparison: dict, threshold: float) -> bool:
    """
    Return True if best_accuracy meets or exceeds threshold (used by branch callable).
    """
    ok = comparison["best_accuracy"] >= threshold
    print(
        f"Quality gate ({threshold:.2%}): "
        f"{'PASS' if ok else 'FAIL'} — best accuracy {comparison['best_accuracy']:.4f}"
    )
    return ok


def write_production_manifest(comparison: dict) -> None:
    """
    Write deployment metadata JSON to working_data/production_manifest.json.

    Args:
        comparison: Dict returned by compare_and_select_best (from XCom).
    """
    import json

    manifest_path = os.path.join(WORKING_DIR, "production_manifest.json")
    payload = {
        "dag": "breast_cancer_training_pipeline",
        "best_model": comparison["best_name"],
        "accuracy": comparison["best_accuracy"],
        "artifact": comparison["best_model_path"],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {manifest_path}")
