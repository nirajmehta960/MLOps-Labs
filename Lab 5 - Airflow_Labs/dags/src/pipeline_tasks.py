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
    Load the Wisconsin breast cancer dataset from sklearn into a Pandas DataFrame.
    The dataset is then serialized and saved as a pickle file to a persistent 
    working directory (`/opt/airflow/working_data/raw.pkl`).
    This simulates data ingestion from an external source or data warehouse.

    Returns:
        str: Absolute path to the saved pickle file (for XCom and downstream tasks).
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
    Preprocess the raw ingested dataset for machine learning.
    This includes:
    1. Performing a stratified train/test split (75/25) to maintain class distributions.
    2. Fitting a `StandardScaler` on the training set to normalize numerical features.
    3. Transforming both training and testing sets using the fitted scaler.
    4. Persisting the resulting numpy arrays, scaler, and metadata into `preprocessed.pkl`.

    Args:
        raw_path: Path to `raw.pkl` from the `ingest_dataset` task.

    Returns:
        str: Absolute path to `preprocessed.pkl` containing the processed arrays and scaler.
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
    Train a Logistic Regression classifier on the preprocessed training set.
    Evaluates the model on the test set to compute its accuracy.
    The fitted model is saved to the model directory for potential production use.

    Args:
        preprocessed_path: Path to the preprocessed data bundle.

    Returns:
        dict: A dictionary containing the model name, accuracy, and path to the saved model.
              This will be pushed to Airflow XCom for the model comparison task.
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
    Train a Random Forest classifier on the preprocessed training set.
    Evaluates the model on the test set to compute its accuracy.
    The fitted model is saved to the model directory.

    Args:
        preprocessed_path: Path to the preprocessed data bundle.

    Returns:
        dict: A dictionary containing the model name, accuracy, and path to the saved model.
              This is returned to Airflow XCom for the downstream comparison task.
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
    Evaluate multiple trained models and select the best one based on test accuracy.
    The winning model is then copied and saved as `best_model.pkl`. This acts as
    the single source of truth for the model to deploy.

    Args:
        lr_result: XCom dictionary from the Logistic Regression training task.
        rf_result: XCom dictionary from the Random Forest training task.

    Returns:
        dict: A summary dictionary containing the name, accuracy, and path of the winning model.
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
    Determine if the best model meets or exceeds the required quality threshold.
    This boolean result determines whether the pipeline branches to deployment
    (recording manifest) or fails out.

    Args:
        comparison: Summary dictionary from the `compare_and_select_best` task.
        threshold: Minimum accuracy required to pass the quality gate.
        
    Returns:
        bool: True if the model accuracy >= threshold, False otherwise.
    """
    ok = comparison["best_accuracy"] >= threshold
    print(
        f"Quality gate ({threshold:.2%}): "
        f"{'PASS' if ok else 'FAIL'} — best accuracy {comparison['best_accuracy']:.4f}"
    )
    return ok


def write_production_manifest(comparison: dict) -> None:
    """
    Write deployment metadata (manifest) to `working_data/production_manifest.json`.
    This JSON contains essential details about the winning model. In a real-world
    setup, this manifest could be picked up by a CD pipeline (like ArgoCD or GitHub Actions)
    to perform a model rollout.

    Args:
        comparison: Summary dictionary of the best model (retrieved via XCom).
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
