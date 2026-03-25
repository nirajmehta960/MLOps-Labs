"""
Training script for the financial condition predictor.

Loads raw data, engineers features, trains a Linear Booster XGBoost classifier,
and persists the model and encoder to disk for use by the API.
"""

from pathlib import Path

import joblib
from xgboost import XGBClassifier

from src.data import load_data, split_data
from src.features import engineer_target_and_features

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model"


def fit_model_financial(X_train, y_train):
    """
    Train a Linear Booster XGBoost classifier and save to model/financial_linear_model.pkl.
    Uses gblinear instead of tree-based booster for interpretability and speed.
    """
    # Instantiate Linear Booster (linear model, not tree-based)
    xgb_classifier = XGBClassifier(
        booster="gblinear",
        random_state=12,
        learning_rate=0.1,
        reg_lambda=1.0,
        n_estimators=200,
    )

    # Fit on training data
    xgb_classifier.fit(X_train, y_train)

    # Persist model for inference (predict.py loads this)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(xgb_classifier, MODEL_DIR / "financial_linear_model.pkl")


if __name__ == "__main__":
    # Step 1: Load raw CSV from data/financial_data.csv
    df_raw = load_data()

    # Step 2: Engineer target (good/bad) and features; encode categoricals; save encoder
    X, y = engineer_target_and_features(df_raw, is_training=True)

    # Step 3: Split into train (70%) and test (30%) with fixed random seed
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: Train model and save to disk
    fit_model_financial(X_train, y_train)

