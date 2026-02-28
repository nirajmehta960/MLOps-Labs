"""
Feature engineering and preprocessing for the financial condition model.

Handles both training (target + encoding) and inference (API payload → model input).
"""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model"


def engineer_target_and_features(df: pd.DataFrame, is_training: bool = True):
    """
    Takes the raw financial dataframe and outputs the cleaned Feature Matrix (X)
    and Target Vector (y) ready for ML training or inference.
    """
    # --- Step 1: Engineer target (only during training) ---
    if is_training:
        # Binary label: 1 = good financial condition, 0 = needs improvement
        # Criteria: credit_score >= 700, savings_to_income > 3.5, debt_to_income < 3.0
        good_financial_condition = (
            (df["credit_score"] >= 700)
            & (df["savings_to_income_ratio"] > 3.5)
            & (df["debt_to_income_ratio"] < 3.0)
        ).astype(int)
        y = good_financial_condition
    else:
        y = None

    # --- Step 2: Drop non-feature columns ---
    # Remove identifiers, target-defining columns, and metadata
    columns_to_drop = [
        "user_id",
        "record_date",
        "credit_score",
        "savings_to_income_ratio",
        "debt_to_income_ratio",
    ]
    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # --- Step 3: Convert categorical columns to numeric ---
    categorical_cols = [
        "gender",
        "education_level",
        "employment_status",
        "job_title",
        "has_loan",
        "loan_type",
        "region",
    ]
    existing_cat_cols = [col for col in categorical_cols if col in X.columns]

    if is_training:
        # Fit encoder on training data; save for reuse at inference
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[existing_cat_cols] = encoder.fit_transform(X[existing_cat_cols])
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(encoder, MODEL_DIR / "categorical_encoder.pkl")
    else:
        # Load fitted encoder and transform API payload
        encoder = joblib.load(MODEL_DIR / "categorical_encoder.pkl")
        X[existing_cat_cols] = encoder.transform(X[existing_cat_cols])

    return X, y


def preprocess_inference_data(financial_request_dict: dict) -> np.ndarray:
    """
    Converts incoming Pydantic API payload into model-ready numpy array.
    Wraps dict as single-row DataFrame, runs engineer_target_and_features (inference mode),
    returns numeric feature matrix for model.predict().
    """
    df = pd.DataFrame([financial_request_dict])
    X, _ = engineer_target_and_features(df, is_training=False)
    return X.values
