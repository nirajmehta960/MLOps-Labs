"""
Inference module for the financial condition model.

Converts API payload → feature matrix, loads model, returns prediction (0 or 1).
"""

import joblib
from pathlib import Path

from src.features import preprocess_inference_data

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model"


def predict_data_financial(financial_request_dict: dict):
    """
    Predict the financial condition for the input API payload.
    Returns a 1D numpy array with predicted class (0 = needs improvement, 1 = good).
    """
    # Step 1: Convert request dict to feature matrix (same pipeline as training)
    X = preprocess_inference_data(financial_request_dict)

    # Step 2: Load persisted model
    model = joblib.load(MODEL_DIR / "financial_linear_model.pkl")

    # Step 3: Run prediction
    y_pred = model.predict(X)
    return y_pred

