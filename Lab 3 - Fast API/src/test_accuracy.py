"""
Standalone script to evaluate the trained Linear Booster model on the test set.

Run after training: python -m src.test_accuracy
"""

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score

from src.data import load_data, split_data
from src.features import engineer_target_and_features

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model"


def main():
    """Load model and data, then compute and print test accuracy."""
    model_path = MODEL_DIR / "financial_linear_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run 'python -m src.train' first."
        )

    # Load persisted model
    model = joblib.load(model_path)

    # Load data and engineer features (same pipeline as training)
    df_raw = load_data()
    X, y = engineer_target_and_features(df_raw, is_training=True)

    # Get test split (same random_state as train.py)
    _, X_test, _, y_test = split_data(X, y)

    # Evaluate and print
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy (Linear Booster): {acc:.2%}")


if __name__ == "__main__":
    main()
