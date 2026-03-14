"""
Flask API for Lab 4: SavVio-style recommendation (GREEN/YELLOW/RED) plus GCS and BigQuery.
Uses training_scenarios-based model. Serves on PORT (default 8080). Set BUCKET_NAME for /upload.
"""
import os
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

PORT = int(os.environ.get("PORT", 8080))
MODEL_DIR = os.environ.get("MODEL_DIR", "/app")
MODEL_PATH = os.path.join(MODEL_DIR, "recommendation_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "feature_config.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)
feature_config = joblib.load(CONFIG_PATH)
FEATURE_ORDER = feature_config["feature_order"]
NUM_COLS = feature_config["num_cols"]
CAT_COLS = feature_config["cat_cols"]
LABELS = feature_config["labels"]


def transform_input(data: dict) -> np.ndarray:
    """Build one row in feature order, impute missing, scale + encode."""
    row = {}
    for c in FEATURE_ORDER:
        v = data.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0.0 if c in NUM_COLS else "Unknown"
        row[c] = v
    df = pd.DataFrame([row])
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    n = scaler.transform(df[NUM_COLS])
    c = encoder.transform(df[CAT_COLS]) if CAT_COLS else np.empty((1, 0))
    return np.hstack([n, c])


@app.route("/")
def home():
    return (
        "Welcome to the SavVio-style Recommendation API on Cloud Run. "
        "Try /predict (POST JSON with scenario features), /upload, /query."
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # For GET, run the model on a fixed example scenario and show the prediction,
    # so opening /predict in a browser immediately returns a recommendation.
    if request.method == "GET":
        example = {
            "monthly_income": 639.15,
            "monthly_expenses": 537.63,
            "savings_balance": 5248.61,
            "liquid_savings": 2658.73,
            "has_loan": 0,
            "loan_amount": 0.0,
            "monthly_emi": 0.0,
            "loan_interest_rate": 0.0,
            "loan_term_months": 0.0,
            "credit_score": 569,
            "employment_status": "Self-employed",
            "region": "Other",
            "discretionary_income": 101.51999999999998,
            "debt_to_income_ratio": 0.0,
            "saving_to_income_ratio": 4.159790346554018,
            "monthly_expense_burden_ratio": 0.8411640459985918,
            "emergency_fund_months": 4.9452783512824805,
            "product_price": 169.99,
            "average_rating": 5.0,
            "rating_number": 1,
            "rating_variance": 0.0,
            "affordability_score": -68.47000000000003,
            "price_to_income_ratio": 0.2659626065868732,
            "residual_utility_score": 4.62909435857374,
            "savings_to_price_ratio": 15.64050826519207,
            "net_worth_indicator": 4.159790346554018,
            "credit_risk_indicator": 0.4909090909090909,
            "value_density": 0.9724589747388704,
            "review_confidence": 0.0641942307825133,
            "rating_polarization": 0.0,
            "quality_risk_score": 0.0,
            "cold_start_flag": 1.0,
            "price_category_rank": 0.0088475298672441,
            "category_rating_deviation": 1.2744255003706453,
            "verified_purchase_ratio": 0.0,
            "helpful_concentration": 0.0,
            "sentiment_spread": 1.0,
            "review_depth_score": 0.96,
            "reviewer_diversity": 1.0,
            "extreme_rating_ratio": 1.0,
            "downgraded": 0,
        }
        X = transform_input(example)
    else:
        try:
            data = request.get_json() or {}
            X = transform_input(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    try:
        pred_idx = int(model.predict(X)[0])
        pred_label = LABELS[pred_idx]
        probs = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            probs = {LABELS[i]: round(float(proba[i]), 4) for i in range(len(LABELS))}
        out = {"final_recommendation": pred_label}
        if probs:
            out["probabilities"] = probs
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/upload")
def upload_file():
    try:
        from google.cloud import storage
        storage_client = storage.Client()
        bucket_name = os.environ.get("BUCKET_NAME")
        if not bucket_name:
            return "BUCKET_NAME environment variable is not set.", 500
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob("hello.txt")
        blob.upload_from_string("Hello, Cloud Storage!")
        return f"File uploaded to bucket: {bucket_name}"
    except Exception as e:
        return str(e), 500


@app.route("/query")
def query_bigquery():
    try:
        from google.cloud import bigquery
        client = bigquery.Client()
        dataset = os.environ.get("BQ_DATASET")
        table = os.environ.get("BQ_TABLE")
        if not dataset or not table:
            return (
                "BQ_DATASET and BQ_TABLE environment variables must be set to "
                "your BigQuery dataset and table that contain training_scenarios-style rows.",
                500,
            )

        # Example: aggregate our own labels from BigQuery table loaded from training_scenarios.csv
        # Table must have a column named final_recommendation (GREEN / YELLOW / RED).
        query = f"""
            SELECT
              final_recommendation,
              COUNT(*) AS num_scenarios
            FROM `{dataset}.{table}`
            GROUP BY final_recommendation
            ORDER BY num_scenarios DESC
            LIMIT 10
        """
        query_job = client.query(query)
        results = query_job.result()
        rows = [f"{row.final_recommendation}: {row.num_scenarios}" for row in results]
        return "Top recommendations in your data (label: count): " + ", ".join(rows)
    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
