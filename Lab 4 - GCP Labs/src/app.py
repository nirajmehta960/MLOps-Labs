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
    if request.method == "GET":
        return jsonify({
            "message": "POST a JSON body with scenario features to get final_recommendation (GREEN/YELLOW/RED).",
            "feature_columns": FEATURE_ORDER,
            "label_values": LABELS,
        })
    if request.method != "POST":
        return "Method not allowed", 405
    try:
        data = request.get_json() or {}
        X = transform_input(data)
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
        query = """
            SELECT name, SUM(number) AS total
            FROM `bigquery-public-data.usa_names.usa_1910_current`
            WHERE state = 'TX'
            GROUP BY name
            ORDER BY total DESC
            LIMIT 10
        """
        query_job = client.query(query)
        results = query_job.result()
        names = [row.name for row in results]
        return f"Top names in Texas: {', '.join(names)}"
    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
