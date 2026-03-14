"""
Train a classifier on SavVio training_scenarios data (GREEN/YELLOW/RED recommendation).
Reads data/training_scenarios.csv, applies imputation + encoding + scaling, trains
a Random Forest classifier, and saves model + preprocessing artifacts for inference.
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# Paths (data dir is at repo root for this lab)
DATA_DIR = os.environ.get("DATA_DIR", "data")
DATA_PATH = os.path.join(DATA_DIR, "training_scenarios.csv")
LABEL_COL = "final_recommendation"
LABELS = ["GREEN", "YELLOW", "RED"]

# Columns to drop (IDs, metadata, label)
DROP_COLUMNS = ["user_id", "product_id", "price_tier", "session_id", LABEL_COL]

# Categorical columns to ordinal-encode (must exist in CSV)
CATEGORICAL_FEATURES = ["employment_status", "region"]


def load_and_prepare(data_path: str):
    df = pd.read_csv(data_path)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found. Columns: {list(df.columns)}")

    y_raw = df[LABEL_COL].copy()
    label_map = {l: i for i, l in enumerate(LABELS)}
    y = y_raw.map(label_map)
    if y.isna().any():
        raise ValueError(f"Unexpected label values. Allowed: {LABELS}")

    drop = [c for c in DROP_COLUMNS if c in df.columns]
    X_df = df.drop(columns=drop, errors="ignore")

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_df.columns]
    num_cols = [c for c in X_df.columns if c not in cat_cols]

    for c in num_cols:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce").fillna(X_df[c].median())
    for c in cat_cols:
        X_df[c] = X_df[c].fillna("Unknown").astype(str)

    return X_df, y, num_cols, cat_cols


def main():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"Training data not found at {DATA_PATH}. "
            "Place training_scenarios.csv in the data/ directory (from SavVio model_pipeline)."
        )

    X_df, y, num_cols, cat_cols = load_and_prepare(DATA_PATH)
    feature_order = num_cols + cat_cols

    X_train, X_rest, y_train, y_rest = train_test_split(
        X_df, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=42, stratify=y_rest
    )

    scaler = StandardScaler()
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_train_cat = encoder.fit_transform(X_train[cat_cols]) if cat_cols else np.empty((len(X_train), 0))
    X_train_final = np.hstack([X_train_num, X_train_cat])

    def transform(X_df):
        n = scaler.transform(X_df[num_cols])
        c = encoder.transform(X_df[cat_cols]) if cat_cols else np.empty((len(X_df), 0))
        return np.hstack([n, c])

    X_val_final = transform(X_val)
    X_test_final = transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_final, y_train)

    for name, X_f, y_f in [("Val", X_val_final, y_val), ("Test", X_test_final, y_test)]:
        pred = model.predict(X_f)
        f1 = f1_score(y_f, pred, average="weighted")
        print(f"{name} set — weighted F1: {f1:.4f}")
    print("\nTest set classification report:")
    print(classification_report(y_test, model.predict(X_test_final), target_names=LABELS))

    save_dir = "output" if os.path.isdir("output") else (os.path.dirname(os.path.abspath(__file__)) or ".")
    if os.path.isdir("/app"):
        save_dir = "/app"

    joblib.dump(model, os.path.join(save_dir, "recommendation_model.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(save_dir, "encoder.pkl"))
    joblib.dump(
        {"feature_order": feature_order, "num_cols": num_cols, "cat_cols": cat_cols, "labels": LABELS},
        os.path.join(save_dir, "feature_config.joblib"),
    )
    print(f"Saved model and preprocessing to {save_dir}")


if __name__ == "__main__":
    main()
