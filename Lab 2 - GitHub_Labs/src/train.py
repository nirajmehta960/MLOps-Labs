import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.datasets import load_iris
from google.cloud import storage
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Constants
MODEL_FILE = "model.joblib"
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
VERSION_FILE_NAME = "model_version.txt"

def load_data():
    """Loads the Iris dataset."""
    print("Loading Iris data...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    return X, y

def preprocess_data(X, y):
    """Splits the data into training and testing sets."""
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Trains the Random Forest model."""
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints metrics."""
    print("Evaluating model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    return accuracy

def get_model_version(bucket_name, version_file_name):
    """Retrieves the current model version from GCS."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(version_file_name)
        if blob.exists():
            version = int(blob.download_as_text())
        else:
            version = 0
        return version
    except Exception as e:
        print(f"Error getting model version: {e}")
        return 0

def update_model_version(bucket_name, version_file_name, new_version):
    """Updates the model version in GCS."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(version_file_name)
        blob.upload_from_string(str(new_version))
        print(f"Model version updated to {new_version}")
    except Exception as e:
        print(f"Failed to update model version: {e}")

def save_model_to_gcs(model, bucket_name, version):
    """Saves the model with version and timestamp to GCS."""
    print("Saving model to GCS...")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    blob_name = f"trained_models/model_v{version}_{timestamp}.joblib"
    
    # Save locally first
    joblib.dump(model, MODEL_FILE)
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(MODEL_FILE)
        print(f"Model saved to gs://{bucket_name}/{blob_name}")
    except Exception as e:
        print(f"Failed to upload to GCS: {e}")

def main():
    print("Starting pipeline...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    if GCS_BUCKET_NAME:
        current_version = get_model_version(GCS_BUCKET_NAME, VERSION_FILE_NAME)
        new_version = current_version + 1
        save_model_to_gcs(model, GCS_BUCKET_NAME, new_version)
        update_model_version(GCS_BUCKET_NAME, VERSION_FILE_NAME, new_version)
    else:
        print("GCS_BUCKET_NAME not set, saving locally only.")
        joblib.dump(model, MODEL_FILE)

if __name__ == "__main__":
    main()
