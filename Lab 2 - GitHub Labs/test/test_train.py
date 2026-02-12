import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import os
from src.train import preprocess_data, train_model, evaluate_model, main, get_model_version

@pytest.fixture
def mock_data():
    """Creates a mock Iris dataset."""
    X = pd.DataFrame({
        "sepal length (cm)": np.random.rand(100),
        "sepal width (cm)": np.random.rand(100),
        "petal length (cm)": np.random.rand(100),
        "petal width (cm)": np.random.rand(100)
    })
    y = pd.Series(np.random.randint(0, 3, 100))
    return X, y

def test_preprocess_data(mock_data):
    """Tests data splitting."""
    X, y = mock_data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

def test_train_model(mock_data):
    """Tests model training."""
    X, y = mock_data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)
    assert model is not None

def test_evaluate_model(mock_data):
    """Tests evaluation metrics."""
    X, y = mock_data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1

@patch("src.train.storage.Client")
def test_get_model_version(mock_client):
    """Tests retrieving model version."""
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    
    # Test existing version
    mock_blob.exists.return_value = True
    mock_blob.download_as_text.return_value = "5"
    version = get_model_version("bucket", "version.txt")
    assert version == 5

    # Test non-existing version
    mock_blob.exists.return_value = False
    version = get_model_version("bucket", "version.txt")
    assert version == 0

@patch("src.train.load_data")
@patch("src.train.update_model_version")
@patch("src.train.save_model_to_gcs")
@patch("src.train.get_model_version")
def test_main_with_gcs(mock_get_version, mock_save_gcs, mock_update_version, mock_load, mock_data):
    """Tests main pipeline with GCS enabled."""
    mock_load.return_value = mock_data
    mock_get_version.return_value = 1
    
    # Simulate GCS_BUCKET_NAME env var being set
    with patch.dict(os.environ, {"GCS_BUCKET_NAME": "test-bucket"}):
        # We need to reload the module or patch constants, but since constant is read at top level, 
        # simpler to just assume the logic inside main checks the constant.
        # However, constant is loaded at import time. 
        # Let's mock the constants/globals if possible or refactor main to read env var.
        # Refactoring src/train.py main to read env var inside function is safer for testing.
        # But for now, we'll patch the constant in the module if possible, or just rely on logic flow if we can.
        # Python's 'from src.train import ...' makes patching constants hard.
        # Let's just mock the 'if GCS_BUCKET_NAME:' check by patching the variable in the module?
        with patch("src.train.GCS_BUCKET_NAME", "test-bucket"):
             main()
             
    mock_get_version.assert_called_once()
    mock_save_gcs.assert_called_once()
    mock_update_version.assert_called_once_with("test-bucket", "model_version.txt", 2)
