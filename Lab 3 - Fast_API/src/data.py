"""
Data loading and splitting utilities for the financial condition pipeline.

Data is expected at data/financial_data.csv. Feature engineering is in src.features.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


def load_data():
    """
    Load the raw financial dataset from data/financial_data.csv.
    Returns a DataFrame; target and feature engineering are done in src.features.
    """
    return pd.read_csv(DATA_DIR / "financial_data.csv")


def split_data(X, y, test_size: float = 0.3, random_state: int = 12):
    """
    Split features (X) and target (y) into train and test sets.
    Default: 70% train, 30% test, with fixed random_state for reproducibility.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

