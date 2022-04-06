"""Pytest fixtures"""
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from training.ml.data import process_data
from training.ml.model import train_model
from training.train_model import CAT_FEATURES, DATA_PATH


@pytest.fixture
def train_data(path=DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    train, _ = train_test_split(df, test_size=0.20)
    return train.sample(frac=0.2, random_state=42)

@pytest.fixture
def processed_train_data(train_data):
    X, y, _, _ = process_data(train_data, categorical_features=CAT_FEATURES, label="salary")
    return (X, y)

@pytest.fixture
def model(processed_train_data):
    X, y = processed_train_data
    return train_model(X, y)
