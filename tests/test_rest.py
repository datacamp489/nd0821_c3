from fastapi.testclient import TestClient

from rest.app import app
from rest.model import CensusData
from training.train_model import DATA_PATH

client = TestClient(app)


def test_greeting():
    """Successful request for greeting"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello everyone!"


def test_prediction_success(train_data):
    """Successful request with random sample from training data"""
    train_data = train_data.drop('salary', axis=1)
    column_rename = {col: col.replace("-", "_") for col in train_data.columns}
    sample = train_data.rename(columns=column_rename).sample(1).iloc[0].to_dict()
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.json() in ("<=50K", ">50K")

def test_prediction_fail():
    """ Failed request with empty request"""
    response = client.post("/predict", json={})
    assert response.status_code == 422

