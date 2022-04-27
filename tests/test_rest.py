import json
from fastapi.testclient import TestClient
from rest.app import app
from rest.model import CensusData
import pandas as pd
from training.train_model import DATA_PATH
client = TestClient(app)


def test_greeting():
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == '"Hello everyone!"'


def test_prediction_success(train_data):
    column_rename = {"hours-per-week": "hours_per_week", "education-num": "education_num", "marital-status": "marital_status",
                     "native-country": "native_country", "capital-gain": "capital_gain", "capital-loss": "capital_loss"}
    sample = train_data.rename(columns=column_rename).sample(1).iloc[0].to_dict()
    print(sample)
    x = CensusData(**sample)
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.text == '"Hello everyone!"'
