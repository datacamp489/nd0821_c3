import joblib
from fastapi import FastAPI

import pandas as pd
import os
import uvicorn

from rest.model import CensusData
from training.ml.data import process_data
from training.ml.model import inference
from training.train_model import MODEL_PATH, ENCODER_PATH, BINARIZER_PATH, CAT_FEATURES

app = FastAPI()
model_items = {}


def load_items():
    model_items['model'] = joblib.load(MODEL_PATH)
    model_items['encoder'] = joblib.load(ENCODER_PATH)
    model_items['lb'] = joblib.load(BINARIZER_PATH)


@app.on_event("startup")
def startup_event():
    """Load model, encoder and lb only on startup"""
    load_items()


@app.get("/")
def greet():
    """Greets everyone

    Returns:
        str: Greeting
    """
    return "Hello everyone!"


@app.post("/predict", response_model=str, summary="Predicts the income class based on census data")
def predict(data: CensusData):
    """Predicts income class >50K or <=50K for given census data  """
    if not model_items:
        load_items()
    data = pd.DataFrame(data.dict(), index=[0])
    column_rename = {col: col.replace("_", "-") for col in data.columns}
    data = data.rename(columns=column_rename)
    X, _, _, _ = process_data(data, categorical_features=CAT_FEATURES, training=False,
                              encoder=model_items['encoder'], lb=model_items['lb'])
    pred = inference(model_items['model'], X)
    pred = model_items['lb'].inverse_transform(pred)
    return pred[0]


def main():
    uvicorn.run(app=app)


if __name__ == "__main__":
    main()
