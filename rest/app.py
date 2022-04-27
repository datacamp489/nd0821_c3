import joblib
from fastapi import FastAPI

import pandas as pd
import os
import uvicorn

from rest.model import CensusData
from training.ml.data import process_data
from training.ml.model import inference
from training.train_model import MODEL_PATH, ENCODER_PATH, BINARIZER_PATH, CAT_FEATURES

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()
model_items = {}


@app.on_event("startup")
def startup_event():
    """Load model, encoder and lb only on startup"""
    model_items['model'] = joblib.load(MODEL_PATH)
    model_items['encoder'] = joblib.load(ENCODER_PATH)
    model_items['lb'] = joblib.load(BINARIZER_PATH)


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
    data = pd.DataFrame(data.dict(), index=[0])
    column_rename = {"marital_status": "marital-status", "native_country": "native-country"}
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
