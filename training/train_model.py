''' Script to train machine learning model. '''

import logging
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from training.ml.data import process_data
from training.ml.model import compute_model_metrics, inference, train_model

ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data/census_cleaned.csv"
MODEL_PATH = ROOT_PATH / "model/classifier.mdl"
ENCODER_PATH = ROOT_PATH / "model/one_hot.enc"
BINARIZER_PATH = ROOT_PATH / "model/label_binarizer.enc"
SLICE_RESULTS_PATH = ROOT_PATH / "slice_out.txt"

CAT_FEATURES = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def main():
  # Add code to load in the data.
    logger.info("Loading data.")
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
    else:
        raise FileNotFoundError(f"{DATA_PATH} could not be found.")

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    logger.info("Performing train-test split.")
    train, test = train_test_split(data, test_size=0.20)

    logger.info("Preprocessing data.")
    # Process the train data and save the encoders
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    joblib.dump(encoder,ENCODER_PATH)
    logger.info(f"Encoder saved to {ENCODER_PATH}")
    joblib.dump(lb, BINARIZER_PATH)
    logger.info(f"Binarizer saved to {BINARIZER_PATH}")

    # Process the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Train and save a model.
    logger.info("Training model")
    model = train_model(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Trained model saved to {MODEL_PATH}.")

    # Calculate train metrics
    logger.info("Calculating train metrics for trained model.")
    preds = inference(model, X_train)
    precision, recall, fbeta = compute_model_metrics(y_train, preds)
    logger.info(f"Model metrics: \n* precision: {precision:.4f}\n* recall: {recall:.4f}\n* fbeta: {fbeta:.4f}")

    # Calculate test metrics
    logger.info("Calculating test metrics for trained model.")
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    logger.info(f"Model metrics: \n* precision: {precision:.4f}\n* recall: {recall:.4f}\n* fbeta: {fbeta:.4f}")
    

    # Caluclate slice performance for education
    SLICE_RESULTS_PATH.unlink(missing_ok=True)
    logger.info(f"Calculating slice performance values for catergorial data slices.\nTarget file: {SLICE_RESULTS_PATH}")
    with open(SLICE_RESULTS_PATH, 'a') as file:
        # header
        file.write(f"feature,value,precision,recall,fbeta\n")
        for slice_feature in CAT_FEATURES:
            feature = test[slice_feature].to_numpy()
            for val, precision, recall, fbeta in slice_performance(feature, y_test, preds):
                # sample for education slice
                if slice_feature == "education":
                    logger.info(f"Model metrics for education={val}: \n* precision: {precision:.4f}\n* recall: {recall:.4f}\n* fbeta: {fbeta:.4f}")
                file.write(f"{slice_feature},{val},{precision:.4f},{recall:.4f},{fbeta:.4f}\n")
    logger.info("Finished.")


if __name__ == "__main__":
    main()
