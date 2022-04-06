''' Script to train machine learning model. '''

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from training.ml.data import process_data
from training.ml.model import compute_model_metrics, inference, save_model, train_model

ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data/census_cleaned.csv"
MODEL_PATH = ROOT_PATH / "model/classifier.mdl"
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

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    # Process the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Train and save a model.
    logger.info("Training model")
    model = train_model(X_train, y_train)
    save_model(model, MODEL_PATH)
    logger.info(f"Trained model saved to {MODEL_PATH}.")

    # Calculate train metrics
    logger.info("Calculating train metrics for trained model.")
    preds = inference(model, X_train)
    precision, recall, fbeta = compute_model_metrics(y_train, preds)
    logger.info(f"Model metrics: \n* precision: {precision}\n* recall: {recall}\n* fbeta: {fbeta}")

    # Calculate test metrics
    logger.info("Calculating test metrics for trained model.")
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    logger.info(f"Model metrics: \n* precision: {precision}\n* recall: {recall}\n* fbeta: {fbeta}")
    logger.info("Finished.")


if __name__ == "__main__":
    main()
