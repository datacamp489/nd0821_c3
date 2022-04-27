from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegressionCV

# Optional: implement hyperparameter tuning.
RANDOM_STATE = 42


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    classifiers = [LogisticRegressionCV(cv=5, random_state=RANDOM_STATE, max_iter=1e5),
                   RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=20, max_depth=10)]
    best_model_score = 0.
    best_model = None
    for clf in classifiers:
        model = clf.fit(X_train, y_train)
        preds = inference(model, X_train)
        _, _, f_beta= compute_model_metrics(y_train, preds)
        if f_beta > best_model_score:
            best_model_score = f_beta
            best_model = model
    return best_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)




