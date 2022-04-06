"""unit tests for model training"""
from sklearn.utils.validation import check_is_fitted

from training.ml.data import process_data
from training.ml.model import compute_model_metrics, inference
from training.train_model import CAT_FEATURES


def test_processed_data_length(train_data):
    """Test split

    Args:
        train_data (pd.DataFrame): input census data
    """
    X, y, _, _ = process_data(train_data, categorical_features=CAT_FEATURES, label="salary")
    assert X.shape[0] == y.shape[0], "Unequal length of X and y in processed dataset"
    assert y.shape[0] == train_data.shape[0], "Unequal length of y and input dataset"

def test_processed_data_cat_encoding(train_data):
    """Test one-hot-encoding

    Args:
        train_data (pd.DataFrame): input census data
    """
    X, _, _, _ = process_data(train_data, categorical_features=CAT_FEATURES, label="salary")
    # length of input minus label column
    expected_columns = train_data.shape[1] - 1
    for cat in CAT_FEATURES:
        # n-1 additional columns for the values of each cat column
        expected_columns += train_data[cat].nunique()-1 
    assert expected_columns == X.shape[1], "Missing columns for one hot encoding"

def test_model_fitted(model):
    """Test model fitting

    Args:
        model (sklearn.BaseEstimator): fitted sklearn model
    """    
    check_is_fitted(model), "train_model returned not-fitted estimator"

def test_inference_shape(processed_train_data, model):
    """Test inference function

    Args:
        processed_train_data (tuple): preprocessed train data, tuple of X and y
        model (sklearn.BaseEstimator): fitted sklearn model
    """    
    X, y = processed_train_data
    preds = inference(model, X)
    assert y.shape == preds.shape, "inference function returned wrong shape"

def test_metrics(processed_train_data, model):
    """Test inference function

    Args:
        processed_train_data (tuple): preprocessed train data, tuple of X and y
        model (sklearn.BaseEstimator): fitted sklearn model
    """
    X, y = processed_train_data
    preds = inference(model, X)
    metrics = compute_model_metrics(y, preds)
    for metric in metrics:
        assert isinstance(metric, float), f"metric has the wrong type, should be float but is {type(metric)}"
        assert metric >= 0., "metric is smaller than 0."
        assert metric <= 1., "metric is greater than 1."
