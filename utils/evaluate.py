
from sklearn.metrics import mean_squared_error
from data_preprocess import get_input_data
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import logging 
def training_mse(targets, predictions):
    logger = logging.getLogger()
    logger.info("MSE score: ", str(mean_squared_error(targets, predictions)))

def display_cross_val_scores(scores):
    logger = logging.getLogger()
    mse_scores = np.sqrt(-scores)
    logger.info("MSE scores:", str(mse_scores))
    logger.info("Mean MSE:", str(mse_scores.mean()))
    logger.info("Std. dev:", str(mse_scores.std()))
    
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def prepare_submission(X_test: pd.DataFrame, predictions) -> pd.DataFrame:
    """Parses predicitons and test-data to get submission-ready df.

    Args:
        X_test (pd.DataFrame): test data / model input
        predictions: predictions / model output

    Returns:
        pd.DataFrame: df ready for submission on kaggle
    """

    submission = X_test.copy()
    submission["prediction"] = predictions
    submission = submission[["id", "prediction"]]

    return submission

def create_submission_csv(model, features=[]):
    X_test = get_input_data()
    predictions = model.predict(X_test[features].values)

    submission = prepare_submission(X_test, predictions)
    submission.to_csv('submissions/mlpgr_regressor.csv', index=False)