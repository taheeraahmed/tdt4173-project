
from sklearn.metrics import mean_squared_error
from data_preprocess import get_input_data
from sklearn.metrics import mean_squared_error
import numpy as np
import os
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

def prepare_submission(X_test: pd.DataFrame, predictions, run_name) -> pd.DataFrame:
    """Parses predictions and test-data to get a submission-ready DataFrame.

    Args:
        X_test (pd.DataFrame): test data / model input
        predictions: predictions / model output

    Returns:
        pd.DataFrame: DataFrame ready for submission on Kaggle
    """
    submission = X_test.reset_index()  # Reset the index to use it as 'id'
    submission = submission.rename(columns={'index': 'id'})  # Rename the index to 'id'
    submission['prediction'] = predictions

    submission = submission['id', 'prediction']

    # Specify the directory and filename for the submission
    submission_directory = 'submissions'  # Change to the desired directory path
    submission_filename = run_name + '.csv'

    # Check if the directory exists; if not, create it
    if not os.path.exists(submission_directory):
        os.makedirs(submission_directory)

    # Save the submission CSV in the specified directory
    submission.to_csv(os.path.join(submission_directory, submission_filename), index=False)