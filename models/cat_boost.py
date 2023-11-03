from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from utils.generate_run_name import generate_run_name
from utils.log_model import fetch_logged_data, write_to_file
from utils.evaluate import prepare_submission, get_input_data, submission_to_csv

import mlflow
import time
import logging

def catboost_reg(num, cat, X_train, y_train, model_name="catboost-regression"):
    """
    Train a CatBoost regression model using given training data and log the results using MLflow.

    Parameters:
    - num (list): List of numerical feature names.
    - cat (list): List of categorical feature names.
    - X_train (DataFrame): Training data features.
    - y_train (Series): Training data target values.

    Returns:
    None. The function logs the training results using MLflow and writes the logged data to a file.
    """
    logger = logging.getLogger()
    logger.info(f'Training {model_name}')

    start_time = time.time()  # <- Start the timer

    # CatBoost can handle categorical variables natively
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num)],
        remainder='passthrough')  # <- This will pass through categorical features without changes

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', CatBoostRegressor(cat_features=cat))])  # <- Specify categorical features here

    run_name = generate_run_name()

    with mlflow.start_run(run_name=run_name) as run:
        model.fit(X_train, y_train)
        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
    
    logged_data = {
        'name': model_name,
        'start_time': start_time,
        'run_name': run_name,
        'params': params,
        'metrics': metrics,
        'tags': tags, 
        'artifacts': artifacts,
    }
    write_to_file(logged_data)

    X_test = get_input_data()
    pred = model.predict(X_test)
    submission = prepare_submission(X_test, pred, run_name)
    submission_to_csv(submission, run_name)
