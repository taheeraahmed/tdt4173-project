from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from catboost import CatBoostRegressor

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from utils.read_data import get_test_data, get_train_targets, load_data, prepare_submission
from utils.data_pipeline import ColumnDropper, FeatureAdder
from utils.generate_run_name import generate_run_name

import numpy as np
import time
import mlflow
import logging

def bayes_search_catboost(model_name="bayes-search-catboost"):
    logger = logging.getLogger()

    logger.info('Processing data')
    
    data_a, data_b, data_c = load_data(mean=True, remove_out=True, roll_avg=True)

    X_train_a, y_train_a = get_train_targets(data_a)
    X_train_b, y_train_b = get_train_targets(data_b)
    X_train_c, y_train_c = get_train_targets(data_c)

    # Get test data
    X_test_a, X_test_b, X_test_c = get_test_data(mean=True, roll_avg=True)

    drop_cols = ['time', 'elevation:m'] # this second line is columns with feature importance == 0

     # Define the data processing pipeline
    data_process_pipeline = Pipeline([
        ('add_features', FeatureAdder()),
        ('drop_cols', ColumnDropper(drop_cols=drop_cols)),
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
    ])

    logger.info('Done processing data')

    locA_pipeline = Pipeline([
        ('data_process', data_process_pipeline),
        ('cat_boost', CatBoostRegressor(silent=True, random_state=2, loss_function='MAE'))
    ])

    locB_pipeline = Pipeline([
        ('data_process', data_process_pipeline),
        ('cat_boost', CatBoostRegressor(silent=True, random_state=3, loss_function='MAE'))
    ])

    locC_pipeline = Pipeline([
        ('data_process', data_process_pipeline),
        ('cat_boost', CatBoostRegressor(silent=True, random_state=2, loss_function='MAE'))
    ])
    
    # Define the search space for BayesSearchCV
    search_space_catboost = {
        'cat_boost__iterations': Integer(100, 300),
        'cat_boost__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'cat_boost__depth': Integer(2, 7),
        'cat_boost__l2_leaf_reg': Real(5, 10, prior='log-uniform'),
        'cat_boost__border_count': Integer(200, 255),
        'cat_boost__random_strength': Real(1e-9, 5, prior='log-uniform'),
        'cat_boost__min_data_in_leaf': Integer(1, 10),
        'cat_boost__grow_policy': Categorical(['SymmetricTree', 'Depthwise', 'Lossguide']),
        'cat_boost__bootstrap_type': Categorical(['Bayesian', 'Bernoulli', 'MVS']),
    }

    bayes_search_a = BayesSearchCV(locA_pipeline, search_space_catboost, cv=5, scoring='neg_mean_squared_error')
    bayes_search_b = BayesSearchCV(locB_pipeline, search_space_catboost, cv=5, scoring='neg_mean_squared_error')
    bayes_search_c = BayesSearchCV(locC_pipeline, search_space_catboost, cv=5, scoring='neg_mean_squared_error')

    # Assuming generate_run_name, fetch_logged_data, write_to_file, get_input_data, prepare_submission are defined elsewhere
    run_name = generate_run_name()
    with mlflow.start_run(run_name=f'{run_name}_A') as run:
        logger.info("Fit for location A")
        bayes_search_a.fit(X_train_a, y_train_a)
        logger.info(f"{run_name}-A: {bayes_search_a.best_params_}")

    with mlflow.start_run(run_name=f'{run_name}_B') as run:
        logger.info("Fit for location B")
        bayes_search_b.fit(X_train_b, y_train_b)
        logger.info(f"B: {bayes_search_b.best_params_}")

    with mlflow.start_run(run_name='f{run_name}_C') as run:
        logger.info("Fit for location C")
        bayes_search_c.fit(X_train_c, y_train_c)
        logger.info(f"C: {bayes_search_c.best_params_}")

    logger.info("Run pipeline for location A")
    pred_a = bayes_search_a.predict(X_test_a.drop(columns=["id", "prediction", "location"]))
    logger.info("Run pipeline for location B")
    pred_b = bayes_search_b.predict(X_test_b.drop(columns=["id", "prediction", "location"]))
    logger.info("Run pipeline for location C")
    pred_c = bayes_search_c.predict(X_test_c.drop(columns=["id", "prediction", "location"]))

    prepare_submission(X_test_a, X_test_b, X_test_c, pred_a, pred_b, pred_c, run_name)