from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from utils.read_data import get_test_data, get_train_targets, load_data, prepare_submission
from utils.data_pipeline import ColumnDropper, FeatureAdder
from utils.generate_run_name import generate_run_name

import numpy as np
import time
import mlflow
import logging

def bayes_search_xgboost(model_name="bayes-search-xgboost"):
    logger = logging.getLogger()

    logger.info('Processing data')

    drop_cols = ['time', 'elevation:m', 'fresh_snow_1h:cm', 'ceiling_height_agl:m', 'snow_density:kgm3', 
             'wind_speed_w_1000hPa:ms', 'snow_drift:idx', 'fresh_snow_3h:cm', 'is_in_shadow:idx', 'dew_or_rime:idx', 'fresh_snow_6h:cm', 'prob_rime:p'] # this second line is columns with feature importance == 0

    data_a, data_b, data_c = load_data(mean=True, remove_out=True, roll_avg=True, cust_feat=True, drop_cols=drop_cols, norm=True)

    X_train_a, y_train_a = get_train_targets(data_a)
    X_train_b, y_train_b = get_train_targets(data_b)
    X_train_c, y_train_c = get_train_targets(data_c)

    X_test_a, X_test_b, X_test_c = get_test_data(mean=True, roll_avg=True, cust_feat=True, norm=True, drop_cols=True)

    logger.info('Done processing data')

    # Define the data processing pipeline
    data_process_pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
    ])

    # Define the search space for BayesSearchCV
    search_space_catboost = {
        'iterations': Integer(100, 300),
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'depth': Integer(2, 7),
        'l2_leaf_reg': Real(5, 10, prior='log-uniform'),
        'border_count': Integer(200, 255),
        'random_strength': Real(1e-9, 5, prior='log-uniform'),
        'min_data_in_leaf': Integer(1, 10),
        'grow_policy': Categorical(['SymmetricTree', 'Depthwise', 'Lossguide']),
        'bootstrap_type': Categorical(['Bayesian', 'Bernoulli', 'MVS']),
    }

    def run_bayes_search(X_train, y_train, pipeline, location_name):
        logger.info(f"Fit for location {location_name}")
        bayes_search = BayesSearchCV(pipeline, search_space_catboost, cv=5, scoring='neg_mean_squared_error')
        bayes_search.fit(X_train, y_train)
        logger.info(f"{model_name}-{location_name}: {bayes_search.best_params_}")
        return bayes_search

    run_name = generate_run_name()
    
    with mlflow.start_run(run_name=f'{run_name}_A') as run:
        bayes_search_a = run_bayes_search(X_train_a, y_train_a, XGBRegressor(silent=True, random_state=42, objective='reg:squarederror'), 'A')

    with mlflow.start_run(run_name=f'{run_name}_B') as run:
        bayes_search_b = run_bayes_search(X_train_b, y_train_b, XGBRegressor(silent=True, random_state=42, objective='reg:squarederror'), 'B')

    with mlflow.start_run(run_name=f'{run_name}_C') as run:
        bayes_search_c = run_bayes_search(X_train_c, y_train_c, Pipeline([('data_process', data_process_pipeline), ('cat_boost', XGBRegressor(silent=True, random_state=42, objective='reg:squarederror'))]), 'C')

    logger.info("Run pipeline for location A")
    pred_a = bayes_search_a.predict(X_test_a.drop(columns=["id", "prediction", "location"]))
    logger.info("Run pipeline for location B")
    pred_b = bayes_search_b.predict(X_test_b.drop(columns=["id", "prediction", "location"]))
    logger.info("Run pipeline for location C")
    pred_c = bayes_search_c.predict(X_test_c.drop(columns=["id", "prediction", "location"]))

    prepare_submission(X_test_a, X_test_b, X_test_c, pred_a, pred_b, pred_c, run_name)

def bayes_search_catboost(model_name="bayes-search-catboost"):
    logger = logging.getLogger()

    logger.info('Processing data')

    drop_cols = ['time', 'elevation:m'] # this second line is columns with feature importance == 0
    
    data_a, data_b, data_c = load_data(mean=True, remove_out=True, roll_avg=True, cust_feat=True, drop_cols=drop_cols)

    X_train_a, y_train_a = get_train_targets(data_a)
    X_train_b, y_train_b = get_train_targets(data_b)
    X_train_c, y_train_c = get_train_targets(data_c)

    # Get test data
    X_test_a, X_test_b, X_test_c = get_test_data(mean=True, roll_avg=True, cust_feat=True, drop_cols=drop_cols)


    logger.info('Done processing data')


     # Define the data processing pipeline
    data_process_pipeline = Pipeline([
        ('drop_cols', ColumnDropper(drop_cols=drop_cols)),
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
    ])


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