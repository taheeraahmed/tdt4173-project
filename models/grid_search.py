from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from utils.data_preprocess_location import get_test_data, get_train_targets, load_data, prepare_submission
from utils.data_preprocess import ColumnDropper
from utils.pipeline import run_pipeline_and_log
from utils.generate_run_name import generate_run_name
from utils.log_model import fetch_logged_data, write_to_file
from utils.evaluate import get_input_data
from catboost import CatBoostRegressor

import numpy as np
import time
import mlflow
import logging 


def grid_search_rf(num, cat, X_train, y_train, model_name="grid-search-rf"):
    """
    Train a random forest model using given training data and log the results using MLflow.

    Parameters:
    - num (list): List of numerical feature names.
    - cat (list): List of categorical feature names.
    - X_train (DataFrame): Training data features.
    - y_train (Series): Training data target values.

    Returns:
    None. The function logs the training results using MLflow and writes the logged data to a file.
    """
    logger = logging.getLogger()
    logger.info(model_name)

    start_time = time.time()  

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num)])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())])

    # Define the parameter grid for grid search
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],  # Example values, adjust as needed
        'regressor__max_depth': [None, 10, 20],    # Example values, adjust as needed
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    
    run_name = generate_run_name()
    with mlflow.start_run(run_name=run_name) as run:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_        
        mlflow.log_params(grid_search.best_params_)
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
    pred = best_model.predict(X_test)
    prepare_submission(X_test, pred, run_name)

def grid_search_gb(num, cat, X_train, y_train, model_name="grid-search-gb"):
    logger = logging.getLogger()
    logger.info(model_name)

    start_time = time.time()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num),
            ('cat', categorical_transformer, cat)])  # Assuming you want to include categorical transformations

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor())])

    # Define the parameter grid for grid search
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],  # Example values, adjust as needed
        'regressor__learning_rate': [0.01, 0.1, 0.2],  # Example values, adjust as needed
        'regressor__max_depth': [3, 4, 5],  # Example values, adjust as needed
        # Add other parameters here as needed
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)

    run_name = generate_run_name()
    with mlflow.start_run(run_name=run_name) as run:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        # Log other metrics or artifacts as needed

    elapsed_time = time.time() - start_time

    params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)

    logged_data = {
        'name': model_name,
        'start_time': start_time,
        'run_name': run_name,
        'params': params,
        'metrics': metrics,
        'tags': tags,
        'artifacts': artifacts,
        'elapsed_time': elapsed_time
    }

    write_to_file(logged_data)

    X_test = get_input_data()
    pred = best_model.predict(X_test)
    prepare_submission(X_test, pred, run_name)

def grid_search_catboost(model_name="grid-search-catboost"):
    data_a, data_b, data_c = load_data()

    X_test_a, X_test_b, X_test_c = get_test_data()
    X_train_a, y_train_a = get_train_targets(data_a)
    X_train_b, y_train_b = get_train_targets(data_b)
    X_train_c, y_train_c = get_train_targets(data_c)

    logger = logging.getLogger()
    logger.info(model_name)

    start_time = time.time()

    drop_cols = ['time', 'date_calc', 'elevation:m', 'fresh_snow_1h:cm', 'wind_speed_u_10m:ms', 
            'wind_speed_u_10m:ms', 'wind_speed_v_10m:ms', 'wind_speed_w_1000hPa:ms', 'prob_rime:p',
            'fresh_snow_12h:cm','fresh_snow_24h:cm', 'fresh_snow_6h:cm', 'super_cooled_liquid_water:kgm2']


    data_process_pipeline = Pipeline([
        ('drop_cols', ColumnDropper(drop_cols=drop_cols)),
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
    ])

    locA_pipeline = Pipeline([
        ('data_process', data_process_pipeline),
        ('cat_boost', CatBoostRegressor(silent=True))
    ])

    locB_pipeline = Pipeline([
        ('data_process', data_process_pipeline),
        ('random_forest', RandomForestRegressor())
    ])

    locC_pipeline = Pipeline([
        ('data_process', data_process_pipeline),
        ('cat_boost', CatBoostRegressor(silent=True))
    ])
    # Define the parameter grid for grid search
    param_grid_catboost = {
        'cat_boost__iterations': [100, 200, 300, 800],  # Example values, adjust as needed
        'cat_boost__learning_rate': [0.01, 0.05, 0.1, 0.2],  # Example values, adjust as needed
        'cat_boost__depth': [3, 4, 5, 8],  # Example values, adjust as needed
        # Add other parameters here as needed
    }

    # Define the parameter grid
    param_grid_rf = {
        'random_forest__n_estimators': [100, 200, 300],
        'random_forest__max_features': ['auto', 'sqrt', 'log2'],
        'random_forest__max_depth': [None, 10, 20, 30],
        'random_forest__min_samples_split': [2, 5, 10],
        'random_forest__min_samples_leaf': [1, 2, 4],
        'random_forest__bootstrap': [True, False]
    }

    grid_search_a = GridSearchCV(locA_pipeline, param_grid_catboost, cv=5, scoring='neg_mean_squared_error')
    grid_search_b = GridSearchCV(locB_pipeline, param_grid_rf, cv=5, scoring='neg_mean_squared_error')
    grid_search_c = GridSearchCV(locC_pipeline, param_grid_catboost, cv=5, scoring='neg_mean_squared_error')

    # Assuming generate_run_name, fetch_logged_data, write_to_file, get_input_data, prepare_submission are defined elsewhere
    run_name = generate_run_name()
    with mlflow.start_run(run_name='f{run_name}_A') as run:
        grid_search_a.fit(X_train_a, y_train_a)
        mlflow.log_params(grid_search_a.best_params_)
        mlflow.log_metric("best_cv_score", grid_search_a.best_score_)

    with mlflow.start_run(run_name='f{run_name}_B') as run:
        grid_search_b.fit(X_train_b, y_train_b)
        mlflow.log_params(grid_search_b.best_params_)
        mlflow.log_metric("best_cv_score", grid_search_b.best_score_)

    with mlflow.start_run(run_name='f{run_name}_C') as run:
        grid_search_c.fit(X_train_c, y_train_c)
        mlflow.log_params(grid_search_c.best_params_)
        mlflow.log_metric("best_cv_score", grid_search_c.best_score_)

    elapsed_time = time.time() - start_time

    params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)

    logged_data = {
        'name': model_name,
        'start_time': start_time,
        'run_name': run_name,
        'params': params,
        'metrics': metrics,
        'tags': tags,
        'artifacts': artifacts,
        'elapsed_time': elapsed_time
    }

    write_to_file(logged_data)

    logger.info("Run pipeline for location A")
    pred_a = run_pipeline_and_log(locA_pipeline, X_train_a, y_train_a, X_test_a, "A-" + run_name)
    logger.info("Run pipeline for location B")
    pred_b = run_pipeline_and_log(locB_pipeline, X_train_b, y_train_b, X_test_a, "B-" + run_name)
    logger.info("Run pipeline for location C")
    pred_c = run_pipeline_and_log(locC_pipeline, X_train_c, y_train_c, X_test_c, "C-" + run_name)

    prepare_submission(X_test_a, X_test_b, X_test_c, pred_a, pred_b, pred_c, run_name)

