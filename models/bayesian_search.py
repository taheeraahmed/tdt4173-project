from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from catboost import CatBoostRegressor

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

from utils.data_preprocess_location import get_test_data, get_train_targets, load_data, prepare_submission
from utils.data_preprocess import ColumnDropper
from utils.generate_run_name import generate_run_name
from utils.log_model import fetch_logged_data, write_to_file
from utils.pipeline import run_pipeline_and_log

import numpy as np
import time
import mlflow
import logging

def bayes_search_catboost_stacked(drop_cols=True, model_name="bayes-search-stacked-catboost"):
    start_time = time.time()
    logger = logging.getLogger()

    logger.info('Preprocessing data')
    data_a, data_b, data_c = load_data()
    X_test_a, X_test_b, X_test_c = get_test_data()
    X_train_a, y_train_a = get_train_targets(data_a)
    X_train_b, y_train_b = get_train_targets(data_b)
    X_train_c, y_train_c = get_train_targets(data_c)
    logger.info('Done preprocessing data')

    if drop_cols: 
        model_name = model_name + '-drop-cols'
        logger.info(model_name)

        drop_cols_lst = ['time', 'date_calc', 'elevation:m', 'fresh_snow_1h:cm',  
             'wind_speed_u_10m:ms', 'wind_speed_v_10m:ms', 'wind_speed_w_1000hPa:ms', 'prob_rime:p',
             'fresh_snow_12h:cm','fresh_snow_24h:cm', 'fresh_snow_6h:cm', 'super_cooled_liquid_water:kgm2']
        
        data_process_pipeline = Pipeline([
            ('drop_cols', ColumnDropper(drop_cols=drop_cols_lst)),
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
        ])
        
    else: 
        model_name = model_name + '-all-features'
        logger.info(model_name)
        drop_cols_lst = ['time', 'date_calc']
        data_process_pipeline = Pipeline([
            ('drop_cols', ColumnDropper(drop_cols=drop_cols_lst)),
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
        ])

    base_models = [
        # ('cat_boost1', cb.CatBoostRegressor(random_state=1, silent=True)),
        # ('cat_boost2', cb.CatBoostRegressor(random_state=2, silent=True)),
        # ('cat_boost3', cb.CatBoostRegressor(random_state=3, silent=True)),
        # ('cat_boost4', cb.CatBoostRegressor(random_state=4, silent=True)),
        ('cat_boost4', CatBoostRegressor(random_state=42, silent=True, border_count=86, depth=9, iterations=384, l2_leaf_reg=2.1607264050691626, learning_rate=0.023800792606525824)),
        ('cat_boost3', CatBoostRegressor(random_state=42, silent=True, border_count=81, depth=8, iterations=704, l2_leaf_reg=9.448753109694545, learning_rate=0.01698158072074776)),
        ('cat_boost2', CatBoostRegressor(random_state=42, silent=True, border_count=81, depth=8, iterations=704, l2_leaf_reg=9.448753109694545, learning_rate=0.01698158072074776)),
        ('cat_boost1', CatBoostRegressor(random_state=42, silent=True))
    ]


    # Define meta-learner
    meta_learner = LinearRegression()

    # Create the stacking regressor
    stacked_model = StackingRegressor(estimators=base_models, final_estimator=meta_learner)

    #NOTE: can instead of using the sacked model just run a single model below:
    whole_model_pipeline = Pipeline([
        ('data_process', data_process_pipeline),
        ('stacked_model', stacked_model)
    ])

    # Define the search space for BayesSearchCV
    search_space_catboost = {
        'stacked_model__cat_boost1__iterations': Integer(100, 900),
        'stacked_model__cat_boost1__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'stacked_model__cat_boost1__depth': Integer(2, 9),
        'stacked_model__cat_boost2__iterations': Integer(100, 900),
        'stacked_model__cat_boost2__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'stacked_model__cat_boost2__depth': Integer(2, 9),
        'stacked_model__cat_boost3__iterations': Integer(100, 900),
        'stacked_model__cat_boost3__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'stacked_model__cat_boost3__depth': Integer(2, 9),
    }

    bayes_search_a = BayesSearchCV(whole_model_pipeline, search_space_catboost, cv=5, scoring='neg_mean_squared_error')
    bayes_search_b = BayesSearchCV(whole_model_pipeline, search_space_catboost, cv=5, scoring='neg_mean_squared_error')
    bayes_search_c = BayesSearchCV(whole_model_pipeline, search_space_catboost, cv=5, scoring='neg_mean_squared_error')

    # Assuming generate_run_name, fetch_logged_data, write_to_file, get_input_data, prepare_submission are defined elsewhere
    run_name = generate_run_name()
    with mlflow.start_run(run_name='f{run_name}_A') as run:
        bayes_search_a.fit(X_train_a, y_train_a)

    with mlflow.start_run(run_name='f{run_name}_B') as run:
        bayes_search_b.fit(X_train_b, y_train_b)

    with mlflow.start_run(run_name='f{run_name}_C') as run:
        bayes_search_c.fit(X_train_c, y_train_c)

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
    pred_a = run_pipeline_and_log(bayes_search_a, X_train_a, y_train_a, X_test_a, "A-" + run_name)
    logger.info("Run pipeline for location B")
    pred_b = run_pipeline_and_log(bayes_search_b, X_train_b, y_train_b, X_test_a, "B-" + run_name)
    logger.info("Run pipeline for location C")
    pred_c = run_pipeline_and_log(bayes_search_c, X_train_c, y_train_c, X_test_c, "C-" + run_name)

    prepare_submission(X_test_a, X_test_b, X_test_c, pred_a, pred_b, pred_c, run_name)

def bayes_search_catboost(drop_cols, model_name="bayes-search-catboost"):
    start_time = time.time()
    logger = logging.getLogger()

    logger.info('Preprocessing data')
    data_a, data_b, data_c = load_data()
    X_test_a, X_test_b, X_test_c = get_test_data()
    X_train_a, y_train_a = get_train_targets(data_a)
    X_train_b, y_train_b = get_train_targets(data_b)
    X_train_c, y_train_c = get_train_targets(data_c)
    logger.info('Done preprocessing data')

    if drop_cols: 
        model_name = model_name + '-drop-cols'
        logger.info(model_name)

        drop_cols_lst = ['time', 'date_calc', 'elevation:m', 'fresh_snow_1h:cm', 'wind_speed_u_10m:ms', 
            'wind_speed_u_10m:ms', 'wind_speed_v_10m:ms', 'wind_speed_w_1000hPa:ms', 'prob_rime:p',
            'fresh_snow_12h:cm','fresh_snow_24h:cm', 'fresh_snow_6h:cm', 'super_cooled_liquid_water:kgm2']
        
        data_process_pipeline = Pipeline([
            ('drop_cols', ColumnDropper(drop_cols=drop_cols_lst)),
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
        ])

    else: 
        model_name = model_name + '-all-features'
        logger.info(model_name)
        drop_cols_lst = ['time', 'date_calc']
        data_process_pipeline = Pipeline([
            ('drop_cols', ColumnDropper(drop_cols=drop_cols_lst)),
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
        ])

    locA_pipeline = Pipeline([
        ('data_process', data_process_pipeline),
        ('cat_boost', CatBoostRegressor(silent=True, random_state=2, loss_function='RMSE'))
    ])

    locB_pipeline = Pipeline([
        ('data_process', data_process_pipeline),
        ('cat_boost', CatBoostRegressor(silent=True, random_state=3, loss_function='RMSE'))
    ])

    locC_pipeline = Pipeline([
        ('data_process', data_process_pipeline),
        ('cat_boost', CatBoostRegressor(silent=True, random_state=2, loss_function='RMSE'))
    ])
    
    # Define the search space for BayesSearchCV
    search_space_catboost = {
        'cat_boost__iterations': Integer(100, 900),
        'cat_boost__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'cat_boost__depth': Integer(2, 9),
    }

    bayes_search_a = BayesSearchCV(locA_pipeline, search_space_catboost, cv=5, scoring='neg_mean_squared_error')
    bayes_search_b = BayesSearchCV(locB_pipeline, search_space_catboost, cv=5, scoring='neg_mean_squared_error')
    bayes_search_c = BayesSearchCV(locC_pipeline, search_space_catboost, cv=5, scoring='neg_mean_squared_error')

    # Assuming generate_run_name, fetch_logged_data, write_to_file, get_input_data, prepare_submission are defined elsewhere
    run_name = generate_run_name()
    with mlflow.start_run(run_name='f{run_name}_A') as run:
        bayes_search_a.fit(X_train_a, y_train_a)

    with mlflow.start_run(run_name='f{run_name}_B') as run:
        bayes_search_b.fit(X_train_b, y_train_b)

    with mlflow.start_run(run_name='f{run_name}_C') as run:
        bayes_search_c.fit(X_train_c, y_train_c)

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