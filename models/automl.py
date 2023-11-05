
from tpot import TPOTRegressor
from utils.data_preprocess_location import load_data, get_train_targets, get_test_data, prepare_submission, remove_ouliers
from utils.data_preprocess import ColumnDropper
from utils.generate_run_name import generate_run_name
from sklearn.model_selection import train_test_split, RepeatedKFold

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
import logging

import mlflow

def run_tpot_and_log(X, y, location):
    with mlflow.start_run(run_name=f"TPOT_{location}"):
        cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
        tpot = TPOTRegressor(generations=100, population_size=100, scoring='neg_mean_absolute_error', cv=cv, verbosity=2, random_state=1, n_jobs=-1)
        
        tpot.fit(X, y)
        
        # Log parameters
        mlflow.log_param("location", location)
        mlflow.log_param("generations", 100)
        mlflow.log_param("population_size", 100)
        mlflow.log_param("scoring", 'neg_mean_absolute_error')
        
        # Log best score
        mlflow.log_metric("neg_mean_absolute_error", tpot._optimized_pipeline_score)
        
        # Export the pipeline to a python script
        export_file = f'tpot/tpot_{location}_best_model.py'
        tpot.export(export_file)
        
        # Log the exported pipeline script as an artifact
        mlflow.log_artifact(export_file, "tpot_models")

def run_pipeline_and_log(pipeline, X_train, y_train, X_test, location):
    with mlflow.start_run(run_name=f"Pipeline_{location}"):
        # Fit the pipeline
        pipeline.fit(X_train, y_train)
        
        # Predict on the test set
        predictions = pipeline.predict(X_test.drop(columns=["id", "prediction", "location"]))
        
        # Log model parameters (you would log all relevant parameters of your model)
        # For example, if you have a RandomForestRegressor in your pipeline:
        if 'random_forest' in pipeline.named_steps:
            rf = pipeline.named_steps['random_forest']
            mlflow.log_param("n_estimators", rf.n_estimators)
            mlflow.log_param("max_features", rf.max_features)
            # ... log other parameters as needed
        
        # Log metrics (you would log the metrics relevant to your problem)
        # For example, if you have a scoring function or validation scores:
        # log_metric("score", your_scoring_function(y_test, predictions))
        
        # Log the model
        mlflow.sklearn.log_model(pipeline, f"pipeline_{location}")
        
        # Log the predictions as an artifact
        np.savetxt(f"predictions_{location}.csv", predictions, delimiter=",")
        mlflow.log_artifact(f"predictions_{location}.csv")
    return predictions

def automl(model_name='auto-ml'):

    logger = logging.getLogger()

    logger.info(model_name)
    run_name = generate_run_name()

    logger.info('Processing data')
    data_a, data_b, data_c = load_data()

    data_a = remove_ouliers(data_a)
    data_b = remove_ouliers(data_b)
    data_c = remove_ouliers(data_c)

    X_train_a, targets_a = get_train_targets(data_a)
    X_train_b, targets_b = get_train_targets(data_b)
    X_train_c, targets_c = get_train_targets(data_c)

    X_test_a, X_test_b, X_test_c = get_test_data()


    drop_cols = ['time', 'date_calc', 'elevation:m', 'fresh_snow_1h:cm', 'wind_speed_u_10m:ms', 
                'wind_speed_u_10m:ms', 'wind_speed_v_10m:ms', 'wind_speed_w_1000hPa:ms', 'prob_rime:p',
                'fresh_snow_12h:cm','fresh_snow_24h:cm', 'fresh_snow_6h:cm', 'super_cooled_liquid_water:kgm2']

    logger.info('Done processing data')
    # ------ for location A -----

    X_A = X_train_a.drop(columns=drop_cols).fillna(0)
    y_A = targets_a

    logger.info("tpot regressor for location A")
    run_tpot_and_log(X_A, y_A, "A-"+ run_name)

    # ------ for location B -----

    X_B = X_train_b.drop(columns=drop_cols).fillna(0)
    y_B = targets_b

    logger.info("tpot regressor for location B")
    run_tpot_and_log(X_B, y_B, "B-"+ run_name)

    # ------ for location C -----

    X_C = X_train_c.drop(columns=drop_cols).fillna(0)
    y_C = targets_c

    logger.info("tpot regressor for location C")
    run_tpot_and_log(X_C, y_C, "C-"+ run_name)

    # ---- run 
    data_process_pipeline = Pipeline([
        ('drop_cols', ColumnDropper(drop_cols=drop_cols)),
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
    ])

    locA_pipeline = Pipeline([
        ('data_process', data_process_pipeline),
        ('random_forest', XGBRegressor(learning_rate=0.1, max_depth=9, min_child_weight=6, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.7000000000000001, verbosity=0))
    ])

    locB_pipeline = Pipeline([
        ('data_process', data_process_pipeline),
        ('random_forest', RandomForestRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=5, min_samples_split=2, n_estimators=100, random_state=1))
    ])

    locC_pipeline = Pipeline([
        ('data_process', data_process_pipeline),
        ('random_forest', XGBRegressor(learning_rate=0.1, max_depth=9, min_child_weight=3, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.8500000000000001, verbosity=0))
    ])

    logger.info("Run pipeline for location A")
    pred_a = run_pipeline_and_log(locA_pipeline, X_train_a, targets_a, X_test_a, "A-" + run_name)
    logger.info("Run pipeline for location B")
    pred_b = run_pipeline_and_log(locB_pipeline, X_train_b, targets_b, X_test_a, "B-" + run_name)
    logger.info("Run pipeline for location C")
    pred_c = run_pipeline_and_log(locC_pipeline, X_train_c, targets_c, X_test_c, "C-" + run_name)

    prepare_submission(X_test_a, X_test_b, X_test_c, pred_a, pred_b, pred_c, run_name)