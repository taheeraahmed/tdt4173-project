
from tpot import TPOTRegressor
from utils.data_preprocess_location import load_data, get_train_targets, get_test_data, prepare_submission, remove_ouliers
from utils.data_preprocess import ColumnDropper
from utils.generate_run_name import generate_run_name
from sklearn.model_selection import train_test_split, RepeatedKFold
from tpot.builtins import ZeroCount, StackingEstimator
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import logging
import mlflow
from utils.pipeline import run_pipeline_and_log

def run_tpot_and_log(X, y, location, run_name):
    with mlflow.start_run(run_name=f"TPOT-{location}-{run_name}"):
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

def tpot_main(model_name='auto-ml'):

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
    run_tpot_and_log(X_A, y_A, "A", run_name)

    # ------ for location B -----

    X_B = X_train_b.drop(columns=drop_cols).fillna(0)
    y_B = targets_b

    logger.info("tpot regressor for location B")
    run_tpot_and_log(X_B, y_B, "B", run_name)

    # ------ for location C -----

    X_C = X_train_c.drop(columns=drop_cols).fillna(0)
    y_C = targets_c

    logger.info("tpot regressor for location C")
    run_tpot_and_log(X_C, y_C, "C", run_name)

def run_tpot_pipeline():
    logger = logging.getLogger()
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
    
    data_process_pipeline = Pipeline([
        ('drop_cols', ColumnDropper(drop_cols=drop_cols)),
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
    ])

    locA_pipeline = make_pipeline(
        data_process_pipeline,
        RobustScaler(),
        StackingEstimator(estimator=RidgeCV()),
        StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.1, min_samples_leaf=3, min_samples_split=3, n_estimators=100)),
        StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.1, min_samples_leaf=3, min_samples_split=3, n_estimators=100)),
        ExtraTreesRegressor(bootstrap=False, max_features=0.5, min_samples_leaf=1, min_samples_split=6, n_estimators=100)
    )

    locB_pipeline = make_pipeline(
        data_process_pipeline,
        StandardScaler(),
        RobustScaler(),
        MinMaxScaler(),
        StackingEstimator(estimator=RidgeCV()),
        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.1, min_samples_leaf=4, min_samples_split=2, n_estimators=100)),
        RobustScaler(),
        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.1, min_samples_leaf=4, min_samples_split=2, n_estimators=100)),
        StackingEstimator(estimator=RandomForestRegressor(bootstrap=False, max_features=0.1, min_samples_leaf=4, min_samples_split=2, n_estimators=100)),
        ExtraTreesRegressor(bootstrap=False, max_features=0.35000000000000003, min_samples_leaf=1, min_samples_split=5, n_estimators=100)
    )

    locC_pipeline = make_pipeline(
        data_process_pipeline,
        ZeroCount(),
        MinMaxScaler(),
        ZeroCount(),
        XGBRegressor(learning_rate=0.1, max_depth=9, min_child_weight=5, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.8500000000000001, verbosity=0)
    )
    run_name = 'Jimp Jota'

    logger.info("Run pipeline for location A")
    pred_a = run_pipeline_and_log(locA_pipeline, X_train_a, targets_a, X_test_a, "A", run_name)
    logger.info("Run pipeline for location B")
    pred_b = run_pipeline_and_log(locB_pipeline, X_train_b, targets_b, X_test_a, "B", run_name)
    logger.info("Run pipeline for location C")
    pred_c = run_pipeline_and_log(locC_pipeline, X_train_c, targets_c, X_test_c, "C",run_name)

    prepare_submission(X_test_a, X_test_b, X_test_c, pred_a, pred_b, pred_c, run_name)