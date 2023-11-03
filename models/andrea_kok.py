from sklearn import preprocessing
from utils.data_preprocess import data_preprocess, get_input_data, get_training_data
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor

from utils.generate_run_name import generate_run_name
from utils.log_model import fetch_logged_data, write_to_file
from utils.evaluate import prepare_submission, get_input_data, submission_to_csv

import mlflow
import time
import pandas as pd
import logging



def andrea_kok(model_name="andrea-kok"):
    logger = logging.getLogger()
    logger.info(model_name)
    
    # --- load data and do some clean-up and feature engineering ---
    data = data_preprocess(one_hot_location=True)

    # add a column for month of the year
    month = data['time'].apply(lambda x: x.month)

    features = ['absolute_humidity_2m:gm3',
        'air_density_2m:kgm3', 'ceiling_height_agl:m', 'clear_sky_energy_1h:J',
        'clear_sky_rad:W', 'cloud_base_agl:m', 'dew_or_rime:idx',
        'dew_point_2m:K', 'diffuse_rad:W', 'diffuse_rad_1h:J', 'direct_rad:W',
        'direct_rad_1h:J', 'effective_cloud_cover:p', 'elevation:m',
        'fresh_snow_12h:cm', 'fresh_snow_1h:cm', 'fresh_snow_24h:cm',
        'fresh_snow_3h:cm', 'fresh_snow_6h:cm', 'is_day:idx',
        'is_in_shadow:idx', 'msl_pressure:hPa', 'precip_5min:mm',
        'precip_type_5min:idx', 'pressure_100m:hPa', 'pressure_50m:hPa',
        'prob_rime:p', 'rain_water:kgm2', 'relative_humidity_1000hPa:p',
        'sfc_pressure:hPa', 'snow_density:kgm3', 'snow_depth:cm',
        'snow_drift:idx', 'snow_melt_10min:mm', 'snow_water:kgm2',
        'sun_azimuth:d', 'sun_elevation:d', 'super_cooled_liquid_water:kgm2',
        't_1000hPa:K', 'total_cloud_cover:p', 'visibility:m',
        'wind_speed_10m:ms', 'wind_speed_u_10m:ms', 'wind_speed_v_10m:ms',
        'wind_speed_w_1000hPa:ms', 'A', 'B', 'C']

    X_train, targets = get_training_data(data, features)

    X_train["month"] = month

    # fill missing values with 0
    X_train["ceiling_height_agl:m"].fillna(0, inplace=True)
    X_train["cloud_base_agl:m"].fillna(0, inplace=True)
    X_train["snow_density:kgm3"].fillna(0, inplace=True)

    # drop these columns, that scored lowest in the feature importance 
    drop_cols = ["precip_type_5min:idx", "is_in_shadow:idx", "rain_water:kgm2", "B", "fresh_snow_12h:cm", "C", "fresh_snow_6h:cm",
                "snow_depth:cm", "snow_density:kgm3", "snow_melt_10min:mm", "prob_rime:p", "fresh_snow_3h:cm", "dew_or_rime:idx",
                "wind_speed_w_1000hPa:ms", "fresh_snow_1h:cm", "is_day:idx", "snow_drift:idx"]

    X_train_drop = X_train.drop(columns=drop_cols)

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X_train_drop.values)
    y = targets


    ## --- stacked model ---

    # Define base models
    base_models = [
        ('random_forest', RandomForestRegressor(random_state=1)),
        ('gradient_boosting', GradientBoostingRegressor(random_state=1)),
        ('SGDRegressor', SGDRegressor(random_state=1)),
        ('bagging_regressor', BaggingRegressor(random_state=1))
    ]

    # Define meta-learner
    meta_learner = LinearRegression()

    # Create the stacking regressor
    stacked_model = StackingRegressor(estimators=base_models, final_estimator=meta_learner)

    # Start timer for logger
    start_time = time.time()  

    run_name = generate_run_name()
    with mlflow.start_run(run_name=run_name) as run:
        stacked_model.fit(X, y)
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

    new_features = [f for f in features if f not in drop_cols]

    X_test = get_input_data()
    X_test['month'] = X_test['time'].apply(lambda x: x.month)
    X_test_features = X_test[new_features].fillna(0)
    X_test_features['month'] = X_test['month']
    scaled_X_test = min_max_scaler.transform(X_test_features.values)
    
    predictions = stacked_model.predict(scaled_X_test)
    prepare_submission(X_test, predictions, run_name)
