"""
Helper functions for:
- loading data
- parsing submission
- ...

NOTE: all functions file should be pasted into the long notebook before submission.
"""

import pandas as pd
import numpy as np
import os

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    

def load_data(mean=False, roll_avg=False, remove_out=False):
    """Loads data, drops rows that have missing values for the target variable."""

    # --- Check if files exist ---
    file_paths = [
        'data/A/train_targets.parquet',
        'data/B/train_targets.parquet',
        'data/C/train_targets.parquet',
        'data/A/X_train_estimated.parquet',
        'data/B/X_train_estimated.parquet',
        'data/C/X_train_estimated.parquet',
        'data/A/X_train_observed.parquet',
        'data/B/X_train_observed.parquet',
        'data/C/X_train_observed.parquet',
    ]

    for file_path in file_paths:
        check_file_exists(file_path)

    # ---- load data from files ----
    train_a = pd.read_parquet('data/A/train_targets.parquet')
    train_b = pd.read_parquet('data/B/train_targets.parquet')
    train_c = pd.read_parquet('data/C/train_targets.parquet')

    X_train_observed_a = pd.read_parquet('data/A/X_train_observed.parquet').rename(columns={'date_forecast': 'time'})
    X_train_observed_b = pd.read_parquet('data/B/X_train_observed.parquet').rename(columns={'date_forecast': 'time'})
    X_train_observed_c = pd.read_parquet('data/C/X_train_observed.parquet').rename(columns={'date_forecast': 'time'})

    X_train_estimated_a = pd.read_parquet('data/A/X_train_estimated.parquet').rename(columns={'date_forecast': 'time'})
    X_train_estimated_b = pd.read_parquet('data/B/X_train_estimated.parquet').rename(columns={'date_forecast': 'time'})
    X_train_estimated_c = pd.read_parquet('data/C/X_train_estimated.parquet').rename(columns={'date_forecast': 'time'})

    # --- get features for each hour before concatinating ---
    if mean:
        X_train_observed_a = get_hourly_mean(X_train_observed_a)
        X_train_observed_b = get_hourly_mean(X_train_observed_b)
        X_train_observed_c = get_hourly_mean(X_train_observed_c)

        X_train_estimated_a = get_hourly_mean(X_train_estimated_a)
        X_train_estimated_b = get_hourly_mean(X_train_estimated_b)
        X_train_estimated_c = get_hourly_mean(X_train_estimated_c)

    else:
        X_train_observed_a = get_hourly(X_train_observed_a)
        X_train_observed_b = get_hourly(X_train_observed_b)
        X_train_observed_c = get_hourly(X_train_observed_c)

        X_train_estimated_a = get_hourly(X_train_estimated_a)
        X_train_estimated_b = get_hourly(X_train_estimated_b)
        X_train_estimated_c = get_hourly(X_train_estimated_c)

    X_train_observed_a.rename(columns={"time_hour": "time"}, inplace=True)
    X_train_observed_b.rename(columns={"time_hour": "time"}, inplace=True)
    X_train_observed_c.rename(columns={"time_hour": "time"}, inplace=True)
    X_train_estimated_a.rename(columns={"time_hour": "time"}, inplace=True)
    X_train_estimated_b.rename(columns={"time_hour": "time"}, inplace=True)
    X_train_estimated_c.rename(columns={"time_hour": "time"}, inplace=True)

    X_train_observed_a["estimated_flag"] = 0
    X_train_observed_b["estimated_flag"] = 0
    X_train_observed_c["estimated_flag"] = 0
    X_train_estimated_a["estimated_flag"] = 1
    X_train_estimated_b["estimated_flag"] = 1
    X_train_estimated_c["estimated_flag"] = 1

    # --- merge observed and estimated data with target data, lining up time-stamps correctly ----
    train_obs_a = pd.merge(train_a, X_train_observed_a, on='time', how='inner')
    train_obs_b = pd.merge(train_b, X_train_observed_b, on='time', how='inner') # NOTE: 4 missing values for target
    train_obs_c = pd.merge(train_c, X_train_observed_c, on='time', how='inner') # NOTE: 6059 missing values for target

    train_est_a = pd.merge(train_a, X_train_estimated_a, on='time', how='inner')
    train_est_b = pd.merge(train_b, X_train_estimated_b, on='time', how='inner')
    train_est_c = pd.merge(train_c, X_train_estimated_c, on='time', how='inner')

    data_a = pd.concat([train_obs_a, train_est_a], axis=0, ignore_index=True)
    data_b = pd.concat([train_obs_b, train_est_b], axis=0, ignore_index=True)
    data_c = pd.concat([train_obs_c, train_est_c], axis=0, ignore_index=True)

    # remove rows that the target value is missing from since they will not be useful in model training
    data_a = data_a.dropna(subset=['pv_measurement'])
    data_b = data_b.dropna(subset=['pv_measurement'])
    data_c = data_c.dropna(subset=['pv_measurement'])

    # add columnns for rolling average
    if roll_avg:
        data_a = rolling_average(data_a)
        data_b = rolling_average(data_b)
        data_c = rolling_average(data_c)

    if remove_out:
        data_a = remove_ouliers(data_a)
        data_b = remove_ouliers(data_b, remove_b_outliers=True)
        data_c = remove_ouliers(data_c)

    return data_a, data_b, data_c


def remove_ouliers(data, remove_b_outliers = False):
    """Removes datapoints that have been static over long stretches (likely due to sensor error!)."""

    threshold = 0.01
    window_size = 24

    # Calculate standard deviation for each window
    std_dev = data['pv_measurement'].rolling(window=window_size, min_periods=1).std()

    # Identify constant stretches and create a mask to filter out these points
    constant_mask = std_dev < threshold

    # Filter out constant stretches from the data
    filtered_data = data[~constant_mask]

    if remove_b_outliers:
        "removing some extra outliers"
        # Remove rows where pv_measurement > 100 and diffuse_rad:W < 30
        filtered_data = filtered_data[~((filtered_data["pv_measurement"] > 100) & (filtered_data["diffuse_rad:W"] < 30))]

        # Remove rows where pv_measurement > 200 and diffuse_rad:W < 40
        filtered_data = filtered_data[~((filtered_data["pv_measurement"] > 200) & (filtered_data["diffuse_rad:W"] < 40))]

    return filtered_data


def get_hourly(df):
    
    df["minute"] = df["time"].dt.minute

    min_vals = df["minute"].unique()

    df_list = []

    for value in min_vals:
        filtered_data = df[df['minute'] == value].copy()
        filtered_data.drop(columns=['minute'], inplace=True)
        filtered_data.columns = [f'{col}_{value}' for col in filtered_data.columns]
        filtered_data["time_hour"] = filtered_data["time_"+str(value)].apply(lambda x: x.floor('H'))
        df_list.append(filtered_data)

    # merge df's on hourly time
    merged_df = pd.merge(df_list[0], df_list[1], on="time_hour")
    for df in df_list[2:]:
        merged_df = pd.merge(merged_df, df, on="time_hour")

    return merged_df

def get_hourly_mean(df):
    """Returns a dataframe in which """
    
    # get a column for the start hour
    df["time_hour"] = df["time"].apply(lambda x: x.floor('H'))
    
    # get the mean value for the entire hour
    mean_df = df.groupby('time_hour').agg('mean').reset_index()

    return mean_df

def rolling_average(df, window_size=24,features=['clear_sky_energy_1h:J','clear_sky_rad:W', 'direct_rad:W', 'direct_rad_1h:J', 'diffuse_rad:W', 'diffuse_rad_1h:J', 'total_cloud_cover:p', 'sun_elevation:d']):
    
    #hard-code new features #TODO: add as param accessible outside of functions.py
    features = ['precip_5min:mm', 'rain_water:kgm2', 'prob_rime:p', 't_1000hPa:K', 'snow_water:kgm2', 'visibility:m',] # just this 7 nov
               # 'clear_sky_energy_1h:J','clear_sky_rad:W', 'direct_rad:W', 'direct_rad_1h:J', 'diffuse_rad:W', 'diffuse_rad_1h:J', 'total_cloud_cover:p', 'sun_elevation:d'] # added for 8 nov
    
    # Ensure the 'time' column is datetime and set as index
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True, drop=False)
    df.sort_index(inplace=True)

    # Calculate rolling averages for the specified features
    for feature in features:
        rolling_feature_name = f"{feature}_rolling_avg_{window_size}"
        df[rolling_feature_name] = df[feature].rolling(window=window_size).mean()

    # Handle missing data if necessary
    df.fillna(method='bfill', inplace=True)  # Forward fill

    return df


def get_train_targets(data):
    """Sepperate out features from the training data"""
    targets = data["pv_measurement"]
    X_train = data.drop(columns=["pv_measurement"])
    return X_train, targets


def get_test_data(mean=False, roll_avg=False):
    """Parse the test data, getting the data that has a kaggle submission id for all locations"""

    # --- Check if files exist ---
    file_paths = [
        'data/A/X_test_estimated.parquet',
        'data/B/X_test_estimated.parquet',
        'data/C/X_test_estimated.parquet',
        'data/test.csv'
    ]

    for file_path in file_paths:
        check_file_exists(file_path)

    # --- load all test data from file ---
    X_test_estimated_a = pd.read_parquet('data/A/X_test_estimated.parquet').rename(columns={'date_forecast': 'time'})
    X_test_estimated_b = pd.read_parquet('data/B/X_test_estimated.parquet').rename(columns={'date_forecast': 'time'})
    X_test_estimated_c = pd.read_parquet('data/C/X_test_estimated.parquet').rename(columns={'date_forecast': 'time'})

    # --- get hourly and rename ---
    if mean:
        X_test_estimated_a = get_hourly_mean(X_test_estimated_a)
        X_test_estimated_b = get_hourly_mean(X_test_estimated_b)
        X_test_estimated_c = get_hourly_mean(X_test_estimated_c)
    else:
        X_test_estimated_a = get_hourly(X_test_estimated_a)
        X_test_estimated_b = get_hourly(X_test_estimated_b)
        X_test_estimated_c = get_hourly(X_test_estimated_c)

    X_test_estimated_a.rename(columns={"time_hour": "time"}, inplace=True)
    X_test_estimated_b.rename(columns={"time_hour": "time"}, inplace=True)
    X_test_estimated_c.rename(columns={"time_hour": "time"}, inplace=True)

    X_test_estimated_a["estimated_flag"] = 1
    X_test_estimated_b["estimated_flag"] = 1
    X_test_estimated_c["estimated_flag"] = 1

    # --- load kaggle submission data ---
    test = pd.read_csv('data/test.csv')
    test["time"] = pd.to_datetime(test["time"]) # convert "time" to datetime format to facilitate merge
    kaggle_submission_a = test[test["location"]=="A"]
    kaggle_submission_b = test[test["location"]=="B"]
    kaggle_submission_c = test[test["location"]=="C"]

    # --- get only the test data with a corresponding kaggle submission id ---
    X_test_a = pd.merge(X_test_estimated_a, kaggle_submission_a, on="time", how="right")
    X_test_b = pd.merge(X_test_estimated_b, kaggle_submission_b, on="time", how="right")
    X_test_c = pd.merge(X_test_estimated_c, kaggle_submission_c, on="time", how="right")

    if roll_avg:
        X_test_a = rolling_average(X_test_a)
        X_test_b = rolling_average(X_test_b)
        X_test_c = rolling_average(X_test_c)

    return X_test_a, X_test_b, X_test_c


def prepare_submission(X_test_a, X_test_b, X_test_c, pred_a, pred_b, pred_c):
    """Parses the test data and predictions into a single df in kaggle submission format"""
    
    submission_a = X_test_a.copy()
    submission_b = X_test_b.copy()
    submission_c = X_test_c.copy()

    submission_a["prediction"] = pred_a
    submission_b["prediction"] = pred_b
    submission_c["prediction"] = pred_c

    submission = pd.concat([submission_a, submission_b, submission_c])

    submission = submission[["id", "prediction"]]

    return submission