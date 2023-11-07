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
import logging
import datetime

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    

def load_data(mean=False):
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

    if mean: 
        data_a = calculate_means_and_replace(data_a)
        data_b = calculate_means_and_replace(data_b)
        data_c = calculate_means_and_replace(data_c)

    return data_a, data_b, data_c


def remove_ouliers(data):
    """Removes datapoints that have been static over long stretches (likely due to sensor error!)."""

    threshold = 0.01
    window_size = 24 

    # Calculate standard deviation for each window
    std_dev = data['pv_measurement'].rolling(window=window_size, min_periods=1).std()

    # Identify constant stretches and create a mask to filter out these points
    constant_mask = std_dev < threshold

    # Filter out constant stretches from the data
    filtered_data = data[~constant_mask]

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


def get_train_targets(data):
    """Sepperate out features from the training data"""
    targets = data["pv_measurement"]
    X_train = data.drop(columns=["pv_measurement"])
    return X_train, targets


def get_test_data(mean=False):
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
    X_test_estimated_a = get_hourly(X_test_estimated_a)
    X_test_estimated_b = get_hourly(X_test_estimated_b)
    X_test_estimated_c = get_hourly(X_test_estimated_c)

    X_test_estimated_a.rename(columns={"time_hour": "time"}, inplace=True)
    X_test_estimated_b.rename(columns={"time_hour": "time"}, inplace=True)
    X_test_estimated_c.rename(columns={"time_hour": "time"}, inplace=True)

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

    if mean: 
        X_test_a = calculate_means_and_replace(X_test_a)
        X_test_a = calculate_means_and_replace(X_test_a)
        X_test_a = calculate_means_and_replace(X_test_a)

    return X_test_a, X_test_b, X_test_c


def calculate_means_and_replace(df):
    """
    Calculate the mean for columns in the DataFrame that match a certain pattern and replace them with a single mean column.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    suffix (str): The suffix pattern that identifies the columns for which to calculate the mean.

    Returns:
    pd.DataFrame: The DataFrame with the original columns replaced by their mean.
    """
    # Get the base patterns by stripping the trailing '_<number>'
    base_patterns = set(col.rsplit('_', 1)[0] for col in df.columns if '_' in col)

    # For each base pattern, calculate the mean and replace the columns
    for base_pattern in base_patterns:
        # Find all columns that start with the base pattern and end with a number
        pattern_columns = [col for col in df.columns if col.startswith(base_pattern) and col.split('_')[-1].isdigit()]
        # Calculate the mean of these columns
        df[base_pattern + '_mean'] = df[pattern_columns].mean(axis=1)
        # Drop the original columns
        df.drop(pattern_columns, axis=1, inplace=True)
    return df

def prepare_submission(X_test_a, X_test_b, X_test_c, pred_a, pred_b, pred_c, run_name):
    """Parses the test data and predictions into a single df in kaggle submission format"""
    logger = logging.getLogger()
    # Create dataframe
    submission_a = X_test_a.copy()
    submission_b = X_test_b.copy()
    submission_c = X_test_c.copy()

    # Everything smaller than 0 is set to 0
    pred_a[pred_a < 0] = 0
    pred_b[pred_b < 0] = 0
    pred_c[pred_c < 0] = 0

    submission_a["prediction"] = pred_a
    submission_b["prediction"] = pred_b
    submission_c["prediction"] = pred_c

    submission = pd.concat([submission_a, submission_b, submission_c])
    submission = submission[["id", "prediction"]]

    # Create filename
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    submission_directory = 'submissions'  # Change to the desired directory path
    run_name = run_name.lower().replace(" ", "-")
    submission_filename = formatted_datetime+ "-"+ run_name + '.csv'

    # Check if the directory exists; if not, create it
    if not os.path.exists(submission_directory):
        os.makedirs(submission_directory)

    # Save the submission CSV in the specified directory
    submission.to_csv(os.path.join(submission_directory, submission_filename), index=False)
    logger.info("Saved submission file " + formatted_datetime + "-" + run_name + '.csv' )
