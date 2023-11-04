"""
(TODO:) make get_input_data work for non-hot-encoding of location 
"""

from typing import Tuple
import pandas as pd

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

def data_preprocess(one_hot_location: bool = False) -> pd.DataFrame:
    """Checks if the data exists, reads data from the parquet files 
    given from data.zip from the exercise, and removes rows that we have no target data for.
    :args
        one_hot_location (bool, optional): include location feature as one-hot encoding. Defaults to False.
    :return 
        X_train: dataframe with all features and targets
    """

    # Check if files exist
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
        'data/A/X_test_estimated.parquet',
        'data/B/X_test_estimated.parquet',
        'data/C/X_test_estimated.parquet'
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

    # --- merge observed and estimated data with target data, lining up time-stamps correctly ----
    train_obs_a = pd.merge(train_a, X_train_observed_a, on='time', how='inner')
    train_obs_b = pd.merge(train_b, X_train_observed_b, on='time', how='inner') # NOTE: 4 missing values for target
    train_obs_c = pd.merge(train_c, X_train_observed_c, on='time', how='inner') # NOTE: 6059 missing values for target

    train_est_a = pd.merge(train_a, X_train_estimated_a, on='time', how='inner')
    train_est_b = pd.merge(train_b, X_train_estimated_b, on='time', how='inner')
    train_est_c = pd.merge(train_c, X_train_estimated_c, on='time', how='inner')

    # ----- merge the different locations together -------
    train_obs_a["location"] = "A"
    train_obs_b["location"] = "B"
    train_obs_c["location"] = "C"

    train_est_a["location"] = "A"
    train_est_b["location"] = "B"
    train_est_c["location"] = "C"

    X_train = pd.concat([train_obs_a, train_obs_b, train_obs_c, train_est_a, train_est_b, train_est_c], axis=0, ignore_index=True)


    # add one-hot encoding for location
    if one_hot_location:
        encoder = OneHotEncoder(sparse=False, categories='auto')
        encoded_location = encoder.fit_transform(X_train[['location']])
        encoded_df = pd.DataFrame(encoded_location, columns=['A', 'B', 'C'])
        onehot_df = pd.concat([X_train, encoded_df], axis=1)
        X_train = onehot_df.drop(columns=["location"])

    # remove rows that the target value is missing from since they will not be useful in model training
    X_train = X_train.dropna(subset=['pv_measurement'])
    X_train = X_train.dropna(subset=['time'])
    assert not X_train['time'].isnull().any(), "There are still NaT values in the 'time' column"

    X_train = remove_ouliers(X_train)

    X_train['month'] = X_train['time'].dt.month.astype(int)
    X_train['day'] = X_train['time'].dt.day.astype(int)
    X_train['hour'] = X_train['time'].dt.hour.astype(int)
    X_train['minute'] = X_train['time'].dt.minute.astype(int)
    X_train['second'] = X_train['time'].dt.second.astype(int)

    # Final check for NaT values

    return X_train

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

def get_training_data(
    X_train_with_targets: pd.DataFrame, 
    features: list = []
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Gets training data for the specified features, and associated target values.

    Args:
        X_train_with_targets (pd.DataFrame):  , for example output from data_preprocess()
        features (list): list of features to use during training

    Returns:
        X_train: df with values to use in model training
        targets: target values (pv_measurement)
    """
    
    targets = X_train_with_targets["pv_measurement"]
    if features == []:
        X_train = X_train_with_targets.drop(columns=["pv_measurement"])
    else:
        X_train = X_train_with_targets[features]

    return X_train, targets


def get_input_data() -> pd.DataFrame:
    """Loads test-data and merges (the time-points to make predictions for) into a single dataframe along with id's for submission.
    
    Returns:
        test_data (pd.DataFrame): df with timepoints to make predictions for, and features for these timepoints
    """    

    # Check if files exist
    file_paths = [
        'data/A/X_test_estimated.parquet',
        'data/B/X_test_estimated.parquet',
        'data/C/X_test_estimated.parquet',
        'data/test.csv'
    ]

    for file_path in file_paths:
        check_file_exists(file_path)

    # --- load test data from file ---
    X_test_estimated_a = pd.read_parquet('data/A/X_test_estimated.parquet').rename(columns={'date_forecast': 'time'})
    X_test_estimated_b = pd.read_parquet('data/B/X_test_estimated.parquet').rename(columns={'date_forecast': 'time'})
    X_test_estimated_c = pd.read_parquet('data/C/X_test_estimated.parquet').rename(columns={'date_forecast': 'time'})

    # --- process data ---

    # merge test data from different locations 
    X_test_estimated_a["location"] = "A"
    X_test_estimated_b["location"] = "B"
    X_test_estimated_c["location"] = "C"

    X_test = pd.concat([X_test_estimated_a, X_test_estimated_b, X_test_estimated_c], axis=0, ignore_index=True)

    # add one-hot encoding to location 
    encoder = OneHotEncoder(sparse=False, categories='auto')
    encoded_location = encoder.fit_transform(X_test[['location']])
    encoded_df = pd.DataFrame(encoded_location, columns=['A', 'B', 'C'])
    onehot_df = pd.concat([X_test, encoded_df], axis=1)
    X_test = onehot_df.drop(columns=["location"])

    # --- combine with "test" to get correct id's for kaggle submission ---
    
    # read data
    test = pd.read_csv('data/test.csv')
    
    # add one-hot encoding for location
    encoder = OneHotEncoder(sparse=False, categories='auto')
    encoded_location = encoder.fit_transform(test[['location']])
    encoded_df = pd.DataFrame(encoded_location, columns=['A', 'B', 'C'])
    onehot_df = pd.concat([test, encoded_df], axis=1)
    test = onehot_df.drop(columns=["location"])

    # convert "time" to datetime format
    test["time"] = pd.to_datetime(test["time"])

    # merge 
    test_data = pd.merge(X_test, test, on=["time", "A", "B", "C"])

    return test_data


