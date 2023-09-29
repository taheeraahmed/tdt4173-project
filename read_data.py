import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

def read_data():
    '''
    Checks if the data exists and reads data from the parquet files 
    given from data.zip from the exercise
    :return 
        X_train: dataframe with all features
        X_train_with_targets: dataframe with all features and targets
    '''
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

    # Read data
    train_a = pd.read_parquet('data/A/train_targets.parquet')
    train_b = pd.read_parquet('data/B/train_targets.parquet')
    train_c = pd.read_parquet('data/C/train_targets.parquet')

    X_train_estimated_a = pd.read_parquet('data/A/X_train_estimated.parquet')
    X_train_estimated_b = pd.read_parquet('data/B/X_train_estimated.parquet')
    X_train_estimated_c = pd.read_parquet('data/C/X_train_estimated.parquet')

    X_train_observed_a = pd.read_parquet('data/A/X_train_observed.parquet')
    X_train_observed_b = pd.read_parquet('data/B/X_train_observed.parquet')
    X_train_observed_c = pd.read_parquet('data/C/X_train_observed.parquet')

    X_test_estimated_a = pd.read_parquet('data/A/X_test_estimated.parquet')
    X_test_estimated_b = pd.read_parquet('data/B/X_test_estimated.parquet')
    X_test_estimated_c = pd.read_parquet('data/C/X_test_estimated.parquet')

    # Concat the dataframes, getting all locations
    X_train_observed_a["location"] = "A"
    X_train_observed_b["location"] = "B"
    X_train_observed_c["location"] = "C"
    X_train_observed = pd.concat([X_train_observed_a, X_train_observed_b, X_train_observed_c], axis=0, ignore_index=True)

    X_train_estimated_a["location"] = "A"
    X_train_estimated_b["location"] = "B"
    X_train_estimated_c["location"] = "C"
    X_train_estimated = pd.concat([X_train_estimated_a, X_train_estimated_b, X_train_estimated_c], axis=0, ignore_index=True)

    train_a["location"] = "A"
    train_b["location"] = "B"
    train_c["location"] = "C"
    train = pd.concat([train_a, train_b, train_c], axis=0, ignore_index=True)

    # Rename columns
    X_train_observed.rename(columns={'date_forecast': 'time'}, inplace=True)
    X_train_estimated.rename(columns={'date_forecast': 'time'}, inplace=True)

    # Join X and Y data
    train_observed = pd.merge(X_train_observed, train, how="left", on=["time", "location"])
    train_estimated = pd.merge(X_train_estimated, train, how="left", on=["time", "location"])

    # Make location column categorical
    train_observed['location'] = pd.Categorical(train_observed.location)
    train_estimated['location'] = pd.Categorical(train_estimated.location)

    # Add time-diffs to est. 
    time_diffs = X_train_estimated["time"] - X_train_estimated["date_calc"]
    X_train_estimated["time_diffs"] = [t.seconds/3600 for t in time_diffs]
    X_train_observed["time_diffs"] = 0

    # Add feature for observed / estimated status
    X_train_estimated["is_observed"] = 0
    X_train_observed["is_observed"] = 1

    X_train = pd.concat([X_train_estimated, X_train_observed], axis=0, ignore_index=True)
    X_train_with_targets = pd.merge(X_train, train, how="left", on=["time", "location"])

    encoder = LabelEncoder()
    train_observed["location"] = encoder.fit_transform(train_observed["location"])
    train_estimated["location"] = encoder.fit_transform(train_estimated["location"])
    X_train["location"] = encoder.fit_transform(X_train["location"])
    X_train_with_targets["location"] = encoder.fit_transform(X_train_with_targets["location"])

    return X_train, X_train_with_targets