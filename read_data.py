import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

def read_data():

    # read data
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

    # concat the dataframes, getting all locations

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

    # rename columns

    X_train_observed.rename(columns={'date_forecast': 'time'}, inplace=True)
    X_train_estimated.rename(columns={'date_forecast': 'time'}, inplace=True)

    # join X and Y data

    train_observed = pd.merge(X_train_observed, train, how="left", on=["time", "location"])
    train_estimated = pd.merge(X_train_estimated, train, how="left", on=["time", "location"])

    # make location column categorical
    train_observed['location'] = pd.Categorical(train_observed.location)
    train_estimated['location'] = pd.Categorical(train_estimated.location)

    # add time-diffs to est. 
    time_diffs = X_train_estimated["time"] - X_train_estimated["date_calc"]
    X_train_estimated["time_diffs"] = [t.seconds/3600 for t in time_diffs]
    X_train_observed["time_diffs"] = 0

    # add feature for observed / estimated status
    X_train_estimated["is_observed"] = 0
    X_train_observed["is_observed"] = 1

    X_train = pd.concat([X_train_estimated, X_train_observed], axis=0, ignore_index=True)
    X_train_with_targets = pd.merge(X_train, train, how="left", on=["time", "location"])

    return X_train, X_train_with_targets