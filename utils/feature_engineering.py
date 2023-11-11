import pandas as pd 
import numpy as np
def remove_outliers(data, remove_b_outliers = False):
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
    mean_df = df.groupby('time_hour').agg('mean', numeric_only=True).reset_index()

    return mean_df

def get_hourly_stats(df, important_features = ['clear_sky_energy_1h:J','clear_sky_rad:W', 'direct_rad:W', 'direct_rad_1h:J', 'diffuse_rad:W', 'diffuse_rad_1h:J', 'total_cloud_cover:p', 'sun_elevation:d']):
    """Returns a dataframe with hourly mean for all features and min/max for selected important features."""
    
    # get a column for the start hour
    df["time_hour"] = df["time"].apply(lambda x: x.floor('H'))
    
    # get the mean value for all features for the entire hour
    mean_df = df.groupby('time_hour').agg('mean', numeric_only=True).reset_index()

    # get min and max for selected important features
    min_max_df = df.groupby('time_hour')[important_features].agg(['min', 'max'], numeric_only=True).reset_index()

    min_max_df.columns = ['{}_{}'.format(col[0], col[1]) if col[1] != 'time_hour' else col[1] for col in min_max_df.columns]
    min_max_df.rename(columns={"time_hour_":"time_hour"}, inplace=True)

    # merge the mean and min/max dataframes on the time_hour column
    result_df = pd.merge(mean_df, min_max_df, on='time_hour')

    return result_df


def fill_pv_values(df):
    """Fill the pv-values to account for the entire hour"""

    # get a column for the start hour
    df["time_hour"] = df["time"].apply(lambda x: x.floor('H'))

    # Calculate linear interpolation for each hour
    df['pv_measurement'] = df.groupby('time_hour')['pv_measurement'].transform(lambda x: x.interpolate())

    # Drop the temporary column used for grouping
    df = df.drop(columns=["time_hour"])

    return df


def rolling_average(df, window_size=24,features=['clear_sky_energy_1h:J','clear_sky_rad:W', 'direct_rad:W', 'direct_rad_1h:J', 'diffuse_rad:W', 'diffuse_rad_1h:J', 'total_cloud_cover:p', 'sun_elevation:d']):
    #hard-code new features #TODO: add as param accessible outside of functions.py
    features = ['precip_5min:mm', 'rain_water:kgm2', 'prob_rime:p', 't_1000hPa:K', 'visibility:m',] # just this 7 nov # 'snow_water:kgm2'
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