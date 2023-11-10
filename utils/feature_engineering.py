import numpy as np 
import pandas as pd

def cyclic_encoding(df):
    df['time'] = pd.to_datetime(df['time'])
    df['normalized_time'] = (df['time'].dt.hour + df['time'].dt.minute / 60 + df['time'].dt.second / 3600) / 24.0
    df['sine_encoded_day'] = np.sin(2 * np.pi * df['normalized_time'])
    df['cosine_encoded_day'] = np.cos(2 * np.pi * df['normalized_time'])

    df['normalized_month'] = (df['time'].dt.month - 1) / 11.0
    df['sine_month'] = np.sin(2 * np.pi * df['normalized_month'])
    df['cosine_month'] = np.cos(2 * np.pi * df['normalized_month'])

    df.drop('normalized_month', axis=1, inplace=True)
    df.drop('normalized_time', axis=1, inplace=True)
    return df


def normalize(df, columns_to_normalize=None):
    if columns_to_normalize is None:
        columns_to_normalize = df.columns  # Normalize all columns by default
    min_max_scale = lambda col: (col - col.min()) / (col.max() - col.min()) if col.min() != col.max() else col
    df[columns_to_normalize] = df[columns_to_normalize].apply(min_max_scale)
    return df

def add_custom_features(X_copy):
    # -- additive effects:
    X_copy["sun_rad_1"] = (X_copy['sun_azimuth:d'] * X_copy['direct_rad:W']) / 1000000
    X_copy["sun_rad_2"] = (X_copy['sun_elevation:d'] * X_copy['direct_rad:W']) / 1000000
    #X_copy["sun_wind_1"] = (X_copy['wind_speed_10m:ms'] * X_copy['direct_rad:W']) / 1000
    X_copy["sun_wind_2"] = (X_copy['wind_speed_10m:ms'] * X_copy['diffuse_rad:W']) / 1000
    X_copy["temp_sun"] = (X_copy['t_1000hPa:K'] * X_copy['sun_azimuth:d'])/1000
    X_copy["rad_day_1"] = (X_copy['is_day:idx'] * X_copy['diffuse_rad:W']) / 1000
    X_copy['mult_coulds'] = (X_copy['clear_sky_rad:W'] * X_copy['cloud_base_agl:m']) / 100000

    #X_copy["dirrad_airdensity"] = (X_copy['direct_rad:W'] * X_copy['air_density_2m:kgm3'])/1000 #unsure
    X_copy["ratio_rad1"] = (X_copy['direct_rad:W'] / X_copy['diffuse_rad:W']) # good one!
    #X_copy["diffrad_airdensity"] = (X_copy['diffuse_rad:W'] * X_copy['air_density_2m:kgm3'])/1000 #unsure
    X_copy["temp_rad_1"] = (X_copy['t_1000hPa:K'] * X_copy['direct_rad:W'])/1000

    # X_copy["ratio_rad1"] = (X_copy['direct_rad:W'] / X_copy['diffuse_rad:W']) # good one!
    # X_copy["temp_rad_1"] = (X_copy['t_1000hPa:K'] * X_copy['direct_rad:W'])/1000
    return X_copy

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
    df["time_hour"] = df["time"].apply(lambda x: x.floor('H'))
    mean_df = df.groupby('time_hour').agg('mean').reset_index()
    return mean_df

def remove_ouliers(data, remove_b_outliers = False):
    """Rem"""
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