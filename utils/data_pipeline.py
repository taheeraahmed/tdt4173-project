from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

"""
Use this: 

data_process_pipeline = Pipeline([
    ('add_features', FeatureAdder()),
    ('drop_cols', ColumnDropper(drop_cols=drop_cols)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
])
"""

class FeatureAdder(BaseEstimator, TransformerMixin):
    """Adds features."""

    def __init__(self, drop_cols = [], is_test = False):
        self.drop_cols = drop_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # add moth
        X_copy['month'] = X_copy['time'].apply(lambda x: x.month)
        # add hour
        X_copy['hour'] = X_copy['time'].apply(lambda x: x.hour)

        # -- additive effects:
        #X_copy["sun_rad_1"] = (X_copy['sun_azimuth:d'] * X_copy['direct_rad:W']) / 1000000
        X_copy["sun_rad_2"] = (X_copy['sun_elevation:d'] * X_copy['direct_rad:W']) / 1000000
        #X_copy["sun_wind_1"] = (X_copy['wind_speed_10m:ms'] * X_copy['direct_rad:W']) / 1000
        X_copy["sun_wind_2"] = (X_copy['wind_speed_10m:ms'] * X_copy['diffuse_rad:W']) / 1000
        X_copy["temp_sun"] = (X_copy['t_1000hPa:K'] * X_copy['sun_azimuth:d'])/1000
        X_copy["rad_day_1"] = (X_copy['is_day:idx'] * X_copy['diffuse_rad:W']) / 1000
        X_copy['mult_coulds'] = (X_copy['clear_sky_rad:W'] * X_copy['cloud_base_agl:m']) / 100000

        # rolling averages
        X_copy = self.rolling_average(X_copy)
        return X_copy
    
    def rolling_average(self, df, window_size=24,features=['clear_sky_energy_1h:J','clear_sky_rad:W', 'direct_rad:W', 'direct_rad_1h:J', 'diffuse_rad:W', 'diffuse_rad_1h:J', 'total_cloud_cover:p', 'sun_elevation:d']):
        # Ensure the 'time' column is datetime and set as index

        features = ['precip_5min:mm', 'rain_water:kgm2', 'prob_rime:p', 't_1000hPa:K', 'snow_water:kgm2', 'visibility:m', # just this 7 nov
            'clear_sky_energy_1h:J','clear_sky_rad:W', 'direct_rad:W', 'direct_rad_1h:J', 'diffuse_rad:W', 'diffuse_rad_1h:J', 'total_cloud_cover:p', 'sun_elevation:d'] # added for 8 nov
        
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

class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops columns from the data."""

    def __init__(self, drop_cols = []):
        self.drop_cols = drop_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        return X_copy.drop(columns=self.drop_cols)