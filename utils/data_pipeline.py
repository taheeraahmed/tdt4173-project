from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from utils.feature_engineering import get_hourly, get_hourly_mean, cyclic_encoding, add_custom_features,remove_ouliers, rolling_average, normalize
import pandas as pd

"""
Use this: 

data_process_pipeline = Pipeline([
    ('add_features', FeatureAdder()),
    ('drop_cols', ColumnDropper(drop_cols=drop_cols)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
])
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

class FeatureAdder(BaseEstimator, TransformerMixin):
    """Adds features."""

    def __init__(self, drop_cols = []):
        self.drop_cols = drop_cols

    def fit(self, X, y=None):
        return self
    
    def cyclic_encoding(self, df):
        df['time'] = pd.to_datetime(df['time'])
        df['normalized_time'] = (df['time'].dt.hour + df['time'].dt.minute / 60 + df['time'].dt.second / 3600) / 24.0
        df['sine_encoded'] = np.sin(2 * np.pi * df['normalized_time'])
        df['cosine_encoded'] = np.cos(2 * np.pi * df['normalized_time'])

        month = df['time'].dt.month
        df['sine_encoded_month'] = np.sin(2 * np.pi * month)
        df['cosine_encoded_month'] = np.cos(2 * np.pi * month)

        df.drop('normalized_time', axis=1, inplace=True)
        return df

    def transform(self, X):
        X_copy = X.copy()

        # # add moth
        # X_copy['month'] = X_copy['time'].apply(lambda x: x.month)

        # # add hour
        # X_copy['hour'] = X_copy['time'].apply(lambda x: x.hour)

        X_copy = self.cyclic_encoding(X_copy)

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
class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops columns from the data."""

    def __init__(self, drop_cols = []):
        self.drop_cols = drop_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        return X_copy.drop(columns=self.drop_cols)