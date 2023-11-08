from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class FeatureAdder(BaseEstimator, TransformerMixin):
    """Adds features."""

    def __init__(self, drop_cols = []):
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

        #X_copy["dirrad_airdensity"] = (X_copy['direct_rad:W'] * X_copy['air_density_2m:kgm3'])/1000 #unsure
        X_copy["ratio_rad1"] = (X_copy['direct_rad:W'] / X_copy['diffuse_rad:W']) # good one!
        #X_copy["diffrad_airdensity"] = (X_copy['diffuse_rad:W'] * X_copy['air_density_2m:kgm3'])/1000 #unsure
        X_copy["temp_rad_1"] = (X_copy['t_1000hPa:K'] * X_copy['direct_rad:W'])/1000

        return X_copy