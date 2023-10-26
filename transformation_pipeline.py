"""Full pipeline to do data processing a bit cleaner, and run model. Remember to use the data processing pipeline to transform test data too."""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from data_preprocess import data_preprocess, get_input_data, get_training_data, prepare_submission
from sklearn.compose import ColumnTransformer

drop_cols = ["precip_type_5min:idx", "is_in_shadow:idx", "rain_water:kgm2", "B", "fresh_snow_12h:cm", "C", "fresh_snow_6h:cm",
             "snow_depth:cm", "snow_density:kgm3", "snow_melt_10min:mm", "prob_rime:p", "fresh_snow_3h:cm", "dew_or_rime:idx",
             "wind_speed_w_1000hPa:ms", "fresh_snow_1h:cm", "snow_drift:idx"]


## --- functions and classes ---

class TimeFeatureAdder(BaseEstimator, TransformerMixin):
    """Adds the features month and hour to the data"""

    def __init__(self, add_features=True):
        self.add_features = add_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X_copy = X.copy()

        timestamps = X["time"]
        month = timestamps.apply(lambda x: x.month)
        hour = timestamps.apply(lambda x: x.hour)
        day = timestamps.apply(lambda x: x.day)

        if self.add_features:
            X_copy["month"] = month
            X_copy["hour"] = hour
            X_copy["day"] = day
            return X_copy
        else:
            return X_copy
        
class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops the time column from the data"""

    def __init__(self, drop_time=True, drop_cols = []):
        self.drop_time = drop_time
        self.drop_cols = drop_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X_copy = X.copy()

        if self.drop_time:
            return X_copy.drop(columns=['time'])

        X_copy.drop(columns=self.drop_cols)

        return X_copy

data_process_pipeline = Pipeline([
    ('add_features', TimeFeatureAdder()),
    ('drop_cols', ColumnDropper(drop_cols=drop_cols)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
    ('minMaxScaling', MinMaxScaler()),
])


# Define base models
base_models = [
    ('random_forest1', RandomForestRegressor(random_state=1)),
    ('random_forest2', RandomForestRegressor(random_state=2)),
    ('random_forest3', RandomForestRegressor(random_state=3)),
    ('random_forest4', RandomForestRegressor(random_state=4))
]

# Define meta-learner
meta_learner = LinearRegression()

# Create the stacking regressor
stacked_model = StackingRegressor(estimators=base_models, final_estimator=meta_learner)


whole_model_pipeline = Pipeline([
    ('data_process', data_process_pipeline),
    ('stacked_model', stacked_model)
])

## --- run the script ---

data = data_preprocess(one_hot_location=True)

features = ['absolute_humidity_2m:gm3',
       'air_density_2m:kgm3', 'ceiling_height_agl:m', 'clear_sky_energy_1h:J',
       'clear_sky_rad:W', 'cloud_base_agl:m', 'dew_or_rime:idx',
       'dew_point_2m:K', 'diffuse_rad:W', 'diffuse_rad_1h:J', 'direct_rad:W',
       'direct_rad_1h:J', 'effective_cloud_cover:p', 'elevation:m',
       'fresh_snow_12h:cm', 'fresh_snow_1h:cm', 'fresh_snow_24h:cm',
       'fresh_snow_3h:cm', 'fresh_snow_6h:cm', 'is_day:idx',
       'is_in_shadow:idx', 'msl_pressure:hPa', 'precip_5min:mm',
       'precip_type_5min:idx', 'pressure_100m:hPa', 'pressure_50m:hPa',
       'prob_rime:p', 'rain_water:kgm2', 'relative_humidity_1000hPa:p',
       'sfc_pressure:hPa', 'snow_density:kgm3', 'snow_depth:cm',
       'snow_drift:idx', 'snow_melt_10min:mm', 'snow_water:kgm2',
       'sun_azimuth:d', 'sun_elevation:d', 'super_cooled_liquid_water:kgm2',
       't_1000hPa:K', 'total_cloud_cover:p', 'visibility:m',
       'wind_speed_10m:ms', 'wind_speed_u_10m:ms', 'wind_speed_v_10m:ms',
       'wind_speed_w_1000hPa:ms', 'A', 'B', 'C', 'time']

cat_cols = ['is_day:idx', 'is_in_shadow:idx','month', 'hour', 'day']
one_hot_cols = ['A', 'B', 'C']
num_cols = ['absolute_humidity_2m:gm3',
       'air_density_2m:kgm3', 'ceiling_height_agl:m', 'clear_sky_energy_1h:J',
       'clear_sky_rad:W', 'cloud_base_agl:m', 'dew_or_rime:idx',
       'dew_point_2m:K', 'diffuse_rad:W', 'diffuse_rad_1h:J', 'direct_rad:W',
       'direct_rad_1h:J', 'effective_cloud_cover:p', 'elevation:m',
       'fresh_snow_12h:cm', 'fresh_snow_1h:cm', 'fresh_snow_24h:cm',
       'fresh_snow_3h:cm', 'fresh_snow_6h:cm','msl_pressure:hPa', 'precip_5min:mm',
       'precip_type_5min:idx', 'pressure_100m:hPa', 'pressure_50m:hPa',
       'prob_rime:p', 'rain_water:kgm2', 'relative_humidity_1000hPa:p',
       'sfc_pressure:hPa', 'snow_density:kgm3', 'snow_depth:cm',
       'snow_drift:idx', 'snow_melt_10min:mm', 'snow_water:kgm2',
       'sun_azimuth:d', 'sun_elevation:d', 'super_cooled_liquid_water:kgm2',
       't_1000hPa:K', 'total_cloud_cover:p', 'visibility:m',
       'wind_speed_10m:ms', 'wind_speed_u_10m:ms', 'wind_speed_v_10m:ms',
       'wind_speed_w_1000hPa:ms']

column_trans = ColumnTransformer(
    [('categories', OneHotEncoder(dtype='int'), cat_cols),
     ('numerical', MinMaxScaler(), num_cols)],
     remainder='drop', verbose_feature_names_out=False)

X_train, targets = get_training_data(data, features)

whole_model_pipeline.fit(X_train, targets)

X_test = get_input_data()
predictions = whole_model_pipeline.predict(X_test[features])

submission = prepare_submission(X_test, predictions)

submission.to_csv('stacked_forests_engineered.csv', index=False)
