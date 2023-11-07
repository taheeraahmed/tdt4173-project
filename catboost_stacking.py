import pandas as pd
import numpy as np
from functions import load_data, get_train_targets, get_test_data, prepare_submission, remove_ouliers
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import catboost as cb
from scipy.stats import uniform, randint
import warnings
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

# Suppress all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

data_a, data_b, data_c = load_data(mean=True, remove_out=True, roll_avg=True)

X_train_a, targets_a = get_train_targets(data_a)
X_train_b, targets_b = get_train_targets(data_b)
X_train_c, targets_c = get_train_targets(data_c)

X_test_a, X_test_b, X_test_c = get_test_data(mean=True, roll_avg=True)


drop_cols = ['time', 'elevation:m', 'fresh_snow_1h:cm', 'ceiling_height_agl:m', 'snow_density:kgm3', 
             'wind_speed_w_1000hPa:ms', 'snow_drift:idx', 'fresh_snow_3h:cm', 'is_in_shadow:idx', 'dew_or_rime:idx', 'fresh_snow_6h:cm', 'prob_rime:p'] # this second line is columns with feature importance == 0


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

data_process_pipeline = Pipeline([
    ('add_features', FeatureAdder()),
    ('drop_cols', ColumnDropper(drop_cols=drop_cols)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
])

# Define base models
base_models = [
    # ('cat_boost1', cb.CatBoostRegressor(random_state=42, silent=True)), #, border_count=86, depth=9, iterations=384, l2_leaf_reg=2.1607264050691626, learning_rate=0.023800792606525824)), #rand search locA seed 42
    # ('cat_boost2', cb.CatBoostRegressor(random_state=42, silent=True)), #, border_count=81, depth=8, iterations=704, l2_leaf_reg=9.448753109694545, learning_rate=0.01698158072074776)), # rand search locB seed 42
    # ('cat_boost3', cb.CatBoostRegressor(random_state=42, silent=True)), #, border_count=81, depth=8, iterations=704, l2_leaf_reg=9.448753109694545, learning_rate=0.01698158072074776)), # rand search locC seed 42
    ('cat_boost4', cb.CatBoostRegressor(random_state=42, silent=True)),
    ('cat_boost5', cb.CatBoostRegressor(random_state=42, silent=True, border_count=216, depth=9, iterations=283, l2_leaf_reg=6.23940646995615, learning_rate=0.04453689534724951)), #best locA model
    ('cat_boost6', cb.CatBoostRegressor(random_state=42, silent=True, border_count=138, depth=7, iterations=239, l2_leaf_reg=5.292895897731217, learning_rate=0.04698405236342185)), #best locB model
    ('cat_boost7', cb.CatBoostRegressor(random_state=42, silent=True, border_count=138, depth=7, iterations=239, l2_leaf_reg=5.292895897731217, learning_rate=0.04698405236342185)), #best locC model
]

# Define meta-learner
meta_learner = LinearRegression()

# Create the stacking regressor
stacked_model = StackingRegressor(estimators=base_models, final_estimator=meta_learner)

#NOTE: can instead of using the stacked model just run a single model below:
whole_model_pipeline = Pipeline([
    ('data_process', data_process_pipeline),
    ('stacked_model', stacked_model)
])

print("training location A model")
whole_model_pipeline.fit(X_train_a, targets_a)
pred_a = whole_model_pipeline.predict(X_test_a.drop(columns=["id", "prediction", "location"]))

print("training location B model")
whole_model_pipeline.fit(X_train_b, targets_b)
pred_b = whole_model_pipeline.predict(X_test_b.drop(columns=["id", "prediction", "location"]))

print("training location C model")
whole_model_pipeline.fit(X_train_c, targets_c)
pred_c = whole_model_pipeline.predict(X_test_c.drop(columns=["id", "prediction", "location"]))

submission = prepare_submission(X_test_a, X_test_b, X_test_c, pred_a, pred_b, pred_c)
submission['prediction'] = submission['prediction'].apply(lambda x: 0 if x < 0.05 else x)

submission.to_csv('submissions/stacked_catboost_7_nov_rand_search_params.csv', index=False)