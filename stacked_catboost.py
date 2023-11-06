import pandas as pd
import numpy as np
from functions import load_data, get_train_targets, get_test_data, prepare_submission, remove_ouliers
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import catboost as cb


data_a, data_b, data_c = load_data()

data_a = remove_ouliers(data_a)
data_b = remove_ouliers(data_b)
data_c = remove_ouliers(data_c)

X_train_a, targets_a = get_train_targets(data_a)
X_train_b, targets_b = get_train_targets(data_b)
X_train_c, targets_c = get_train_targets(data_c)

X_test_a, X_test_b, X_test_c = get_test_data()

drop_cols = ['time', 'date_calc', 'elevation:m', 'fresh_snow_1h:cm',  
             'wind_speed_u_10m:ms', 'wind_speed_v_10m:ms', 'wind_speed_w_1000hPa:ms', 'prob_rime:p',
             'fresh_snow_12h:cm','fresh_snow_24h:cm', 'fresh_snow_6h:cm', 'super_cooled_liquid_water:kgm2']


class FeatureAdder(BaseEstimator, TransformerMixin):
    """Adds features."""

    def __init__(self, drop_cols = []):
        self.drop_cols = drop_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['month'] = X_copy['time'].apply(lambda x: x.month)
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
    ('add_month', FeatureAdder()),
    ('drop_cols', ColumnDropper(drop_cols=drop_cols)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
])

# Define base models
base_models = [
    # ('cat_boost1', cb.CatBoostRegressor(random_state=1, silent=True)),
    # ('cat_boost2', cb.CatBoostRegressor(random_state=2, silent=True)),
    # ('cat_boost3', cb.CatBoostRegressor(random_state=3, silent=True)),
    # ('cat_boost4', cb.CatBoostRegressor(random_state=4, silent=True)),
    ('cat_boost1', cb.CatBoostRegressor(random_state=42, silent=True, border_count=86, depth=9, iterations=384, l2_leaf_reg=2.1607264050691626, learning_rate=0.023800792606525824)), #rand search locA seed 42
    ('cat_boost2', cb.CatBoostRegressor(random_state=42, silent=True, border_count=81, depth=8, iterations=704, l2_leaf_reg=9.448753109694545, learning_rate=0.01698158072074776)), # rand search locB seed 42
    ('cat_boost3', cb.CatBoostRegressor(random_state=42, silent=True, border_count=81, depth=8, iterations=704, l2_leaf_reg=9.448753109694545, learning_rate=0.01698158072074776)), # rand search locC seed 42
    ('cat_boost4', cb.CatBoostRegressor(random_state=42, silent=True)),
    # ('cat_boost5', cb.CatBoostRegressor(random_state=42, silent=True, border_count=216, depth=9, iterations=283, l2_leaf_reg=6.23940646995615, learning_rate=0.04453689534724951)), # rand search locA seed 12
    # ('cat_boost6', cb.CatBoostRegressor(random_state=42, silent=True, border_count=250, depth=9, iterations=124, l2_leaf_reg=7.1343511453892265, learning_rate=0.04716464647756951)), # rand search locB seed 12
    # ('cat_boost7', cb.CatBoostRegressor(random_state=42, silent=True, border_count=138, depth=7, iterations=239, l2_leaf_reg=5.292895897731217, learning_rate=0.04698405236342185)), # rand search locc seed 12
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

submission.to_csv('submissions/stacked_catboost_.csv', index=False)