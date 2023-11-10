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
from featureadder import FeatureAdder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Suppress all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

data_a, data_b, data_c = load_data(mean=True, remove_out=True, roll_avg=True)

X_train_a, targets_a = get_train_targets(data_a)
X_train_b, targets_b = get_train_targets(data_b)
X_train_c, targets_c = get_train_targets(data_c)

X_test_a, X_test_b, X_test_c = get_test_data(mean=True, roll_avg=True)


drop_cols = ['time', 'elevation:m', 'fresh_snow_1h:cm', 'ceiling_height_agl:m', 'snow_density:kgm3', 
             'wind_speed_w_1000hPa:ms', 'snow_drift:idx', 'fresh_snow_3h:cm', 'is_in_shadow:idx', 'dew_or_rime:idx', 'fresh_snow_6h:cm', 'prob_rime:p'] # this second line is columns with feature importance == 0

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
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ('standar', StandardScaler()),
])


base_modelsA = [
    ('cat_boost1', cb.CatBoostRegressor(random_state=1, silent=True, objective="MAE", depth=10)),
    ('cat_boost2', cb.CatBoostRegressor(random_state=2, silent=True, depth=10)),
    ('xgb_reg1', XGBRegressor(random_state=12, eval_metric="mae", colsample_bytree=0.6396572695576039, gamma=0.3122839968432516, learning_rate=0.030962102428926407, max_depth=8, n_estimators=258, subsample=0.9202906451199873)),
    ('xgb_reg2', XGBRegressor(random_state=42)),
    ('xgb_reg3', XGBRegressor(random_state=16, eval_metric="mae")),
    ('cat_boost3', cb.CatBoostRegressor(random_state=3, silent=True)),
]

base_modelsB = [
    ('cat_boost1', cb.CatBoostRegressor(random_state=1, silent=True, objective="MAE", depth=10)),
    ('cat_boost2', cb.CatBoostRegressor(random_state=2, silent=True, depth=10)),
    ('xgb_reg1', XGBRegressor(random_state=12, eval_metric="mae")),
    ('xgb_reg2', XGBRegressor(random_state=42)),
    ('cat_boost3', cb.CatBoostRegressor(random_state=3, silent=True)),
]

base_modelsC = [
    ('cat_boost1', cb.CatBoostRegressor(random_state=1, silent=True, objective="MAE", depth=10)),
    ('cat_boost2', cb.CatBoostRegressor(random_state=2, silent=True, depth=10)),
    ('xgb_reg1', XGBRegressor(random_state=12, eval_metric="mae")),
    ('xgb_reg2', XGBRegressor(random_state=42)),
    ('cat_boost3', cb.CatBoostRegressor(random_state=3, silent=True)),
]

# Define meta-learner
meta_learnerA = LinearRegression()
meta_learnerB = LinearRegression()
meta_learnerC = LinearRegression()

# Create the stacking regressor
stacked_modelA = StackingRegressor(estimators=base_modelsA, final_estimator=meta_learnerA)
stacked_modelB = StackingRegressor(estimators=base_modelsB, final_estimator=meta_learnerB)
stacked_modelC = StackingRegressor(estimators=base_modelsC, final_estimator=meta_learnerC)

modelA_pipeline = Pipeline([
    ('data_process', data_process_pipeline),
    ('stacked_model', stacked_modelA)
])

modelB_pipeline = Pipeline([
    ('data_process', data_process_pipeline),
    ('stacked_model', stacked_modelB)
])

modelC_pipeline = Pipeline([
    ('data_process', data_process_pipeline),
    ('stacked_model', stacked_modelC)
])

print("training location A model")
modelA_pipeline.fit(X_train_a, targets_a)
pred_a = modelA_pipeline.predict(X_test_a.drop(columns=["id", "prediction", "location"]))

print("training location B model")
modelB_pipeline.fit(X_train_b, targets_b)
pred_b = modelB_pipeline.predict(X_test_b.drop(columns=["id", "prediction", "location"]))

print("training location C model")
modelC_pipeline.fit(X_train_c, targets_c)
pred_c = modelC_pipeline.predict(X_test_c.drop(columns=["id", "prediction", "location"]))

submission = prepare_submission(X_test_a, X_test_b, X_test_c, pred_a, pred_b, pred_c)
submission['prediction'] = submission['prediction'].apply(lambda x: 0 if x < 0.1 else x)

submission.to_csv('submissions/10_nov_1030.csv', index=False)