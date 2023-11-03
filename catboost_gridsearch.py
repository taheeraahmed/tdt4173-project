import pandas as pd
import numpy as np
from functions import load_data, get_train_targets, get_test_data, prepare_submission, remove_ouliers
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
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
    ('drop_cols', ColumnDropper(drop_cols=drop_cols)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
])

locA_pipeline = Pipeline([
    ('data_process', data_process_pipeline),
    ('cat_boost', cb.CatBoostRegressor(random_state=42, silent=True))
])

locB_pipeline = Pipeline([
    ('data_process', data_process_pipeline),
    ('cat_boost', cb.CatBoostRegressor(random_state=42, silent=True))
])

locC_pipeline = Pipeline([
    ('data_process', data_process_pipeline),
    ('cat_boost', cb.CatBoostRegressor(random_state=42, silent=True))
])

param_gridA = {
    'cat_boost__iterations': [175, 200, 225],
    'cat_boost__learning_rate': [0.01, 0.05],
    'cat_boost__depth': [7, 8, 9],
    'cat_boost__l2_leaf_reg': [3, 4, 5]
}

print("grid search for locA")
grid_searchA = GridSearchCV(estimator=locA_pipeline, param_grid=param_gridA, cv=3)
grid_searchA.fit(X_train_a, targets_a)

# Print the best parameters and corresponding score
print("Best Parameters: ", grid_searchA.best_params_)
print("Best Score: ", grid_searchA.best_score_)

param_gridB = {
    'cat_boost__iterations': [175, 200, 225],
    'cat_boost__learning_rate': [0.05],
    'cat_boost__depth': [6],
    'cat_boost__l2_leaf_reg': [0.4, 0.5, 0.6]
}


print("grid search for locB")
grid_searchB = GridSearchCV(estimator=locB_pipeline, param_grid=param_gridB, cv=3)
grid_searchB.fit(X_train_b, targets_b)

# Print the best parameters and corresponding score
print("Best Parameters: ", grid_searchB.best_params_)
print("Best Score: ", grid_searchB.best_score_)

param_gridC = {
    'cat_boost__iterations': [275, 300, 325],
    'cat_boost__learning_rate': [0.05],
    'cat_boost__depth': [4],
    'cat_boost__l2_leaf_reg': [0.75, 1, 1.25]
}

print("grid search for locC")
grid_searchC = GridSearchCV(estimator=locA_pipeline, param_grid=param_gridC, cv=3)
grid_searchC.fit(X_train_c, targets_c)

# Print the best parameters and corresponding score
print("Best Parameters: ", grid_searchC.best_params_)
print("Best Score: ", grid_searchC.best_score_)
