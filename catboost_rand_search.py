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

print("randomized search for locA")
# Define the parameter grid for the randomized search
param_distA = {
    'cat_boost__iterations': randint(100, 1000),  # Number of boosting iterations
    'cat_boost__learning_rate': uniform(0.01, 0.3),  # Step size shrinkage used to prevent overfitting
    'cat_boost__depth': randint(1, 10),  # Depth of the trees
    'cat_boost__l2_leaf_reg': uniform(1, 10),  # L2 regularization term on weights
    'cat_boost__border_count': randint(1, 255),  # The number of splits for numerical features
}

# Create a randomized search object with the specified parameters and cross-validation
random_searchA = RandomizedSearchCV(
    estimator=locA_pipeline,
    param_distributions=param_distA,
    n_iter=100,  # Number of parameter settings that are sampled
    scoring='neg_mean_squared_error',  # Use negative mean squared error as the evaluation metric
    cv=3,  # Number of cross-validation folds
    random_state=42,  # Random seed for reproducibility
    n_jobs=-1,  # Use all available CPU cores
    verbose=1  # Print progress messages
)

# Perform the randomized search on your data and labels
random_searchA.fit(X_train_a, targets_a)

# Print the best parameters and corresponding score
print("Best Parameters: ", random_searchA.best_params_)
print("Best Negative Mean Squared Error: ", random_searchA.best_score_)


print("randomized search for locB")
# Define the parameter grid for the randomized search
param_distB = {
    'cat_boost__iterations': randint(100, 1000),  # Number of boosting iterations
    'cat_boost__learning_rate': uniform(0.01, 0.3),  # Step size shrinkage used to prevent overfitting
    'cat_boost__depth': randint(1, 10),  # Depth of the trees
    'cat_boost__l2_leaf_reg': uniform(1, 10),  # L2 regularization term on weights
    'cat_boost__border_count': randint(1, 255),  # The number of splits for numerical features
}

# Create a randomized search object with the specified parameters and cross-validation
random_searchB = RandomizedSearchCV(
    estimator=locB_pipeline,
    param_distributions=param_distB,
    n_iter=100,  # Number of parameter settings that are sampled
    scoring='neg_mean_squared_error',  # Use negative mean squared error as the evaluation metric
    cv=3,  # Number of cross-validation folds
    random_state=42,  # Random seed for reproducibility
    n_jobs=-1,  # Use all available CPU cores
    verbose=1  # Print progress messages
)

# Perform the randomized search on your data and labels
random_searchB.fit(X_train_b, targets_b)

# Print the best parameters and corresponding score
print("Best Parameters: ", random_searchB.best_params_)
print("Best Negative Mean Squared Error: ", random_searchB.best_score_)

print("randomized search for locC")
# Define the parameter grid for the randomized search
param_distC = {
    'cat_boost__iterations': randint(100, 1000),  # Number of boosting iterations
    'cat_boost__learning_rate': uniform(0.01, 0.3),  # Step size shrinkage used to prevent overfitting
    'cat_boost__depth': randint(1, 10),  # Depth of the trees
    'cat_boost__l2_leaf_reg': uniform(1, 10),  # L2 regularization term on weights
    'cat_boost__border_count': randint(1, 255),  # The number of splits for numerical features
}

# Create a randomized search object with the specified parameters and cross-validation
random_searchC = RandomizedSearchCV(
    estimator=locC_pipeline,
    param_distributions=param_distC,
    n_iter=100,  # Number of parameter settings that are sampled
    scoring='neg_mean_squared_error',  # Use negative mean squared error as the evaluation metric
    cv=3,  # Number of cross-validation folds
    random_state=42,  # Random seed for reproducibility
    n_jobs=-1,  # Use all available CPU cores
    verbose=1  # Print progress messages
)

# Perform the randomized search on your data and labels
random_searchC.fit(X_train_c, targets_c)

# Print the best parameters and corresponding score
print("Best Parameters: ", random_searchC.best_params_)
print("Best Negative Mean Squared Error: ", random_searchC.best_score_)