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

class FeatureAdder(BaseEstimator, TransformerMixin):
    """Adds features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = rolling_average(X_copy)
        X_copy = cyclic_encoding(X.copy)
        X_copy = add_custom_features(X_copy)
        X_copy = remove_ouliers(X_copy)
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