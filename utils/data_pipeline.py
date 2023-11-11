from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from utils.feature_engineering import cyclic_encoding, feature_interaction
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
    


    def transform(self, X):
        X_copy = X.copy()

        # # add moth
        # X_copy['month'] = X_copy['time'].apply(lambda x: x.month)

        # # add hour
        # X_copy['hour'] = X_copy['time'].apply(lambda x: x.hour)

        X_copy = cyclic_encoding(X_copy)
        X_copy = feature_interaction(X_copy)
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