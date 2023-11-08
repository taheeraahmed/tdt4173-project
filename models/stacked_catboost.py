from utils.read_data import get_train_targets, load_data, get_test_data, prepare_submission
from utils.generate_run_name import generate_run_name
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
import catboost as cb
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from autogluon.tabular import TabularPredictor
import logging

class AutoGluonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = TabularPredictor.load(self.model_path)

    def fit(self, X, y=None):
        # AutoGluon models are pre-trained, so `fit` doesn't need to do anything
        return self

    def transform(self, X):
        # Use the AutoGluon model to make predictions and return them
        predictions = self.model.predict(X, as_pandas=True)
        return np.atleast_2d(predictions).T  # Ensure the output is 2D


def stacked_catboost(model_name='stacked-catboost'):
    logger = logging.getLogger()
    logger.info('Processing data')
    drop_cols = ['time', 'elevation:m', 'fresh_snow_1h:cm', 'ceiling_height_agl:m', 'snow_density:kgm3', 
             'wind_speed_w_1000hPa:ms', 'snow_drift:idx', 'fresh_snow_3h:cm', 'is_in_shadow:idx', 'dew_or_rime:idx', 'fresh_snow_6h:cm', 'prob_rime:p'] # this second line is columns with feature importance == 0

    data_a, data_b, data_c = load_data(mean=True, remove_out=True, roll_avg=True, cust_feat=True, drop_cols=drop_cols)
    X_test_a, X_test_b, X_test_c = get_test_data(mean=True, roll_avg=True, cust_feat=True)

    # Assuming get_train_targets is defined and returns the features and target
    X_train_a, y_a = get_train_targets(data_a)
    X_train_b, y_b = get_train_targets(data_b)
    X_train_c, y_c = get_train_targets(data_c)

    drop_cols = ['time', 'date_calc']
    
    run_name = generate_run_name()
    logger.info(f'Model name: {model_name}')

    # Define base models with the best parameters found for each location

    """
    cat_boost1: bayesian_search on params given feat_eng from 07.nov
    cat_boost2: from best kaggle submission
    """
    base_models_a = [
        ('cat_boost1', cb.CatBoostRegressor(
            bootstrap_type='Bayesian',
            border_count=255,
            depth=7,
            grow_policy='SymmetricTree',
            iterations=300,
            l2_leaf_reg=5.000000000000001,
            learning_rate=0.09957010602734577,
            min_data_in_leaf=1,
            random_strength=1e-09,
            silent=True
        )),
        ('cat_boost2', cb.CatBoostRegressor(random_state=42, silent=True, border_count=216, depth=9, iterations=283, l2_leaf_reg=6.23940646995615, learning_rate=0.04453689534724951)), #best locA model
        ('cat_boost3', cb.CatBoostRegressor(random_state=42, silent=True)),
        ('cat_boost4', cb.CatBoostRegressor(random_state=42, silent=True)),
    ]

    base_models_b = [
        ('cat_boost1', cb.CatBoostRegressor(
            bootstrap_type='MVS',
            border_count=200,
            depth=7,
            grow_policy='SymmetricTree',
            iterations=153,
            l2_leaf_reg=5.677917242722324,
            learning_rate=0.05426238595755628,
            min_data_in_leaf=3,
            random_strength=3.701054654379414e-07,
            silent=True
        )),
        ('cat_boost2', cb.CatBoostRegressor(random_state=42, silent=True, border_count=138, depth=7, iterations=239, l2_leaf_reg=5.292895897731217, learning_rate=0.04698405236342185)), #best locB model
        ('cat_boost3', cb.CatBoostRegressor(random_state=42, silent=True)),
        ('cat_boost4', cb.CatBoostRegressor(random_state=42, silent=True)),
    ]

    base_models_c = [
        ('cat_boost1', cb.CatBoostRegressor(
            bootstrap_type='Bernoulli',
            border_count=242,
            depth=5,
            grow_policy='Lossguide',
            iterations=290,
            l2_leaf_reg=8.850908338670033,
            learning_rate=0.02839226005231253,
            min_data_in_leaf=8,
            random_strength=0.2656511459585781,
            silent=True
        )),
        ('cat_boost2', cb.CatBoostRegressor(random_state=42, silent=True, border_count=138, depth=7, iterations=239, l2_leaf_reg=5.292895897731217, learning_rate=0.04698405236342185)), #best locC model
        ('cat_boost3', cb.CatBoostRegressor(random_state=42, silent=True)),
        ('cat_boost4', cb.CatBoostRegressor(random_state=42, silent=True)),
    ]

    # Define meta-learners for each location
    meta_learner_a = LinearRegression()
    meta_learner_b = LinearRegression()
    meta_learner_c = LinearRegression() 

    # Create a dictionary to store predictions
    predictions = {}

    # Loop through each location
    for loc, X_train, X_test, y, base_models, meta_learner in [
        ('A', X_train_a, X_test_a, y_a, base_models_a, meta_learner_a),
        ('B', X_train_b, X_test_b, y_b, base_models_b, meta_learner_b),
        ('C', X_train_c, X_test_c, y_c, base_models_c, meta_learner_c)
    ]:
        # Create the stacking regressor for the current location
        stacked_model = StackingRegressor(estimators=base_models, final_estimator=meta_learner)

        # Create the whole model pipeline for the current location
        whole_model_pipeline = Pipeline([
            ('stacked_model', stacked_model)
        ])

        logger.info(f"Training location {loc} model")
        whole_model_pipeline.fit(X_train, y)
        predictions[loc] = whole_model_pipeline.predict(X_test.drop(columns=["id", "prediction", "location"]))

    # Prepare submission using the predictions
    prepare_submission(X_test_a, X_test_b, X_test_c, predictions['A'], predictions['B'], predictions['C'], run_name=run_name)
