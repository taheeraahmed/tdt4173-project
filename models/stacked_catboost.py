from utils.data_pipeline import ColumnDropper, FeatureAdder
from utils.read_data import get_train_targets, load_data, get_test_data, prepare_submission
from utils.generate_run_name import generate_run_name
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
import catboost as cb

def stacked_catboost(model_name='stacked-catboost'):
    logger = logging.getLogger()
    logger.info('Processing data')
    data_a, data_b, data_c = load_data()

    X_train_a, y_a = get_train_targets(data_a)
    X_train_b, y_b = get_train_targets(data_b)
    X_train_c, y_c = get_train_targets(data_c)

    X_test_a, X_test_b, X_test_c = get_test_data()

    drop_cols = ['time', 'date_calc']

    data_process_pipeline = Pipeline([
        ('add_month', FeatureAdder()),
        ('drop_cols', ColumnDropper(drop_cols=drop_cols)),
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=0)),
    ])
    
    run_name = generate_run_name()

    logger.info(f'Model name: {model_name}')

    # Define base models
    base_models = [
        ('cat_boost1', cb.CatBoostRegressor(random_state=42, silent=True)),
        ('cat_boost2', cb.CatBoostRegressor(random_state=42, silent=True)),
        ('cat_boost3', cb.CatBoostRegressor(random_state=42, silent=True)),
        ('cat_boost4', cb.CatBoostRegressor(random_state=42, silent=True))
    ]

    # Define meta-learner
    meta_learner = LinearRegression()

    # Create the stacking regressor
    stacked_model = StackingRegressor(estimators=base_models, final_estimator=meta_learner)

    whole_model_pipeline = Pipeline([
        ('data_process', data_process_pipeline),
        ('stacked_model', stacked_model)
    ])

    logger.info("training location A model")
    whole_model_pipeline.fit(X_train_a, y_a)
    pred_a = whole_model_pipeline.predict(X_test_a.drop(columns=["id", "prediction", "location"]))

    logger.info("training location B model")
    whole_model_pipeline.fit(X_train_b, y_b)
    pred_b = whole_model_pipeline.predict(X_test_b.drop(columns=["id", "prediction", "location"]))

    logger.info("training location C model")
    whole_model_pipeline.fit(X_train_c, y_c)
    pred_c = whole_model_pipeline.predict(X_test_c.drop(columns=["id", "prediction", "location"]))

    prepare_submission(X_test_a, X_test_b, X_test_c, pred_a, pred_b, pred_c, run_name=run_name)
