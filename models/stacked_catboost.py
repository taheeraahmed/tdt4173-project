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
    data_a, data_b, data_c = load_data(mean=True, remove_out=True, roll_avg=True)
    X_test_a, X_test_b, X_test_c = get_test_data(mean=True, roll_avg=True)

    # Assuming get_train_targets is defined and returns the features and target
    X_train_a, y_a = get_train_targets(data_a)
    X_train_b, y_b = get_train_targets(data_b)
    X_train_c, y_c = get_train_targets(data_c)

    drop_cols = ['time', 'date_calc']

    # Define the data processing pipeline (common for all locations)
    data_process_pipeline = Pipeline([
        # ('add_month', FeatureAdder()), in read data:)
        ('drop_cols', ColumnDropper(drop_cols=drop_cols)),
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=0)),
    ])
    
    run_name = generate_run_name()
    logger.info(f'Model name: {model_name}')

    # Define base models with the best parameters found for each location

    """
    cat_boost1: bayesian_search on params given feat_eng from 07.nov
    
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
        # ... Add more models if you're using an ensemble of multiple CatBoost models
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
        # ... Add more models if needed
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
        # ... Add more models if needed
    ]

    # Define meta-learners for each location
    meta_learner_a = LinearRegression()
    meta_learner_b = LinearRegression()
    meta_learner_c = LinearRegression() 

    # Create a dictionary to store predictions
    predictions = {}

    # Loop through each location
    for loc, X_train, y, base_models, meta_learner in [
        ('A', X_train_a, y_a, base_models_a, meta_learner_a),
        ('B', X_train_b, y_b, base_models_b, meta_learner_b),
        ('C', X_train_c, y_c, base_models_c, meta_learner_c)
    ]:
        # Create the stacking regressor for the current location
        stacked_model = StackingRegressor(estimators=base_models, final_estimator=meta_learner)

        # Create the whole model pipeline for the current location
        whole_model_pipeline = Pipeline([
            ('data_process', data_process_pipeline),
            ('stacked_model', stacked_model)
        ])

        logger.info(f"training location {loc} model")
        whole_model_pipeline.fit(X_train, y)
        X_test = locals()[f'X_test_{loc}'].drop(columns=["id", "prediction", "location"])
        predictions[loc] = whole_model_pipeline.predict(X_test)

    # Prepare submission using the predictions
    prepare_submission(X_test_a, X_test_b, X_test_c, predictions['A'], predictions['B'], predictions['C'], run_name=run_name)
