from utils.data_preprocess import ColumnDropper, FeatureAdder
from utils.data_preprocess_location import get_train_targets, remove_ouliers, load_data, get_test_data, prepare_submission
import numpy as np
import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
import catboost as cb

def stacked_catboost():
    logger = logging.getLogger()
    logger.info('Processing data')
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
        ('cat_boost1', cb.CatBoostRegressor(random_state=42, silent=True, border_count=86, depth=9, iterations=384, l2_leaf_reg=2.1607264050691626, learning_rate=0.023800792606525824)),
        ('cat_boost2', cb.CatBoostRegressor(random_state=42, silent=True, border_count=81, depth=8, iterations=704, l2_leaf_reg=9.448753109694545, learning_rate=0.01698158072074776)),
        ('cat_boost3', cb.CatBoostRegressor(random_state=42, silent=True, border_count=81, depth=8, iterations=704, l2_leaf_reg=9.448753109694545, learning_rate=0.01698158072074776)),
        ('cat_boost4', cb.CatBoostRegressor(random_state=42, silent=True))
    ]

    # Define meta-learner
    meta_learner = LinearRegression()

    # Create the stacking regressor
    stacked_model = StackingRegressor(estimators=base_models, final_estimator=meta_learner)

    #NOTE: can instead of using the sacked model just run a single model below:
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

    prepare_submission(X_test_a, X_test_b, X_test_c, pred_a, pred_b, pred_c)
