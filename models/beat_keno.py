
import pandas as pd
import numpy as np
from utils.read_data import load_data, get_train_targets, get_test_data, prepare_submission
from utils.generate_run_name import generate_run_name
from utils.data_pipeline import ColumnDropper, FeatureAdder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import catboost as cb
import warnings
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import logging 
warnings.filterwarnings("ignore", category=FutureWarning)

def fuck_keno(model_name='keno-is-down'):
    logger = logging.getLogger()
    logger.info('Processing data')
    run_name = generate_run_name()
    logger.info(f'Model name: {model_name}')
    data_a, data_b, data_c = load_data(mean=True, remove_out=True, roll_avg=True)

    X_train_a, targets_a = get_train_targets(data_a)
    X_train_b, targets_b = get_train_targets(data_b)
    X_train_c, targets_c = get_train_targets(data_c)

    X_test_a, X_test_b, X_test_c = get_test_data(mean=True, roll_avg=True)


    drop_cols = ['time', 'elevation:m', 'fresh_snow_1h:cm', 'ceiling_height_agl:m', 'snow_density:kgm3', 
                'wind_speed_w_1000hPa:ms', 'snow_drift:idx', 'fresh_snow_3h:cm', 'is_in_shadow:idx', 'dew_or_rime:idx', 'fresh_snow_6h:cm', 'prob_rime:p'] # this second line is columns with feature importance == 0

    logger.info('Done preprocessing data')
    data_process_pipeline = Pipeline([
        ('add_features', FeatureAdder()),
        ('drop_cols', ColumnDropper(drop_cols=drop_cols)),
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
        ('standar', StandardScaler()),
    ])

    lgb_params_jita_mira = {
        'learning_rate': 0.05,
        'extra_trees': True,
        'num_threads': 24,
        'objective': 'regression',
        'verbose': -1,
        'metric': 'rmse',
        'seed': 0,
        'num_iterations': 10000,
        'early_stopping_round': None
    }

    catboost_params_11_11_00_14 = {
        'border_count': 157,
        'depth': 13,
        'iterations': 828,
        'l2_leaf_reg': 7.677745179031975,
        'learning_rate': 0.012997359346271088
    }
    # Create the LGBM model
    #lgb_regressor = lgb.LGBMRegressor(**lgb_params_jita_mira)
    base_modelsA = [
        ('cat_boost1', cb.CatBoostRegressor(**catboost_params_11_11_00_14)), #andrea gjør søk
        ('cat_boost2', cb.CatBoostRegressor(random_state=2, silent=True, depth=10)),
        ('xgb_reg1', XGBRegressor(random_state=18, eval_metric="mae")), #Taheera gjør søk
        ('xgb_reg2', XGBRegressor(random_state=42)),
        ('xgb_reg3', XGBRegressor(random_state=16, eval_metric="mae")),
        ('cat_boost3', cb.CatBoostRegressor(random_state=3, silent=True)),
        ('cat_boost4', cb.CatBoostRegressor(random_state=32, silent=True, objective="MAE", depth=10)), #lagt til
        ('cat_boost5', cb.CatBoostRegressor(random_state=100, silent=True, objective="RMSE", depth=10)), #lagt til
        #('lgbm', lgb_regressor)
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

    logger.info("Training location A model")
    modelA_pipeline.fit(X_train_a, targets_a)
    pred_a = modelA_pipeline.predict(X_test_a.drop(columns=["id", "prediction", "location"]))

    logger.info("Training location B model")
    modelB_pipeline.fit(X_train_b, targets_b)
    pred_b = modelB_pipeline.predict(X_test_b.drop(columns=["id", "prediction", "location"]))

    logger.info("Training location C model")
    modelC_pipeline.fit(X_train_c, targets_c)
    pred_c = modelC_pipeline.predict(X_test_c.drop(columns=["id", "prediction", "location"]))

    prepare_submission(X_test_a, X_test_b, X_test_c, pred_a, pred_b, pred_c, run_name)