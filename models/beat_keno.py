
import pandas as pd
import numpy as np
from utils.read_data import load_data, get_train_targets, get_test_data, prepare_submission
from utils.generate_run_name import generate_run_name
from utils.data_pipeline import ColumnDropper
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import catboost as cb
import lightgbm as lgb
import warnings
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
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
    lgb_regressor = lgb.LGBMRegressor(**lgb_params_jita_mira)
    base_modelsA = [
        ('cat_boost1', cb.CatBoostRegressor(**catboost_params_11_11_00_14)), #andrea gjør søk
        ('cat_boost2', cb.CatBoostRegressor(random_state=2, silent=True, depth=10)),
        ('xgb_reg1', XGBRegressor(random_state=18, eval_metric="mae")), #Taheera gjør søk
        ('xgb_reg2', XGBRegressor(random_state=42)),
        ('xgb_reg3', XGBRegressor(random_state=16, eval_metric="mae")),
        ('cat_boost3', cb.CatBoostRegressor(random_state=3, silent=True)),
        ('cat_boost4', cb.CatBoostRegressor(random_state=32, silent=True, objective="MAE", depth=10)), #lagt til
        ('cat_boost5', cb.CatBoostRegressor(random_state=100, silent=True, objective="RMSE", depth=10)), #lagt til
        ('lgbm', lgb_regressor)
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

class FeatureAdder(BaseEstimator, TransformerMixin):
    """Adds features."""

    def __init__(self, drop_cols = []):
        self.drop_cols = drop_cols

    def fit(self, X, y=None):
        return self
    
    def cyclic_encoding(self, df):
        df['time'] = pd.to_datetime(df['time'])
        df['normalized_time'] = (df['time'].dt.hour + df['time'].dt.minute / 60 + df['time'].dt.second / 3600) / 24.0
        df['sine_encoded'] = np.sin(2 * np.pi * df['normalized_time'])
        df['cosine_encoded'] = np.cos(2 * np.pi * df['normalized_time'])

        month = df['time'].dt.month
        df['sine_encoded_month'] = np.sin(2 * np.pi * month)
        df['cosine_encoded_month'] = np.cos(2 * np.pi * month)

        df.drop('normalized_time', axis=1, inplace=True)
        return df

    def transform(self, X):
        X_copy = X.copy()

        # # add moth
        # X_copy['month'] = X_copy['time'].apply(lambda x: x.month)

        # # add hour
        # X_copy['hour'] = X_copy['time'].apply(lambda x: x.hour)

        X_copy = self.cyclic_encoding(X_copy)

        # -- additive effects:
        X_copy["sun_rad_1"] = (X_copy['sun_azimuth:d'] * X_copy['direct_rad:W']) / 1000000
        X_copy["sun_rad_2"] = (X_copy['sun_elevation:d'] * X_copy['direct_rad:W']) / 1000000
        #X_copy["sun_wind_1"] = (X_copy['wind_speed_10m:ms'] * X_copy['direct_rad:W']) / 1000
        X_copy["sun_wind_2"] = (X_copy['wind_speed_10m:ms'] * X_copy['diffuse_rad:W']) / 1000
        X_copy["temp_sun"] = (X_copy['t_1000hPa:K'] * X_copy['sun_azimuth:d'])/1000
        X_copy["rad_day_1"] = (X_copy['is_day:idx'] * X_copy['diffuse_rad:W']) / 1000
        X_copy['mult_coulds'] = (X_copy['clear_sky_rad:W'] * X_copy['cloud_base_agl:m']) / 100000

        #X_copy["dirrad_airdensity"] = (X_copy['direct_rad:W'] * X_copy['air_density_2m:kgm3'])/1000 #unsure
        X_copy["ratio_rad1"] = (X_copy['direct_rad:W'] / X_copy['diffuse_rad:W']) # good one!
        #X_copy["diffrad_airdensity"] = (X_copy['diffuse_rad:W'] * X_copy['air_density_2m:kgm3'])/1000 #unsure
        X_copy["temp_rad_1"] = (X_copy['t_1000hPa:K'] * X_copy['direct_rad:W'])/1000

        # X_copy["ratio_rad1"] = (X_copy['direct_rad:W'] / X_copy['diffuse_rad:W']) # good one!
        # X_copy["temp_rad_1"] = (X_copy['t_1000hPa:K'] * X_copy['direct_rad:W'])/1000

        return X_copy