from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from utils.read_data import get_train_targets, load_data
from utils.generate_run_name import generate_run_name

import numpy as np
import mlflow
import logging


def param_search_randomized_catboost(model_name="param-search-random-catboost-for"):
    """
    
    """
    logger = logging.getLogger()

    logger.info('Processing data')

    drop_cols = ['time', 'elevation:m', 'fresh_snow_1h:cm', 'ceiling_height_agl:m', 'snow_density:kgm3', 
             'wind_speed_w_1000hPa:ms', 'snow_drift:idx', 'fresh_snow_3h:cm', 'is_in_shadow:idx', 'dew_or_rime:idx', 'fresh_snow_6h:cm', 'prob_rime:p'] # this second line is columns with feature importance == 0

    data_a, _,_ = load_data(mean_stats=True,  roll_avg=True, remove_out=True, cust_feat=True, drop_cols=drop_cols, cycle_encoding = True)
    X_train_a, y_train_a = get_train_targets(data_a)
    data_process_pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),
        ('standard', StandardScaler()),
    ])
    
    logger.info('Done processing data')

    search_space_catboost = {
        'cat_boost__iterations': randint(100, 1000),  # Number of boosting iterations
        'cat_boost__learning_rate': uniform(0.01, 0.3),  # Step size shrinkage used to prevent overfitting
        'cat_boost__depth': randint(5, 15),  # Depth of the trees
        'cat_boost__l2_leaf_reg': uniform(1, 10),  # L2 regularization term on weights
        'cat_boost__border_count': randint(1, 255),  # The number of splits for numerical features
    }


    run_name = generate_run_name()

    pipeline = Pipeline([
        ('data_process', data_process_pipeline), 
        ('cat_boost', CatBoostRegressor(random_state=4, silent=True, loss_function='MAE', objective='MAE'))
    ])
    
    logger.info("Run pipeline for location A")
    with mlflow.start_run(run_name=f'{run_name}_A') as run:
        logger.info("Fit for location 'A'")
        rand_search = RandomizedSearchCV(pipeline, search_space_catboost, cv=5, scoring='neg_mean_squared_error')                                # random state for replicability
        rand_search.fit(X_train_a, y_train_a)
        logger.info(f"{model_name}-A: {rand_search.best_params_}")