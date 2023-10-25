from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from utils.generate_run_name import generate_run_name
from utils.log_model import fetch_logged_data, write_to_file

import mlflow
import time

def random_forest(num, cat, X_train, y_train):
    start_time = time.time()  # <- Start the timer

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num),
            ('cat', categorical_transformer, cat)])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())])

    run_name = generate_run_name()
    with mlflow.start_run(run_name=run_name) as run:
        # Perform 5-fold cross-validation and calculate the metrics for each fold
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        
        # Convert negative MSE to positive (optional, depends on your preference)
        mse_values = -scores
        
        # Log the metrics
        for i, mse in enumerate(mse_values):
            mlflow.log_metric(f'MSE_fold_{i}', mse)
        
        # Fetch and print logged data
        params, metrics, tags, artifacts, log_time = fetch_logged_data(run.info.run_id)
        

    logged_data = {
        'name': 'Linear regression',
        'time': log_time,
        'run_name': run_name,
        'params': params,
        'metrics': metrics,
        'tags': tags, 
        'artifacts': artifacts,
    }

    write_to_file(logged_data, start_time)