from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from utils.generate_run_name import generate_run_name
from utils.log_model import fetch_logged_data, write_to_file
from utils.evaluate import prepare_submission, get_input_data

import time
import mlflow

def grid_search(num, cat, X_train, y_train, model_name="grid-search"):
    """
    Train a random forest model using given training data and log the results using MLflow.

    Parameters:
    - num (list): List of numerical feature names.
    - cat (list): List of categorical feature names.
    - X_train (DataFrame): Training data features.
    - y_train (Series): Training data target values.

    Returns:
    None. The function logs the training results using MLflow and writes the logged data to a file.
    """
    start_time = time.time()  
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num)])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())])

    # Define the parameter grid for grid search
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],  # Example values, adjust as needed
        'regressor__max_depth': [None, 10, 20],    # Example values, adjust as needed
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    
    run_name = generate_run_name()
    with mlflow.start_run(run_name=run_name) as run:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        mlflow.sklearn.log_model(best_model, "Random forest")
        
        # Log the best parameters found by grid search
        mlflow.log_params(grid_search.best_params_)
        
        # Perform 5-fold cross-validation and calculate the metrics for each fold
        scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        
        # Convert negative MSE to positive (optional, depends on your preference)
        mse_values = -scores
        
        # Log the metrics
        for i, mse in enumerate(mse_values):
            mlflow.log_metric(f'MSE_fold_{i}', mse)
        
        # Fetch and print logged data
        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
        
    logged_data = {
        'name': model_name,
        'start_time': start_time,
        'run_name': run_name,
        'params': params,
        'metrics': metrics,
        'tags': tags, 
        'artifacts': artifacts,
    }

    write_to_file(logged_data)

    X_test = get_input_data(drop_time_date=True)
    pred = best_model.predict(X_test)
    prepare_submission(X_test, pred, run_name)
