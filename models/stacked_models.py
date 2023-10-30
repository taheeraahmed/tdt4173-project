from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from utils.generate_run_name import generate_run_name
from utils.log_model import fetch_logged_data, write_to_file
from utils.evaluate import prepare_submission, get_input_data
import time
import mlflow
import warnings

def random_forest_xgboost_stacking(num, cat, X_train, y_train, model_name="stacked-model"):
    """
    Train a stacked model using Random Forest and XGBoost and log the results using MLflow.

    Parameters:
    - num (list): List of numerical feature names.
    - cat (list): List of categorical feature names.
    - X_train (DataFrame): Training data features.
    - y_train (Series): Training data target values.

    Returns:
    None. The function logs the training results using MLflow and writes the logged data to a file.
    """
    start_time = time.time()  
    
    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num)])

    # Define individual models
    rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())])

    xgb_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor())])

    # Create a Stacking ensemble model with Random Forest and XGBoost
    stacked_model = StackingRegressor(
        estimators=[('random_forest', rf_model), ('xgboost', xgb_model)],
        final_estimator=LinearRegression())

    run_name = generate_run_name()
    with mlflow.start_run(run_name=run_name) as run:
        stacked_model.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(stacked_model, "Stacked Model")
        
        # Perform 5-fold cross-validation and calculate the metrics for each fold
        scores = cross_val_score(stacked_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        
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
    pred = stacked_model.predict(X_test)
    prepare_submission(X_test, pred, run_name)
