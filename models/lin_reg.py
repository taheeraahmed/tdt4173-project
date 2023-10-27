from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from utils.generate_run_name import generate_run_name
from utils.log_model import fetch_logged_data, write_to_file
import mlflow
import time

mlflow.autolog()

def lin_reg(num, cat, X_train, y_train):
    """
    Train a linear regression model using given training data and log the results using MLflow.

    Parameters:
    - num (list): List of numerical feature names.
    - cat (list): List of categorical feature names.
    - X_train (DataFrame): Training data features.
    - y_train (Series): Training data target values.

    Returns:
    None. The function logs the training results using MLflow and writes the logged data to a file.
    """
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
        ('regressor', LinearRegression())])

    run_name = generate_run_name()

    with mlflow.start_run(run_name=run_name) as run:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        # Convert negative MSE to positive (optional, depends on your preference)
        mse_values = -scores

        # Log the metrics
        for i, mse in enumerate(mse_values):
            mlflow.log_metric(f'MSE_fold_{i}', mse)

              
        # Log the model artifact
        mlflow.sklearn.log_model(model, "Linear regression")
        
        # Fetch and print logged data
        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
    
    logged_data = {
        'name': 'Linear regression',
        'run_name': run_name,
        'params': params,
        'metrics': metrics,
        'tags': tags, 
        'artifacts': artifacts,
    }

    write_to_file(logged_data, start_time)