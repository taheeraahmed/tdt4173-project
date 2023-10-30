from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import time
from utils.log_model import fetch_logged_data, write_to_file
from utils.generate_run_name import generate_run_name
import mlflow


def decision_tree(num, cat, X_train, y_train):
    """
    Train a decision tree model using given training data and log the results using MLflow.

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
        ('regressor', tree.DecisionTreeRegressor(max_depth=7))])
    
    run_name = generate_run_name()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.sklearn.log_model(model, "DecisionTree")
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        # Convert negative MSE to positive (optional, depends on your preference)
        mse_values = -scores
        # Log the metrics
        for i, mse in enumerate(mse_values):
            mlflow.log_metric(f'MSE_fold_{i}', mse)
        # Fetch and print logged data
        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
    
        
    logged_data = {
        'name': 'Linear regression',
        'start_time': start_time,
        'run_name': run_name,
        'params': params,
        'metrics': metrics,
        'tags': tags, 
        'artifacts': artifacts,
    }

    write_to_file(logged_data)

