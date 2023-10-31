from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score

from utils.log_model import fetch_logged_data, write_to_file
from utils.evaluate import prepare_submission, get_input_data
from utils.generate_run_name import generate_run_name

import mlflow
import time


def decision_tree(num, cat, X_train, y_train, model_name="decision-tree"):
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

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num)])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', tree.DecisionTreeRegressor(max_depth=7))])
    
    run_name = generate_run_name()
    with mlflow.start_run(run_name=run_name) as run:
        model.fit(X_train,y_train)
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
    pred = model.predict(X_test)
    prepare_submission(X_test, pred, run_name)
