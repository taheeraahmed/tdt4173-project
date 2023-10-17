from data_preprocess import data_preprocess, get_training_data, get_input_data
import pandas as pd
import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient
import random
import nltk
from nltk.corpus import words as nltk_words
import xgboost as xgb  # Import XGBoost
import warnings
from datetime import date

today = date.today()
warnings.filterwarnings("ignore", category=UserWarning, module="_distutils_hack")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow.data.digest_utils")
nltk.download("words")

# enable autologging
mlflow.sklearn.autolog()
filename = "logging.txt"

# TODO Se på model evaluation i mlflow
# TODO Suppress warnings

def log(string): 
    print(today.strftime("%m/%d/%y") + ' ' + time.time() + ' LOG '+ string)

def generate_run_name():
    english_words = nltk_words.words()
    random_words = random.sample(english_words, 2)
    run_name = " ".join(random_words)
    run_name = run_name.title()  # Capitalize the first letter of each word
    log(run_name)
    return run_name

def fetch_logged_data(run_id):
    client = MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def write_to_file(logged_data, start_time, filename = "logging.txt"):
    end_time = time.time()  # <- End the timer
    elapsed_time = end_time - start_time  # <- Calculate elapsed time
    with open(filename, 'a') as file:
        file.write('-------------------------------\n')

        file.write('Name: ')
        file.write(str(logged_data['name']) + '\n')

        file.write('Run name: ')
        file.write(str(logged_data['run_name']) + '\n')

        file.write('Elapsed time: ')
        file.write(str(elapsed_time) + ' seconds \n')

        file.write('Metrics \n')
        file.write(str(logged_data['metrics']) + '\n')

        best_model = logged_data.get('best_model')
        if best_model:
            file.write('Best model \n')
            file.write(str(best_model) + '\n')

        best_params = logged_data.get('best_params')
        if best_params:
            file.write('Best params \n')
            file.write(str(best_params) + '\n')
        

def lin_reg(num, cat, X_train, y_train):
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
        # Perform 5-fold cross-validation and calculate the metrics for each fold
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
        'run_name': run_name,
        'params': params,
        'metrics': metrics,
        'tags': tags, 
        'artifacts': artifacts,
    }

    write_to_file(logged_data, start_time)

def xgbost_model(num, cat, X_train, y_train):
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
        ('regressor', xgb.XGBRegressor())])  # Use XGBRegressor

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
        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)

    logged_data = {
        'name': 'XGBoost',
        'run_name': run_name,
        'params': params,
        'metrics': metrics,
        'tags': tags, 
        'artifacts': artifacts,
    }

    write_to_file(logged_data, start_time)

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
        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)


    logged_data = {
        'name': 'Random forest',
        'run_name': run_name,
        'params': params,
        'metrics': metrics,
        'tags': tags, 
        'artifacts': artifacts,
    }

    write_to_file(logged_data, start_time)

def other_imputer(num, cat, X_train, y_train):
    start_time = time.time()  # <- Start the timer
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer()),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
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
        # Perform 5-fold cross-validation and calculate the metrics for each fold
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        
        # Convert negative MSE to positive (optional, depends on your preference)
        mse_values = -scores
        
        # Log the metrics
        for i, mse in enumerate(mse_values):
            mlflow.log_metric(f'MSE_fold_{i}', mse)
        
        # Fetch and print logged data
        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)


    logged_data = {
        'name': 'Other imputer',
        'run_name': run_name,
        'params': params,
        'metrics': metrics,
        'tags': tags, 
        'artifacts': artifacts,
    }

    write_to_file(logged_data, start_time)

    return model
    
def grid_search(num, cat, X_train, y_train):

    start_time = time.time()  # <- Start the timer
    param_grid = [
        {"n_estimators": [70, 80, 90], "max_features": [3]},
        {"bootstrap": [False], "n_estimators": [70, 80, 90], "max_features": [3]}
        ]

    model = other_imputer(num, cat, X_train, y_train)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

    run_name = generate_run_name()
    with mlflow.start_run(run_name=run_name) as run:
        # Perform 5-fold cross-validation and calculate the metrics for each fold
        scores = cross_val_score(grid_search, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        
        # Convert negative MSE to positive (optional, depends on your preference)
        mse_values = -scores
        
        # Log the metrics
        for i, mse in enumerate(mse_values):
            mlflow.log_metric(f'MSE_fold_{i}', mse)
        
        # Fetch and print logged data
        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)


    logged_data = {
        'name': 'Grid search',
        'run_name': run_name,
        'params': params,
        'metrics': metrics,
        'tags': tags, 
        'artifacts': artifacts,
    }

    write_to_file(logged_data, start_time)

    best_params = grid_search.best_params_

    # Fit the model with the best parameters to your data
    best_model = grid_search.best_estimator_
    
    run_name = generate_run_name()
    
    with mlflow.start_run(run_name=run_name) as run:
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
        'name': 'Grid search (best model)',
        'run_name': run_name,
        'params': params,
        'metrics': metrics,
        'tags': tags, 
        'artifacts': artifacts,
        'best_params': best_params,
        'best_model': best_model
    }

    write_to_file(logged_data, start_time)

def main():
    log('Preprocessing data')

    data = data_preprocess(one_hot_location=False)
    X, y = get_training_data(data)
    X = X.drop(columns=['time', 'date_calc'])
    log('Done with preprocessing data')
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0, random_state=42)

    numeric_features = X_train.select_dtypes(include=['float32']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    log('The training is beginning')

    log('XGBoost')
    xgbost_model(numeric_features, categorical_features, X_train, y_train)

    log('Linear regression')
    lin_reg(numeric_features, categorical_features, X_train, y_train)

    log('Random forest')
    random_forest(numeric_features, categorical_features, X_train, y_train)

    log('Other imputer')
    other_imputer(numeric_features, categorical_features, X_train, y_train)

    log('Grid search')
    grid_search(numeric_features, categorical_features, X_train, y_train)





if __name__ == "__main__":
    main()
    
