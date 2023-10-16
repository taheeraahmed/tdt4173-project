from data_preprocess import data_preprocess, get_training_data, get_input_data, prepare_submission
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

filename = "mse_results.txt"

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def write_to_file(filename, mse, scores, model, start_time):
    end_time = time.time()  # <- End the timer
    elapsed_time = end_time - start_time  # <- Calculate elapsed time
    with open(filename, 'w+') as file:
        file.write('-------------------------------\n')
        file.write(model + '\n')
        file.write('mse: ' + str(mse) + '\n')
        file.write('mean mse: ' + str(scores) + '\n')
        file.write('elapsed time: ' + str(elapsed_time) + ' seconds\n\n')  # <- Write elapsed time

def lin_reg(num, cat, X_train, y_train, X_test, y_test):
    start_time = time.time()  # <- Start the timer
    # Define preprocessor
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num),
            ('cat', categorical_transformer, cat)])

    # Define model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LinearRegression())])

    # Fitting the model to your data
    model.fit(X_train, y_train)

    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_mse = -scores.mean()

    # Open the file with write ('w') or append ('a') mode and write the message
    write_to_file(filename, mse, mean_mse, 'Linear regressor', start_time)

def random_forest(num, cat, X_train, y_train, X_test, y_test):
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
        ('classifier', RandomForestRegressor())])

    # Fitting the model to your data
    model.fit(X_train, y_train)

    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_mse = -scores.mean()

    write_to_file(filename, mse, mean_mse, 'Random forest', start_time)

def other_imputer(num, cat, X_train, y_train, X_test, y_test ):
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
        ('classifier', LinearRegression())])
    
    model.fit(X_train, y_train)

    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_mse = -scores.mean()

    write_to_file(filename, mse, mean_mse, 'Other imputer', start_time)

    return model
    
def grid_search(num, cat, X_train, y_train, X_test, y_test ):

    start_time = time.time()  # <- Start the timer
    param_grid = [
        {"n_estimators": [70, 80, 90], "max_features": [3]},
        {"bootstrap": [False], "n_estimators": [70, 80, 90], "max_features": [3]}
        ]
    # Create a grid search object
    model = other_imputer(num, cat, X_train, y_train)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

    # Fit the grid search object to your data
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Print the best parameters
    with open(filename, 'w') as file:
        file.write("Grid search best parameters: ", best_params)

    # Fit the model with the best parameters to your data
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_mse = -scores.mean()

    write_to_file(filename, mse, mean_mse, 'Grid search', start_time)

def main():
    print('It has begun')
    # Specify the filename
    print('Preprocessing data')
    # Split the data into training and testing sets
    data = data_preprocess(one_hot_location=True)
    X, y = get_training_data(data)
    X = X.drop(columns=['time', 'date_calc'])
    print('Done with preprocessing data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    numeric_features = X_train.select_dtypes(include=['float32']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    print('Linear regression')
    lin_reg(numeric_features, categorical_features, X_train, y_train, X_test, y_test)
    print('Random forest')
    random_forest(numeric_features, categorical_features, X_train, y_train, X_test, y_test)
    print('Other imputer')
    _ = other_imputer(numeric_features, categorical_features, X_train, y_train, X_test, y_test)
    print('Grid search')
    grid_search(numeric_features, categorical_features, X_train, y_train, X_test, y_test)



if __name__ == "__main__":
    main()
    
