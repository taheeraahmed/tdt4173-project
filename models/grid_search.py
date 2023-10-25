from sklearn.linear_model import LinearRegression
import mlflow
import utils
from sklearn.model_selection import cross_val_score, GridSearchCV

def grid_search(num, cat, X_train, y_train):
    start_time = time.time()  # <- Start the timer
    param_grid = [
        {"n_estimators": [70, 80, 90], "max_features": [3]},
        {"bootstrap": [False], "n_estimators": [70, 80, 90], "max_features": [3]}
        ]

    model = LinearRegression()
    model.fit(X_train, y_train)

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

    run_name = utils.generate_run_name()
    with mlflow.start_run(run_name=run_name) as run:
        grid_search.fit(X_train, y_train)
        for i, mse in enumerate(mse_values):
            mlflow.log_metric(f'MSE_fold_{i}', mse)
        params, metrics, tags, artifacts, time = utils.fetch_logged_data(run.info.run_id)


    logged_data = {
        'name': 'Grid search',
        'run_name': run_name,
        'time': time,
        'params': params,
        'metrics': metrics,
        'tags': tags, 
        'artifacts': artifacts,
    }
    utils.write_to_file(logged_data, start_time)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    run_name = utils.generate_run_name()
    with mlflow.start_run(run_name=run_name) as run:
        scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        mse_values = -scores
        for i, mse in enumerate(mse_values):
            mlflow.log_metric(f'MSE_fold_{i}', mse)
        params, metrics, tags, artifacts, time = utils.fetch_logged_data(run.info.run_id)

    logged_data = {
        'name': 'Grid search (best model)',
        'time': time,
        'run_name': run_name,
        'params': params,
        'metrics': metrics,
        'tags': tags, 
        'artifacts': artifacts,
        'best_params': best_params,
        'best_model': best_model
    }
    utils.write_to_file(logged_data, start_time)