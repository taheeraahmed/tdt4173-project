import mlflow

def run_pipeline_and_log(pipeline, X_train, y_train, X_test, location):
    with mlflow.start_run(run_name=f"pipeline_{location}"):
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test.drop(columns=["id", "prediction", "location"]))
        if 'random_forest' in pipeline.named_steps:
            rf = pipeline.named_steps['random_forest']
            mlflow.log_param("n_estimators", rf.n_estimators)
            mlflow.log_param("max_features", rf.max_features)
        mlflow.sklearn.log_model(pipeline, f"pipeline_{location}")
    return predictions
