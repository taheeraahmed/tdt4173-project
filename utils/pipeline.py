import mlflow

def run_pipeline_and_log(pipeline, X_train, y_train, X_test, location):
    with mlflow.start_run(run_name=f"pipeline_{location}"):
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test.drop(columns=["id", "prediction", "location"]))
    return predictions
