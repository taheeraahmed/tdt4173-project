from mlflow.tracking import MlflowClient
from datetime import date
import time

def fetch_logged_data(run_id):
    # Create an instance of the MlflowClient
    client = MlflowClient()
    
    TODAY = date.today()
    print(TODAY.strftime("%m/%d/%y") + ' ' + str(time.time()))
    
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    
    return data.params, data.metrics, tags, artifacts, time
