import time
from datetime import date

def log(string): 
    TODAY = date.today()
    print(TODAY.strftime("%m/%d/%y") + ' ' + str(time.time()) + ' LOG ' + string)


def fetch_logged_data(run_id, client):
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts