from mlflow.tracking import MlflowClient
from datetime import date
import time

def fetch_logged_data(run_id):
    # Create an instance of the MlflowClient
    client = MlflowClient()
    
    TODAY = date.today()
    
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    
    return data.params, data.metrics, tags, artifacts, time

def write_to_file(logged_data, start_time, filename = "log_model.txt"):
    end_time = time.time()  # <- End the timer
    elapsed_time = end_time - start_time  # <- Calculate elapsed time
    with open(filename, 'a') as file:
        file.write('-------------------------------\n')

        file.write('Name: ')
        file.write(str(logged_data['name']) + '\n')

        file.write('Run name: ')
        file.write(str(logged_data['run_name']) + '\n')

        file.write('Time logged: ')
        file.write(str(logged_data['time']) + '\n')


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
