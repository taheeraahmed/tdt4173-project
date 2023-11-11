import os

def check_create_directory(base_path, run_name):
    model_path = os.path.join(base_path, run_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path