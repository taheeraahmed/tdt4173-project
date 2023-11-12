import pyfiglet
import logging
import warnings
import mlflow
import random

def set_up():
    result = pyfiglet.figlet_format("Gosling slayers", font = "slant") 
    print(result) 

    random.seed(10)
    LOG_FILE = "logs/log_file.txt"
    WARNINGS_TO_SUPPRESS = [
        ("ignore", UserWarning, "_distutils_hack"),
        ("ignore", FutureWarning, "mlflow.data.digest_utils"),
        ("ignore", FutureWarning, "sklearn.preprocessing._encoders"),
        ("ignore", UserWarning, "optional dependency `torch` is not available"),
    ]

    for action, category, module in WARNINGS_TO_SUPPRESS:
        warnings.filterwarnings(action, category=category, module=module)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    logging.basicConfig(level=logging.INFO, 
                    format='[%(levelname)s] %(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE),
                        logging.StreamHandler()
                    ])
    mlflow.autolog()
    logging.getLogger('mlflow').setLevel(logging.ERROR)
