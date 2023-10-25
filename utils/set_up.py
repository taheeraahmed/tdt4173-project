import pyfiglet
import logging
import warnings

def set_up():
    result = pyfiglet.figlet_format("Gosling slayers", font = "slant"  ) 
    print(result) 
    # Constants
    LOG_FILE = "log_file.txt"
    WARNINGS_TO_SUPPRESS = [
    ("ignore", UserWarning, "_distutils_hack"),
    ("ignore", FutureWarning, "mlflow.data.digest_utils")
    ]

    for action, category, module in WARNINGS_TO_SUPPRESS:
        warnings.filterwarnings(action, category=category, module=module)


    logging.basicConfig(level=logging.INFO, 
                    format='[%(levelname)s] %(asctime)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE),
                        logging.StreamHandler()
                    ])