from models.automl import automl
import logging
from utils.set_up import set_up

def main():
    set_up()
    logger = logging.getLogger()

    logger.info('The training is beginning')
    automl()

if __name__ == "__main__":
    main()
    
