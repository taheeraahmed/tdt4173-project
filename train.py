from utils.set_up import set_up
from models.autogluon import autogluon
import logging

def main():
    set_up()
    logger = logging.getLogger()
    logger.info('The training is beginning')

    autogluon()

if __name__ == "__main__":
    main()
    
