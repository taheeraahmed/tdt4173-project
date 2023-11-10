from utils.set_up import set_up
from models.autogluon import autogluon
from models.beat_keno import fuck_keno
import logging

def main():
    set_up()
    logger = logging.getLogger()
    logger.info('The training is beginning')

    fuck_keno()

if __name__ == "__main__":
    main()
    
