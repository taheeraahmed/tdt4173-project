from models.stacked_catboost import stacked_catboost
import logging
from utils.set_up import set_up

def main():
    set_up()
    logger = logging.getLogger()
    logger.info('The training is beginning')
    
    stacked_catboost()


if __name__ == "__main__":
    main()
    
