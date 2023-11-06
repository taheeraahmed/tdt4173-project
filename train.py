from models.bayesian_search import bayes_search_catboost_stacked
from models.autogluon import autogluon
import logging
from utils.set_up import set_up

def main():
    set_up()
    logger = logging.getLogger()
    logger.info('The training is beginning')
    
    autogluon()


if __name__ == "__main__":
    main()
    
