from models.bayesian_search import param_search_bayes_xgboost
import logging
from utils.set_up import set_up

def main():
    set_up()
    logger = logging.getLogger()
    logger.info('The training is beginning')

    param_search_bayes_xgboost()
    

if __name__ == "__main__":
    main()
    
