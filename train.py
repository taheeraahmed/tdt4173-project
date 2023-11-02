from models.lin_reg import lin_reg
from models.stacked_models import random_forest_xgboost_stacking
from models.grid_search_random_forest import grid_search
from models.random_forest import random_forest
from models.decision_tree import decision_tree
from models.andrea_kok import andrea_kok
from data_preprocess import data_preprocess, get_training_data
import logging
from utils.set_up import set_up

def main():
    set_up()
    logger = logging.getLogger()

    logger.info('Preprocessing data')
    data = data_preprocess(one_hot_location=True)
    X, y = get_training_data(data)
    logger.info('Done with preprocessing data')

    numeric_features = X.select_dtypes(include=['float32']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    logger.info('The training is beginning')

    logger.info('Linear regression')
    lin_reg(numeric_features, categorical_features, X, y)

    logger.info('Andrea kok')
    andrea_kok()

    logger.info('Decision tree')
    decision_tree(numeric_features, categorical_features,X, y)
    
    logger.info('Random forest regression')
    random_forest(numeric_features, categorical_features, X, y)

    logger.info('Grid search with random forest')
    grid_search(numeric_features, categorical_features, X, y)

    logger.info('Random forest xgboost stacking')
    random_forest_xgboost_stacking(numeric_features,categorical_features, X,y) 

if __name__ == "__main__":
    main()
    
