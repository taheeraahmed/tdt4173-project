from models.lin_reg import lin_reg
from models.stacked_models import random_forest_xgboost_stacking
from models.grid_search_random_forest import grid_search
from models.random_forest import random_forest
from models.decision_tree import decision_tree
from models.andrea_kok import andrea_kok
from utils.data_preprocess import data_preprocess, get_training_data
import logging
from utils.set_up import set_up

def feature_engineering(data):
    pass

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

    lin_reg(numeric_features, categorical_features, X, y, 'linear-regression-baseline')
    andrea_kok(model_name='andrea-kok')
    decision_tree(numeric_features, categorical_features, X, y, model_name='decision-tree')
    random_forest(numeric_features, categorical_features, X, y, model_name='random-forest')
    grid_search(numeric_features, categorical_features, X, y, model_name='grid-search')
    random_forest_xgboost_stacking(numeric_features,categorical_features, X,y, model_name='random-forest-xgboost-stacking') 

if __name__ == "__main__":
    main()
    
