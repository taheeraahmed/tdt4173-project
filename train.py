from utils.data_preprocess import data_preprocess, get_input_data, remove_ouliers, get_training_data
from models.cat_boost import catboost_reg
import logging
from utils.set_up import set_up

def main():
    set_up()
    logger = logging.getLogger()

    logger.info('Preprocessing data')
    data = data_preprocess(one_hot_location=True)
    data = remove_ouliers(data)
    X, y = get_training_data(data)

    logger.info('Done with preprocessing data')

    numeric_features = X.select_dtypes(include=['float32']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    logger.info('The training is beginning')

    catboost_reg(numeric_features, categorical_features, X, y, "catboost-removed-outliers")



if __name__ == "__main__":
    main()
    
