import warnings
from models.lin_reg import lin_reg
from models.random_forest import random_forest
from sklearn.model_selection import train_test_split
from data_preprocess import data_preprocess, get_training_data
import logging
from utils.set_up import set_up

def main():
    set_up()
    logger = logging.getLogger()

    logger.info('Preprocessing data')
    data = data_preprocess(one_hot_location=False)
    X, y = get_training_data(data)
    X = X.drop(columns=['time', 'date_calc'])
    logger.info('Done with preprocessing data')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X_train.select_dtypes(include=['float32']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    logger.info('The training is beginning')

    logger.info('Linear regression')
    lin_reg(numeric_features, categorical_features, X_train, y_train)

    #logger.info('Random forest regression')
    #random_forest(numeric_features, categorical_features, X_train, y_train)


if __name__ == "__main__":
    main()
    
