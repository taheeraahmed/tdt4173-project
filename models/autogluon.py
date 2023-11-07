from autogluon.tabular import TabularDataset, TabularPredictor
import logging 
from utils.generate_run_name import generate_run_name
from utils.data_preprocess_location import load_data, get_test_data, prepare_submission
from utils.data_preprocess import ColumnDropper
from sklearn.pipeline import Pipeline
import numpy as np

def autogluon(model_name = 'autogluon-with-more-feature-eng'):
    logger = logging.getLogger()

    logger.info(model_name)
    run_name = generate_run_name()

    logger.info('Processing data')
    data_a, data_b, data_c = load_data(mean=True, roll_avg=True, remove_out=True)

    

    X_test_a, X_test_b, X_test_c = get_test_data(mean=True)

    # Specify the column name that contains the target variable to predict
    label = 'pv_measurement'

    logger.info('Done processing data')

    # Assuming 'logger' is already defined and 'label' and 'run_name' are defined elsewhere
    data_frames = [('A', data_a, X_test_a), ('B', data_b, X_test_b), ('C', data_c, X_test_c)]
    models = {}
    predictions = {}

    for location, data, X_test in data_frames:
        # Train the model
        logger.info(f'Training for location {location}')
        model = TabularPredictor(label=label, problem_type='regression').fit(data)
        models[location] = model

        # Save the model
        model_path = f'autogluon/{location.lower()}'
        model.save(model_path)
        logger.info(f'Model saved for location {location} at {model_path}')

        # Make predictions
        predictions[location] = model.predict(X_test.drop(columns=["id", "prediction", "location"]))

    # Prepare submission
    prepare_submission(X_test_a, X_test_b, X_test_c, predictions['A'], predictions['B'], predictions['C'], run_name)
