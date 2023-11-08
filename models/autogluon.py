from autogluon.tabular import TabularPredictor
import logging 
from utils.generate_run_name import generate_run_name
from utils.read_data import load_data, get_test_data, prepare_submission
from utils.log_model import check_create_directory

def autogluon(model_name = 'autogluon-with-more-feature-eng'):
    logger = logging.getLogger()
    logger.info(model_name)
    run_name = generate_run_name()

    logger.info('Processing data')
    
    drop_cols = ['time', 'elevation:m', 'fresh_snow_1h:cm', 'ceiling_height_agl:m', 'snow_density:kgm3',
            'wind_speed_w_1000hPa:ms', 'snow_drift:idx', 'fresh_snow_3h:cm', 'is_in_shadow:idx', 'dew_or_rime:idx', 'fresh_snow_6h:cm', 'prob_rime:p', 'fresh_snow_1h:cm'] # this second line is columns with feature importance == 0

    data_a, data_b, data_c = load_data(mean=True, remove_out=True, roll_avg=True, cust_feat=True, drop_cols=drop_cols)
    X_test_a, X_test_b, X_test_c = get_test_data(mean=True, roll_avg=True, cust_feat=True)

    logger.info('Done processing data')
    # Specify the column name that contains the target variable to predict
    label = 'pv_measurement'


    # Assuming 'logger' is already defined and 'label' and 'run_name' are defined elsewhere
    data_frames = [('A', data_a, X_test_a), ('B', data_b, X_test_b), ('C', data_c, X_test_c)]
    models = {}
    predictions = {}

    for location, data, X_test in data_frames:
        # train the model
        logger.info(f'Training for location {location}')
        # the base directory where you want to check/create
        base_path = 'autogluon'  
        model_path = check_create_directory(base_path, run_name+'-'+location)
        model = TabularPredictor(label=label, problem_type='regression', path=model_path).fit(data)
        models[location] = model
        logger.info(f'Model saved for location {location} at {model_path}')
        
        # make predictions
        predictions[location] = model.predict(X_test.drop(columns=["id", "prediction", "location"]))

    # Prepare submission
    prepare_submission(X_test_a, X_test_b, X_test_c, predictions['A'], predictions['B'], predictions['C'], run_name)
