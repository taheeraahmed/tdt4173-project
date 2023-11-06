from autogluon.tabular import TabularDataset, TabularPredictor
import logging 
from utils.generate_run_name import generate_run_name
from utils.data_preprocess_location import load_data, get_test_data, prepare_submission

def autogluon(model_name = 'autogluon'):
    logger = logging.getLogger()

    logger.info(model_name)
    run_name = generate_run_name()

    logger.info('Processing data')
    data_a, data_b, data_c = load_data()

    X_test_a, X_test_b, X_test_c = get_test_data()


    drop_cols = ['time', 'date_calc', 'elevation:m', 'fresh_snow_1h:cm', 'wind_speed_u_10m:ms', 
                'wind_speed_u_10m:ms', 'wind_speed_v_10m:ms', 'wind_speed_w_1000hPa:ms', 'prob_rime:p',
                'fresh_snow_12h:cm','fresh_snow_24h:cm', 'fresh_snow_6h:cm', 'super_cooled_liquid_water:kgm2']
    
    # Function to drop columns from a dataframe
    def drop_columns(df, columns):
        return df.drop(columns=columns, errors='ignore')

    # Drop the columns from all dataframes
    data_a = drop_columns(data_a, drop_cols)
    data_b = drop_columns(data_b, drop_cols)
    data_c = drop_columns(data_c, drop_cols)
    X_test_a = drop_columns(X_test_a, drop_cols)
    X_test_b = drop_columns(X_test_b, drop_cols)
    X_test_c = drop_columns(X_test_c, drop_cols)

    print(data_a.head())

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
