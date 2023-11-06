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

    # Train the model
    logger.info('Training 4 location A')
    model_a = TabularPredictor(label=label, problem_type='regression').fit(data_a)
    logger.info('Training 4 location B')
    model_b = TabularPredictor(label=label, problem_type='regression').fit(data_b)
    logger.info('Training 4 location C')
    model_c = TabularPredictor(label=label, problem_type='regression').fit(data_c)

    # Save the model
    logger.info('Training 4 location A')
    model_a.save('autogluon/a')
    logger.info('Training 4 location B')
    model_b.save('autogluon/b')
    logger.info('Training 4 location C')
    model_c.save('autogluon/c')

    # Make predictions
    pred_a = model_a.predict(X_test_a.drop(columns=["id", "prediction", "location"]))
    pred_b = model_b.predict(X_test_b.drop(columns=["id", "prediction", "location"]))
    pred_c = model_c.predict(X_test_c.drop(columns=["id", "prediction", "location"]))

    prepare_submission(X_test_a, X_test_b, X_test_c, pred_a, pred_b, pred_c, run_name)