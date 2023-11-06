import os
import sys
import time
import json
from datetime import datetime, timedelta

from clickhouse_driver import Client
import mindsdb_sdk

# Training State
STATE_FILE = 'training_state.json'
RETRAIN_HOURS = 1                           # 1 hour interval for retraining

# Clickhouse DB
DB_HOST = 'clickhouse'
DB_PORT = 9000
CLICKHOUSE_TABLE = 'candle_data'

# Candle Interval
TIMEFRAME: str = sys.argv[1]
GAP_THRESHOLD_SECONDS = 60 * 5              # Define gap threshold, e.g., 5 minutes for 1m candles
EXPECTED_INTERVAL_SECONDS = 60              # Replace with the expected number of seconds between rows

# MindsDB
MINDBS_API_URL = 'http://mindsdb:47334'     # Replace with your MindsDB API URL
MIN_ROWS = 144                              # Rows to consider for training
PROJECT_ID = 'mindsdb'
DATASOURCE_NAME = 'clickhouse_datasource'

# Delete Models
DELETE_MODELS = False

# Establish ClickHouse connection
client = Client(host=DB_HOST, port=DB_PORT)

# Connect to the MindsDB server
con = mindsdb_sdk.connect(MINDBS_API_URL)
con.get_database(DATASOURCE_NAME)

project = con.get_project(PROJECT_ID)

def save_training_state(model_name, current_time, file_path=STATE_FILE):
    state = load_training_state(file_path)
    state['last_trained'][model_name] = current_time.strftime('%Y-%m-%d %H:%M:%S')
    with open(file_path, 'w') as f:
        json.dump(state, f, indent=4)

def load_training_state(file_path=STATE_FILE):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        # If the file does not exist or is corrupt, return a default state
        return {'last_trained': {}, 'timeframe': TIMEFRAME}

def is_retrain_due(model_name, current_time, file_path=STATE_FILE):
    state = load_training_state(file_path)
    last_trained_str = state['last_trained'].get(model_name)
    retrain_interval = timedelta(hours=RETRAIN_HOURS)  # Set to 1 hour or whatever is appropriate

    if last_trained_str:
        last_trained = datetime.strptime(last_trained_str, '%Y-%m-%d %H:%M:%S')
        time_until_next_retraining = (last_trained + retrain_interval) - current_time
        retrain_due = time_until_next_retraining.total_seconds() <= 0
    else:
        # If the model has never been trained, we need to train it now
        time_until_next_retraining = timedelta(0)
        retrain_due = True

    return retrain_due, time_until_next_retraining

def get_valid_symbols(client, min_rows=144, expected_interval_seconds=EXPECTED_INTERVAL_SECONDS, timeframe=TIMEFRAME):
    valid_symbols = []

    # Query to check for gaps in timestamps and NaNs in the data
    query = f"""
    SELECT 
        t1.symbol, 
        count() AS row_count
    FROM {CLICKHOUSE_TABLE}.candles_{timeframe} t1
    LEFT JOIN {CLICKHOUSE_TABLE}.candles_{timeframe} t2 ON t1.symbol = t2.symbol AND t2.start = t1.start + INTERVAL {expected_interval_seconds} SECOND
    WHERE 
        t2.symbol IS NOT NULL 
        AND t1.open IS NOT NULL 
        AND t1.close IS NOT NULL 
        AND t1.high IS NOT NULL 
        AND t1.low IS NOT NULL 
        AND t1.volume IS NOT NULL
    GROUP BY t1.symbol
    HAVING row_count >= {min_rows}
    """
    
    results = client.execute(query)
    for symbol, row_count in results:
        valid_symbols.append(symbol)
    
    return valid_symbols

def drop_all_models(project):
    # Iterate over the list of models and delete each one
    existing_models = project.list_models()
    for model in existing_models:
        model_name = model.name  # Assuming 'name' is the correct attribute for the model's name
        query = project.drop_model(model_name)
        print(f"Model {model_name} dropped successfully.")
        
def generate_view_query(symbol, timeframe):
    # Assuming this function returns the correct SQL query for your data
    min_rows = MIN_ROWS
    return f"SELECT * FROM {DATASOURCE_NAME}.candles_{timeframe} WHERE symbol = '{symbol}'"

def get_or_create_view(project, symbol, timeframe, query_func):
    """
    Check if a view for a given symbol and timeframe exists, and create it if not.

    Parameters:
    - project: The MindsDB project instance
    - symbol: The trading symbol
    - timeframe: The timeframe for the candles
    - query_func: Function to generate the SQL query for the view

    Returns:
    The view object
    """
    view_name = f'{symbol}_candles_{timeframe}'

    try:
        # Try to get an existing view
        my_view = project.get_view(view_name)
        print(f"View {view_name} already exists.")
    except AttributeError:
        # If the view does not exist, catch the specific error
        print(f"View {view_name} does not exist, creating a new one.")
        # Call query_func to create a new view
        sql_query = query_func(symbol, timeframe)
        my_view = project.create_view(view_name, sql_query)

    return my_view

def check_retraining_status(symbol, model_name_suffix):
    timeframe = TIMEFRAME
    model_name = f'{symbol}_{timeframe}_{model_name_suffix}'
    current_time = datetime.now()
    retrain_due, time_until_next_retraining = is_retrain_due(model_name, current_time)
    return retrain_due, time_until_next_retraining, model_name, current_time

def manage_view(project, symbol):
    timeframe = TIMEFRAME
    return get_or_create_view(project, symbol, timeframe, generate_view_query)

def list_existing_models(project):
    try:
        return project.list_models()
    except Exception as e:
        print(f"Failed to list models: {e}")
        return []

def generate_sql_query(project, symbol, limit):
    timeframe = TIMEFRAME
    return project.query(f"""
        SELECT *
        FROM {DATASOURCE_NAME}.{symbol}_candles_{timeframe}
        WHERE symbol = '{symbol.upper()}' AND interval = '1m'
        ORDER BY start 
        LIMIT {limit}
        """)

def retrain_existing_model(model_to_retrain, sql_query, datasource_name):
    print(f"Retraining initiated for model `{model_to_retrain.name}`.")
    retrain_options = {"stop_after": 60}
    model_to_retrain.retrain(query=sql_query, database=datasource_name, options=retrain_options)

def create_new_model(project, symbol, datasource_name, sql_query, model_name):
    print(f"Training new model `{model_name}` for symbol: {symbol}")
    # Model training parameters
    create_params = {
        "name": model_name,
        "datasource_name": datasource_name,
        "query": sql_query,  # Ensure this is just a SELECT statement
        "predict": "open",
        "engine": "lightwood",
        "stop_after": 60,
        "using": {
            "model": {
                "args": {
                    "submodels": [
                        {
                            "module": "NHitsMixer",
                            "args": {
                                # Make sure to fill these in as required
                                "window": 64,
                                "horizon": 12,
                                "order_by": "start",
                                "group_by": "symbol"
                            }
                        }
                    ]
                }
            }
        },
        "timeseries_options": {
            # Make sure to fill these in as required
            "window": 64,
            "horizon": 12,
            "order": "start",
            "group": "symbol"
        }
    }

    try:
        project.models.create(**create_params)
        time.sleep(5)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def print_next_training_time(time_until_next_retraining, model_name):
    remaining_str = str(time_until_next_retraining).split('.')[0]
    print(f"Next training for {model_name} in {remaining_str}.")

def train_mindsdb_model(project, symbol, datasource_name, model_name_suffix='model', limit=MIN_ROWS):
    retrain_due, time_until_next_retraining, model_name, current_time = check_retraining_status(symbol, model_name_suffix)
    my_view = manage_view(project, symbol)
    
    if retrain_due:
        existing_models = list_existing_models(project)
        if not existing_models:
            return
        
        sql_query = generate_sql_query(project, symbol, limit)
        model_to_retrain = next((model for model in existing_models if model.name.lower() == model_name.lower()), None)
        
        if model_to_retrain is not None:
            retrain_existing_model(model_to_retrain, sql_query, datasource_name)
        else:
            create_new_model(project, symbol, datasource_name, sql_query, model_name)
        
        save_training_state(model_name, current_time)
    else:
        print_next_training_time(time_until_next_retraining, model_name)


if __name__ == '__main__':
    valid_symbols = get_valid_symbols(client)

    if DELETE_MODELS:
        drop_all_models(project)

    else:
        unique_symbols = set(valid_symbols)
        for symbol in unique_symbols:
            symbol_upper = symbol.upper()
            print(f"Training model for symbol: {symbol_upper}")
            train_mindsdb_model(project, symbol_upper, DATASOURCE_NAME, model_name_suffix='model')
            time.sleep(30)
            break
            
