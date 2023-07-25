from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models import SkipMixin
from airflow.exceptions import AirflowSkipException
from datetime import datetime, timedelta
import os
import requests
import json
import shutil
import pandas as pd
import numpy as np

# define directory to handle data
WAITING_LIST = '/usr/local/airflow/WaitingList'
NEW_CLIENTS = '/usr/local/airflow/NewClients'
PROCESSED_cLIENTS = '/usr/local/airflow/ProcessedClients'
LOGS = '/usr/local/airflow/Logs/log.txt'

# define the url of your model service
model_service_url = 'http://Backend:8000/predict'
features = 'http://Backend:8000/get_features'

def check_for_new_data(**context):

    new_files = read_log_file(LOGS)

    if new_files:
        # Pass the new files to the next tasks
        context['task_instance'].xcom_push('new_files', new_files)
    else:
        # If no new files, skip the DAG
        raise AirflowSkipException
    
def make_predictions(**context):
    # Retrieve new files from the previous task
    new_files = context['task_instance'].xcom_pull(task_ids='check_for_new_data', key='new_files')

    # Function to prepare data
    data = prepare_data_for_api(new_files)

    # Call to endpoint
    response = requests.post(model_service_url, json=data)

def prepare_data_for_api(files):
    NUM_COLS, CAT_COLS, FEATURESS = get_features()

    dataframes = []

    for file in files:
        if 'GOOD' in file or 'CHANGED' in file:
            iterator = pd.read_csv(f'{FOLDER_C}/{file}').sample(1000)
            dataframes.append(iterator)

    if len(dataframes) == 0:
        raise AirflowSkipException    

    if len(dataframes) > 1:
        merged_df = dataframes[0]  # Start with the first DataFrame in the list

        # Merge the remaining DataFrames one by one
        for i in range(1, len(dataframes)):
            merged_df = pd.merge(merged_df, dataframes[i])
    else:
        merged_df = dataframes[0]  # Only one DataFrame, no need to merge

        
    input_data = merged_df

    input_data[CAT_COLS] = input_data[CAT_COLS].fillna("na")
    input_data[NUM_COLS] = input_data[NUM_COLS].fillna(0)
    input_data[FEATURESS] = input_data[FEATURESS].fillna(0)
    input_data = input_data.replace([np.inf, -np.inf], np.nan)

    input_data_dict = input_data.to_dict(orient='records')
    for item in input_data_dict:
        for key, value in item.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                item[key] = str(value)
    
    return input_data_dict

def get_unique_vals():
    uv_response = requests.get(unique_vals)
    unique_vals_str = uv_response.text
    try:
        unique_vals = json.loads(unique_vals_str)
        unique_vals = json.loads(unique_vals)  # Convert to JSON object
    except (json.JSONDecodeError, TypeError) as e:
        pass    
    
    return unique_vals

def get_features():
    response = requests.get(features)
    features_set = response.json()
    NUM_COLS = features_set[0]
    CAT_COLS = features_set[1]
    FEATURES = features_set[2]

    return NUM_COLS, CAT_COLS, FEATURES

def read_log_file(file_path):
    # Read the log file and convert lines into a list
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    
    # Clean the log file
    with open(file_path, 'w') as f:
        f.write('')
    
    return lines

default_args = {
    'owner': 'Jonathan Camacho',
    'depends_on_past': False,
    'start_date': datetime.now(),
    'email': ['jonathan.camacho@epita.fr'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'predictive_model_dag',
    default_args=default_args,
    description='A simple predictive model DAG',
    schedule_interval='*/2 * * * *',
)

t1 = PythonOperator(
    task_id='check_for_new_data',
    python_callable=check_for_new_data,
    provide_context=True,
    dag=dag,
    catchup=False
)

t2 = PythonOperator(
    task_id='make_predictions',
    python_callable=make_predictions,
    provide_context=True,
    dag=dag,
    catchup=False
)

# Define task order
t1 >> t2
