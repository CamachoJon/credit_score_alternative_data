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
PROCESSED_CLIENTS = '/usr/local/airflow/ProcessedClients'
LOGS = '/usr/local/airflow/Logger/log.txt'

# define the url of your model service
model_service_url = 'http://Backend:8000/predict'

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

    for file in new_files:
        df = pd.read_csv(os.path.join(NEW_CLIENTS, file))

        # Replace NaN values with "NaN" or "null"
        df.fillna("NaN", inplace=True)

        df_dict = df.to_dict(orient='records')

        # Call to endpoint
        response = requests.post(model_service_url, json=df_dict)


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
    'start_date': datetime(2023, 7, 24),
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
    catchup=False
)

t1 = PythonOperator(
    task_id='check_for_new_data',
    python_callable=check_for_new_data,
    provide_context=True,
    dag=dag
)

t2 = PythonOperator(
    task_id='make_predictions',
    python_callable=make_predictions,
    provide_context=True,
    dag=dag
)

# Define task order
t1 >> t2
