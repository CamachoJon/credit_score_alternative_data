from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models import SkipMixin
from airflow.exceptions import AirflowSkipException
from datetime import datetime, timedelta
import os
import shutil
import pandas as pd

WAITING_LIST = '/usr/local/airflow/WaitingList'
NEW_CLIENTS = '/usr/local/airflow/NewClients'
PROCESSED_CLIENTS = '/usr/local/airflow/ProcessedClients'
LOGS = '/usr/local/airflow/Logger/log.txt'

def read_from_folder_a(**context):
    """
    Moves the first file from the WAITING_LIST directory to the NEW_CLIENTS directory and saves its name in the log file.

    Args:
        **context: Airflow context with information about the task execution.

    Returns:
        None
    """
    # Check if there are new files in the directory
    source_data_files = os.listdir(WAITING_LIST)
    file = source_data_files[0]

    shutil.move(os.path.join(WAITING_LIST, file), os.path.join(NEW_CLIENTS, file))    
    save_in_log(file)

def save_in_log(file):
    """
    Writes the moved file name to the log.txt file.

    Args:
        file (str): The name of the file to be saved in the log.

    Returns:
        None
    """
    print(file)
    # Write the moved file name to log.txt
    with open(LOGS, 'a') as f:
        f.write(f"{file}\n")

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
    'ingestion_model_dag',
    default_args=default_args,
    description='DAG for data ingestion',
    schedule_interval=timedelta(seconds=10),
    catchup=False
)

t1 = PythonOperator(
    task_id='check_for_new_data',
    python_callable=read_from_folder_a,
    provide_context=True,
    dag=dag
)

# Define task order
t1
