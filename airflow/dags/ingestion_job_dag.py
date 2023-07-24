from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models import SkipMixin
from airflow.exceptions import AirflowSkipException
from datetime import datetime, timedelta
import os
import shutil
import pandas as pd
import random

FOLDER_A = '/usr/local/airflow/Folder_A'
FOLDER_B = '/usr/local/airflow/Folder_B'
FOLDER_C = '/usr/local/airflow/Folder_C'
LOGS = '/usr/local/airflow/Logs/log.txt'

def read_from_folder_a(**context):
    # Check if there are new files in the directory
    source_data_files = os.listdir(FOLDER_A)
    destiny = FOLDER_C
    file = source_data_files[0]
    test = random.randint(0,2)
    
    # Testing 0 Files with too many errors
    if test == 0:
        destiny = FOLDER_B
        pass
    # Testin 1 Files with just a few errors
    elif test == 1:
        # destiny = FOLDER_B
        pass

    shutil.move(os.path.join(FOLDER_A, file), os.path.join(destiny, file))    
    save_in_log(file)

def save_in_log(file):
    # Write the moved file name to log.txt
    with open(LOGS, 'a') as f:
        f.write(f"{file}\n")

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
    'ingestion_model_dag',
    default_args=default_args,
    description='DAG for data ingestion',
    schedule_interval='*/1 * * * *',
)

t1 = PythonOperator(
    task_id='check_for_new_data',
    python_callable=read_from_folder_a,
    provide_context=True,
    dag=dag,
)

# Define task order
t1
