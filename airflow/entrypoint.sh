#!/usr/bin/env bash
set -e

# Initialize the database
airflow db init

# Change configurations in the airflow.cfg file
AIRFLOW_CFG_FILE=/usr/local/airflow/airflow.cfg

sed -i 's/^ *enable_xcom_pickling *=.*/enable_xcom_pickling = True/' $AIRFLOW_CFG_FILE
sed -i 's/^ *load_examples *=.*/load_examples = False/' $AIRFLOW_CFG_FILE
sed -i 's/^ *expose_config *=.*/expose_config = True/' $AIRFLOW_CFG_FILE
sed -i 's/^ *cookie_secure *=.*/cookie_secure = False/' $AIRFLOW_CFG_FILE

# Create a user
airflow users create \
    --username admin \
    --firstname admin \
    --lastname admin \
    --role Admin \
    --email admin@admin.com

# Start the web server, default port is 8080
exec airflow webserver -p 8080