FROM python:3.9-slim-buster
WORKDIR /usr/local/airflow

# Copy project files into the docker image
COPY requirements.txt ./
COPY dags ./dags
COPY NewClients ./NewClients
COPY ProcessedClients ./ProcessedClients
COPY WaitingList ./WaitingList

# Copy supervisord.conf
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf   

# Upgrade pip
RUN pip install --upgrade pip

# Install build dependencies for pyfarmhash
RUN apt-get update && apt-get upgrade -y && apt-get install -y build-essential gcc

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create and set the Airflow home directory
ENV AIRFLOW_HOME=/usr/local/airflow
# ENV AIRFLOW_HOME=${PWD}/airflow
RUN mkdir -p $AIRFLOW_HOME

# Initialize Airflow database
RUN airflow db init

# Create a new user `airflow`
RUN adduser --disabled-password --gecos '' airflow

# Change ownership of the airflow home directory to `airflow` user
RUN chown -R airflow:airflow $AIRFLOW_HOME

# Install supervisor
RUN apt-get install -y supervisor

# Switch to `airflow` user
USER airflow

# Start supervisor, which will start the web server and scheduler
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
