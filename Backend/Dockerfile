# Base Image
FROM python:3.10-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    unixodbc \
    unixodbc-dev \
    gnupg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Add the Microsoft repository as a trusted source
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list

# Install the Microsoft ODBC Driver 17 for SQL Server
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
COPY shap_service.py shap_service.py

RUN pip install -r requirements.txt

COPY ./ ./

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

