    FROM python:3.10-slim-buster

    WORKDIR /app

    COPY requirements.txt requirements.txt
    COPY __init__.py __init__.py

    RUN pip install -r requirements.txt

    COPY ./ ./

    CMD ["streamlit", "run", "streamlit.py"]
