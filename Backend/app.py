import json
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from dotenv import load_dotenv
import os
import numpy as np
import pyodbc
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
import uvicorn
import datetime as dt
from typing import List, Dict, Union
import random
from fastapi.encoders import jsonable_encoder


load_dotenv()

#region Database

def database_connection():
    server = os.getenv('SERVER')
    database = os.getenv('DATABASE')
    username = os.getenv('ADMINLOGIN')
    password = os.getenv('PASSWORD')
    driver= '{ODBC Driver 17 for SQL Server}'
    conn_str = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};Encrypt=yes;TrustServerCertificate=no;"

    connection = pyodbc.connect(conn_str)
    return connection

def database_read(command):
    #Connect to the database
    conn = database_connection()

    #Execute query
    result = pd.read_sql(command, conn)

    conn.close()

    return result

def database_write(command):
    #Connect to the database
    conn = database_connection()
    cursor = conn.cursor()
    cursor.execute(command)
    cursor.commit()
    conn.close()
    

def format_sql_command(table: str, data: dict) -> str:
    columns = ', '.join(data.keys())
    values = ', '.join(f"'{value}'" if isinstance(value, str) else str(value) for value in data.values())
    return f'INSERT INTO {table} ({columns}) VALUES ({values});'

#endregion

# define the app and the base URL
app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)