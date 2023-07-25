import json
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import numpy as np
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
import uvicorn
import datetime as dt
from typing import List, Dict, Union
import random
from fastapi.encoders import jsonable_encoder
from Database import Database
from DataPreparation import DataPreparation


# define the app and the base URL
app = FastAPI()

@app.get('/user_data')
async def get_user_data():
    db = Database()
    query = "SELECT * FROM [dbo].[USERS]"
    results = db.read(query)
    results_json = results.to_json(orient='records')

    return results_json

# define the index
@app.get("/")
async def root():
    return {"message": "This is the root route of the API."}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)