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
import sys
sys.path.append("..")
from credit_risk import inference


# define the app and the base URL
app = FastAPI()

@app.get('/user_data', response_model=List[Dict])
async def get_user_data():
    db = Database()
    query = "SELECT * FROM [dbo].[USERS]"
    results = db.read(query)
    
    results.replace([np.inf, -np.inf], np.nan, inplace=True)
    results.fillna(value="NaN", inplace=True)  # use a string to represent NaN

    results_dict = results.to_dict(orient='records')

    # Use FastAPI's jsonable_encoder to convert our data into JSON compatible format
    return jsonable_encoder(results_dict)


# define the index
@app.get("/")
async def root():
    return {"message": "This is the root route of the API."}

@app.post("/predict")
async def predict(features: List[Dict[str, Union[str, int, float]]]):
    # to do prediction using the dataframe
    df = pd.DataFrame(columns=features[0])
    df = df.append(features, ignore_index=True)

    current_date = dt.date.today().strftime("%d-%m-%Y")

    predictions = inference.make_predictions(df)
    df["PREDICTION_DEFAULTER"] = predictions
    final_features = df.to_dict(orient='records')

    # insert into  db here
    for input in final_features:
        #prediction = random.randint(0, 1)
        prediction = input['PREDICTION_DEFAULTER']

        columns = list(input.keys())
        columns.append("PredictionDate")
        columns.append("Cancelled")
        columns = ', '.join(columns)

        # ********************** TO ADD DATABASE WRITE QUERY ***************************
        #
        #
        #

        #query_command = f'INSERT INTO Predictions ({columns}) VALUES ({values});'
        
        #database_write(query_command)

        input['DatePrediction'] = str(current_date)

    json_data = json.dumps(final_features)

    return json_data


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)

@app.get("/get_unique_vals")
def get_unique_values():
    uniq_val_dict = joblib.load("Model/credit-cat-cols-uniq-vals.joblib")
    json_data = json.dumps(uniq_val_dict, cls=NpEncoder)
    return json_data

@app.get("/get_features")
def get_feature_sets():
    fs_list = joblib.load("Model/credit-features.joblib")
    json_data = jsonable_encoder(fs_list)
    return json_data


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)