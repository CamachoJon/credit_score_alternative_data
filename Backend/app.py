import datetime
from faker import Faker
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
from datetime import datetime
from typing import List, Dict, Union
import random
from fastapi.encoders import jsonable_encoder
from Database import Database
from DataPreparation import DataPreparation
import sys
sys.path.append("..")


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
async def predict(features: List[Dict[str, Union[str, int, float]]]) -> None:
    db = Database()
    og_df = pd.DataFrame(features)
    og_df = DataPreparation.remove_unnecessary_cols(og_df)
    df = prepare_data(features)
    predictions = make_prediction(df)
    og_df = add_target_and_date(og_df, predictions)
    write_to_db(og_df, db)
    response = og_df.to_dict()
    
    return response

@app.get("/generate_report")
async def generate_report():
    pass

def prepare_data(features: List[Dict[str, Union[str, int, float]]]) -> pd.DataFrame:
    df = pd.DataFrame(features)
    data_preparation = DataPreparation(df)
    processed_df = data_preparation.prepare_data()
    return processed_df


def make_prediction(df: pd.DataFrame) -> List:
    model = joblib.load('/app/Model/xgb_model.joblib')
    predictions = model.predict(df).tolist()
    return predictions


def add_target_and_date(df: pd.DataFrame, predictions: List) -> pd.DataFrame:
    df.insert(0, 'TARGET', predictions)
    df['DATE'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return df


def write_to_db(df: pd.DataFrame, db: Database) -> None:
    dict_list = df.to_dict(orient='records')
    for record in dict_list:
        # fake_data = generate_fake_data()
        # user_info_query = Database.format_sql_command('USERS_INFO', fake_data)
        user_analysis_query = Database.format_sql_command('USERS', record)
        db.write(user_analysis_query)
        # db.write(user_info_query)


def generate_fake_data():
    fake = Faker()
    fake_name = fake.name()
    fake_lastname = fake.last_name()
    fake_birthdate = fake.date_of_birth().strftime("%Y-%m-%d")

    data_dict = {
        'Name': fake_name,
        'LastName': fake_lastname,
        'Birthdate': fake_birthdate
    }
    return data_dict


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


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
