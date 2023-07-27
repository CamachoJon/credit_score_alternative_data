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
import matplotlib.pyplot as plt
import io
from fastapi.responses import FileResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import shap
shap.initjs()


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

@app.get('/get_user_info')
async def get_user_info(name: str = '', lastname: str = ''):

    query = f'''SELECT us.[TARGET]
      ,us.[CODE_GENDER]
      ,us.[FLAG_OWN_CAR]
      ,us.[FLAG_OWN_REALTY]
      ,us.[CNT_CHILDREN]
      ,us.[AMT_INCOME_TOTAL]
      ,us.[AMT_CREDIT]
      ,us.[AMT_ANNUITY]
      ,us.[AMT_GOODS_PRICE]
      ,us.[NAME_TYPE_SUITE]
      ,us.[NAME_INCOME_TYPE]
      ,us.[NAME_EDUCATION_TYPE]
      ,us.[NAME_FAMILY_STATUS]
      ,us.[NAME_HOUSING_TYPE]
      ,us.[REGION_POPULATION_RELATIVE]
      ,us.[DAYS_BIRTH]
      ,us.[DAYS_EMPLOYED]
      ,us.[DAYS_REGISTRATION]
      ,us.[DAYS_ID_PUBLISH]
      ,us.[OWN_CAR_AGE]
      ,us.[OCCUPATION_TYPE]
      ,us.[CNT_FAM_MEMBERS]
      ,us.[REGION_RATING_CLIENT]
      ,us.[REGION_RATING_CLIENT_W_CITY]
      ,us.[WEEKDAY_APPR_PROCESS_START]
      ,us.[HOUR_APPR_PROCESS_START]
      ,us.[ORGANIZATION_TYPE]
      ,us.[EXT_SOURCE_1]
      ,us.[EXT_SOURCE_2]
      ,us.[EXT_SOURCE_3]
      ,us.[EMERGENCYSTATE_MODE]
      ,us.[OBS_30_CNT_SOCIAL_CIRCLE]
      ,us.[DEF_30_CNT_SOCIAL_CIRCLE]
      ,us.[OBS_60_CNT_SOCIAL_CIRCLE]
      ,us.[DEF_60_CNT_SOCIAL_CIRCLE]
      ,us.[DAYS_LAST_PHONE_CHANGE]
      ,us.[AMT_REQ_CREDIT_BUREAU_HOUR]
      ,us.[AMT_REQ_CREDIT_BUREAU_DAY]
      ,us.[AMT_REQ_CREDIT_BUREAU_WEEK]
      ,us.[AMT_REQ_CREDIT_BUREAU_MON]
      ,us.[AMT_REQ_CREDIT_BUREAU_QRT]
      ,us.[AMT_REQ_CREDIT_BUREAU_YEAR]
      ,us.[DATE] 
      FROM [dbo].[USERS_INFO] AS ui
      INNER JOIN USERS us
      ON ui.ID = us.ID
      WHERE ui.NAME = \'{name}\' AND ui.LASTNAME = \'{lastname}\''''
    
    db = Database()
    results = db.read(query)

    results.replace([np.inf, -np.inf], np.nan, inplace=True)
    results.fillna(value="NaN", inplace=True)  # use a string to represent NaN

    # convert results to JSON
    results_json = results.to_json(orient="records")

    # return past predictions as JSON
    return jsonable_encoder(results_json)

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

# @app.post("/generate_decision_plot")
# async def generate_decision_plot(request: Request):
#     data = await request.json()
#     instance = data['instance']  # The instance you want to explain
#     instance_df = pd.DataFrame([instance])

#     # Load your model
#     model = joblib.load('/app/Model/xgb_model.joblib')

#     # Initialize the explainer
#     explainer = shap.TreeExplainer(model)

#     # Calculate SHAP values
#     try:
#         shap_values = explainer.shap_values(instance_df)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

#     # Generate the decision plot and save it to a file
#     fig, ax = plt.subplots()
#     shap.decision_plot(explainer.expected_value, shap_values[0], instance_df, show=False)
#     plt.savefig("shap_plot.png")

#     # Create a PDF
#     buffer = io.BytesIO()
#     doc = SimpleDocTemplate(buffer, pagesize=letter)
#     story = []
#     styles = getSampleStyleSheet()

#     # Add the SHAP plot
#     story.append(Image("shap_plot.png", width=500, height=400))
#     story.append(Spacer(1, 12))
    
#     # Add some text
#     text = "<b>SHAP Decision Plot</b><br/>This plot provides a detailed view of the feature contributions to the model prediction for a single instance."
#     story.append(Paragraph(text, styles["Normal"]))
#     story.append(Spacer(1, 12))

#     # Generate the PDF
#     doc.build(story)

#     # Return the PDF as a response
#     buffer.seek(0)
#     return FileResponse(buffer, media_type="application/pdf", filename="report.pdf")

# demo TODO: Remove it and update the correct one
@app.get("/generate_decision_plot")
async def generate_decision_plot():

    # Let's generate a random instance with 10 features
    instance = np.random.randn(10)
    instance_df = pd.DataFrame([instance], columns=[f'feature_{i}' for i in range(10)])

    # We're generating random SHAP values and an expected value
    shap_values = [np.random.randn(instance_df.shape[1])]
    expected_value = np.random.randn(1)

    # Generate the decision plot and save it to a file
    fig, ax = plt.subplots()
    shap.decision_plot(expected_value, shap_values[0], instance_df, link='logit', show=False)
    plt.savefig("shap_plot.png")

    # Create a PDF
    doc = SimpleDocTemplate("report.pdf", pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Add the SHAP plot
    story.append(Image("shap_plot.png", width=500, height=400))
    story.append(Spacer(1, 12))
    
    # Add some text
    text = "<b>SHAP Decision Plot</b><br/>This plot provides a detailed view of the feature contributions to the model prediction for a single instance."
    story.append(Paragraph(text, styles["Normal"]))
    story.append(Spacer(1, 12))

    # Generate the PDF
    doc.build(story)

    # Return the PDF as a response
    return FileResponse("report.pdf", media_type="application/pdf", filename="report.pdf")

@app.get('/get_unique_vals')
async def get_unique_vals():
    unique_vals = joblib.load('/app/Model/credit-cat-cols-uniq-vals.joblib')
    json_data = json.dumps(unique_vals, cls=NpEncoder)
    return json_data


def prepare_data(features: List[Dict[str, Union[str, int, float]]]) -> pd.DataFrame:
    df = pd.DataFrame(features)
    data_preparation = DataPreparation(df)
    processed_df = data_preparation.prepare_data()
    return processed_df


def make_prediction(df: pd.DataFrame) -> List:
    model = joblib.load('/app/Model/credit-cat-cols-uniq-vals.joblib')
    predictions = model.predict(df).tolist()
    return predictions


def add_target_and_date(df: pd.DataFrame, predictions: List) -> pd.DataFrame:
    df.insert(0, 'TARGET', predictions)
    df['DATE'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return df


def write_to_db(df: pd.DataFrame, db: Database) -> None:
    dict_list = df.to_dict(orient='records')
    for record in dict_list:
        fake_data = generate_fake_data()
        user_info_query = Database.format_sql_command('USERS_INFO', fake_data)
        user_analysis_query = Database.format_sql_command('USERS', record)
        db.write(user_analysis_query)
        # db.write(user_info_query)


def generate_fake_data():
    fake = Faker()
    fake_name = fake.name()
    fake_lastname = fake.last_name()
    fake_birthdate = fake.date_of_birth().strftime("%Y-%m-%d")

    data_dict = {
        'NAME': fake_name,
        'LASTNAME': fake_lastname,
        'BIRTHDATE': fake_birthdate
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
