import tempfile
from PIL import Image as PILImage
import shap
import shap_service
import requests
from io import BytesIO
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as platypusImage
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from fastapi.responses import FileResponse
import io
import matplotlib.pyplot as plt
import datetime
from faker import Faker
import json
from fastapi import FastAPI, UploadFile, File, Form
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
shap.initjs()


# define the app and the base URL
app = FastAPI()


class names(BaseModel):
    key1: str
    key2: str


@app.get('/user_data', response_model=List[Dict])
async def get_user_data():
    """
    Fetches user data from the database.

    Returns:
        List[Dict]: A list of dictionaries representing user data records.
    """
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
    """
    Retrieves user information based on the given name and last name.

    Args:
        name (str): The first name of the user.
        lastname (str): The last name of the user.

    Returns:
        str: JSON string containing the user information.
    """
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
    """
    Root endpoint of the API.

    Returns:
        Dict[str, str]: A dictionary containing a welcome message.
    """
    return {"message": "This is the root route of the API."}


@app.post("/predict")
async def predict(features: List[Dict[str, Union[str, int, float]]]) -> None:
    """
    Predicts the credit risk based on the provided features.

    Args:
        features (List[Dict[str, Union[str, int, float]]]): List of dictionaries containing user data features.

    Returns:
        str: JSON string containing the predicted credit risk and user data.
    """
    db = Database()
    og_df = pd.DataFrame(features)
    og_df = DataPreparation.remove_unnecessary_cols(og_df)
    df = prepare_data(features)
    predictions = make_prediction(df)
    og_df = add_target_and_date(og_df, predictions)
    write_to_db(og_df, db)
    final_df = og_df.to_dict(orient='records')
    json_data = json.dumps(final_df)

    return json_data


@app.post("/generate_report")
async def generate_report(name: str = Form(...), lastname: str = Form(...), imp_f: str = Form(...), cat: str = Form(...), image: UploadFile = File(...)):
    """
    Generates a PDF report with credit risk analysis results.

    Args:
        name (str): The first name of the user.
        lastname (str): The last name of the user.
        imp_f (str): Comma-separated string of important features for credit risk analysis.
        cat (str): Category of the user for credit risk analysis.
        image (UploadFile): Uploaded image file used in the report.

    Returns:
        FileResponse: PDF file response with the generated report.
    """    
    imp_f_list = imp_f.split(",")

    # Read the uploaded image file
    image_content = await image.read()
    print(f'Personal data {name}, {lastname}, {cat}')
    # Convert image_content into an Image object
    pil_image = PILImage.open(BytesIO(image_content))

    # Save PIL Image to temporary file
    temp_file_name = "temp.png"
    pil_image.save(temp_file_name)

    # Create a PDF
    doc = SimpleDocTemplate("report.pdf", pagesize=letter)
    story = []

    styles = getSampleStyleSheet()

    title_style = styles['Title']
    heading_style = styles['Heading2']
    paragraph_style = styles['Normal']

    title = Paragraph("Credit Risk Analysis", title_style)
    h1 = Paragraph(f"User: {name} {lastname}", heading_style)
    h1_1 = Paragraph(f"Category: {cat}", heading_style)
    h2 = Paragraph(
        "Understanding our Credit Risk Analysis System", heading_style)
    p1 = Paragraph("In our system, various factors are ranked based on their significance in determining credit risk. \
                The middle line represents a neutral point (0.5). \
                Any value between 0 and 0.5 indicates a lower risk (non-defaulter), and \
                any value between 0.5 and 1 indicates a higher risk (defaulter).", paragraph_style)
    h3 = Paragraph(
        "Most Significant Factors in this Risk Assessment: ", heading_style)

    # Add the SHAP plot
    story.append(title)
    story.append(h1)
    story.append(h1_1)
    story.append(h2)
    story.append(p1)
    # this line now uses the temporary file
    story.append(platypusImage(temp_file_name, width=300, height=250))
    story.append(Spacer(1, 12))
    story.append(h3)
    for i in range(len(imp_f_list)):
        story.append(Paragraph(f"* {imp_f_list[i]}", paragraph_style))

    story.append(Spacer(1, 12))

    # Generate the PDF
    doc.build(story)

    # Delete the temporary file
    os.remove(temp_file_name)

    # Return the PDF as a response
    return FileResponse("report.pdf", media_type="application/pdf", filename="report.pdf")


@app.get('/get_unique_vals')
async def get_unique_vals():
    """
    Retrieves unique values for categorical columns used in credit risk analysis.

    Returns:
        str: JSON string containing unique values for categorical columns.
    """
    unique_vals = joblib.load('Model/credit-cat-cols-uniq-vals.joblib')
    json_data = json.dumps(unique_vals, cls=NpEncoder)
    return json_data


def prepare_data(features: List[Dict[str, Union[str, int, float]]]) -> pd.DataFrame:
    """
    Prepares the user data for credit risk analysis.

    Args:
        features (List[Dict[str, Union[str, int, float]]]): List of dictionaries containing user data features.

    Returns:
        pd.DataFrame: Processed DataFrame for credit risk analysis.
    """
    df = pd.DataFrame(features)
    data_preparation = DataPreparation(df)
    processed_df = data_preparation.prepare_data()
    return processed_df


def make_prediction(df: pd.DataFrame) -> List:
    """
    Makes predictions for credit risk based on the provided DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing user data features.

    Returns:
        List: List of predictions (credit risk values).
    """
    try:
        model = joblib.load('/app/Model/xgb_model.joblib')
        joblib.dump(df, "app/Model/x_test_proc_shap.joblib")
    except:
        model = joblib.load('Model/xgb_model.joblib')
        joblib.dump(df, "Model/x_test_proc_shap.joblib")
    predictions = model.predict(df).tolist()
    return predictions


def add_target_and_date(df: pd.DataFrame, predictions: List) -> pd.DataFrame:
    """
    Adds the target (credit risk predictions) and date to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing user data features.
        predictions (List): List of predictions (credit risk values).

    Returns:
        pd.DataFrame: DataFrame with added target and date columns.
    """
    df.insert(0, 'TARGET', predictions)
    df['DATE'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return df


def write_to_db(df: pd.DataFrame, db: Database) -> None:
    """
    Writes user analysis and user info to the database.

    Args:
        df (pd.DataFrame): DataFrame containing user data features and credit risk predictions.
        db (Database): Database connection object.

    Returns:
        None
    """
    dict_list = df.to_dict(orient='records')
    for record in dict_list:
        user_analysis_query = Database.format_sql_command('USERS', record)
        user_id = db.write(user_analysis_query)
        fake_data = generate_fake_data(user_id)
        user_info_query = Database.format_sql_command('USERS_INFO', fake_data)
        db.write(user_info_query)


def generate_fake_data(user_id):
    """
    Generates fake user data.

    Args:
        user_id: User ID.

    Returns:
        dict: Dictionary containing fake user data (name, lastname, birthdate).
    """
    fake = Faker()
    fake_name = fake.name()
    fake_lastname = fake.last_name()
    fake_birthdate = fake.date_of_birth().strftime("%Y-%m-%d")

    data_dict = {
        'ID': user_id,
        'NAME': fake_name,
        'LASTNAME': fake_lastname,
        'BIRTHDATE': fake_birthdate
    }
    return data_dict


class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy data types.

    This class is used to ensure JSON serialization of NumPy data types.

    """
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


@app.get("/shap")
async def get_shap_values():
    """
    Retrieves SHAP (SHapley Additive exPlanations) values for the credit risk analysis model.

    Returns:
        str: JSON string containing SHAP values, expected value, x_test, class_0, and class_1.
    """
    shap_values, expected_value, x_test_processed, class_0, class_1 = shap_service.create_explainer()
    sv = shap_values.tolist()
    ev = expected_value.tolist()
    xpr = x_test_processed.to_dict(orient='records')
    pr_df = json.dumps(xpr)

    jd = {
        "shap_val": sv,
        "exp_val": ev,
        "x_test": pr_df,
        "class_0": class_0,
        "class_1": class_1
    }
    json_data = json.dumps(jd)
    return json_data

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
