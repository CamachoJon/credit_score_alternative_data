import pandas as pd
import numpy as np
import joblib
import os
from credit_risk import NUM_COLS, CAT_COLS, MODEL_DIR, TARGET, BOOL_COLS
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def get_scaler(x: pd.DataFrame, training: bool = True) -> StandardScaler:
    """
    Scaling the features X
    :param x:
    :param training: (True - Training, False - Testing)
    :return: StandardScaler after fitting
    """
    SCALER_DIR = os.path.join(MODEL_DIR, "scaler.joblib")
    if training is True:
        scaler = StandardScaler()
        scaler.fit(x)
        joblib.dump(scaler, SCALER_DIR)
    else:
        if os.path.isfile(SCALER_DIR):
            scaler = joblib.load(SCALER_DIR)
        else:
            raise FileNotFoundError
    return scaler


def get_encoder(x: pd.DataFrame, training: bool = True) -> OneHotEncoder:
    """
    Encoding the features X
    :param x:
    :param training: (True - Training, False - Testing)
    :return: OneHotEncoder after fitting
    """
    ENCODER_DIR = os.path.join(MODEL_DIR, "encoder.joblib")
    if training is True:
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(x)
        joblib.dump(encoder, ENCODER_DIR)
    else:
        if os.path.isfile(ENCODER_DIR):
            encoder = joblib.load(ENCODER_DIR)
        else:
            raise FileNotFoundError
    return encoder


def map_features(x:pd.DataFrame) -> pd.DataFrame:
    """
    To transform the features to boolean cols (yes/no, gender)
    :param x: DataFrame
    :return: processed_x
    """
    mapping_YN = {'N': 0, 'Y': 1, 'No': 0, 'Yes': 1}
    mapping_gender = {'M': 0, 'F': 1}

    for col in BOOL_COLS:
        x.loc[:,col] = x[col].map(mapping_YN).fillna(0).astype(float).astype(int)
    
    x['CODE_GENDER'] = x['CODE_GENDER'].map(mapping_gender).fillna(0).astype(float).astype(int)
    x['EMERGENCYSTATE_MODE'] = x['EMERGENCYSTATE_MODE'].fillna(0).astype(float).astype(int)

    return x


def get_cyclical_day_encoding(x:pd.DataFrame) -> pd.DataFrame:
    """
    To transform the day features using cyclical encoding
    :param x: DataFrame
    :return: processed_x
    """
    day_x = pd.DataFrame(x['WEEKDAY_APPR_PROCESS_START'])

    mapping = {'MONDAY': 1, 'TUESDAY': 2, 'WEDNESDAY': 3, 'THURSDAY': 4, 'FRIDAY': 5, 'SATURDAY': 6, 'SUNDAY': 7}

    day_x.loc[:,'WEEKDAY_APPR_PROCESS_START'] = day_x['WEEKDAY_APPR_PROCESS_START'].map(mapping)

    day_x.loc[:,'DAY_WEEK_SIN'] = np.sin(day_x['WEEKDAY_APPR_PROCESS_START'] * (2 * np.pi / 7))
    day_x.loc[:,'DAY_WEEK_COS'] = np.cos(day_x['WEEKDAY_APPR_PROCESS_START'] * (2 * np.pi / 7))

    day_x = day_x.drop(columns=['WEEKDAY_APPR_PROCESS_START'])

    return day_x


def get_cyclical_hour_encoding(x:pd.DataFrame) -> pd.DataFrame:
    """
    To transform the Hour features using cyclical encoding
    :param x: DataFrame
    :return: processed_x
    """
    hour_x = pd.DataFrame(x['HOUR_APPR_PROCESS_START'], dtype=int)
    
    # Convert the hour (in 24h format) to a number between 0 and 1, and multiply it by 2*pi to convert it to radians
    hour_x.loc[:,'HOUR_APPR_PROCESS_START_rad'] = hour_x['HOUR_APPR_PROCESS_START'] / 24 * 2 * np.pi

    # Create the two new features using sine and cosine
    hour_x.loc[:,'HOUR_APPR_PROCESS_START_sin'] = np.sin(hour_x['HOUR_APPR_PROCESS_START_rad'])
    hour_x.loc[:,'HOUR_APPR_PROCESS_START_cos'] = np.cos(hour_x['HOUR_APPR_PROCESS_START_rad'])

    # Drop the original 'HOUR_APPR_PROCESS_START' column and the intermediary radians column
    hour_x = hour_x.drop(['HOUR_APPR_PROCESS_START_rad'], axis=1)

    return hour_x


def feature_engineering(x: pd.DataFrame,
                        training: bool = True) -> (pd.DataFrame):
    """
    To transform the features using encoders and scalers
    :param x: DataFrame
    :param y: DataFrame
    :param training: (True - Training, False - Testing)
    :return: processed_x
    """
    try:

        # Numerical cols
        xscaler = get_scaler(x[NUM_COLS], training)
        x_scl = xscaler.transform(x[NUM_COLS])
        x_scl_df = pd.DataFrame(x_scl, columns=NUM_COLS)

        # Categorical cols
        encoder = get_encoder(x[CAT_COLS].apply(lambda x: x.fillna(x.mode()[0]), axis=0), training)
        x_enc = encoder.transform(x[CAT_COLS])
        x_enc_df = pd.DataFrame(x_enc.todense(),
                                columns=encoder.get_feature_names_out())

        # Map yes/no cols
        x_bool = map_features(x[BOOL_COLS])

        # day encoding - cyclical
        x_day = get_cyclical_day_encoding(x)
        
        # hour encoding - cyclical
        x_hour = get_cyclical_hour_encoding(x)


    except Exception as e:
        print(e)

    temp_x_1 = x_scl_df.join(x_enc_df)
    temp_x_2 = temp_x_1.join(x_bool)
    temp_x_3 = temp_x_2.join(x_day)
    new_x = temp_x_3.join(x_hour)

    return new_x