# import the necessary libraries at the top
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import joblib


class DataPreparation:

    def __init__(self, input_df):
        self.input_df = input_df

        # Features classification
        self.unused_features = ['NAME_CONTRACT_TYPE', 'SK_ID_CURR', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',
                                'WALLSMATERIAL_MODE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
                                'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
                                'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
                                'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                                'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
                                'FLAG_DOCUMENT_21',
                                'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
                                'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG',
                                'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
                                'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
                                'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',
                                'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',
                                'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI',
                                'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI',
                                'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI',
                                'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
                                'TOTALAREA_MODE']

        self.categorical_features = ['NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                                     'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']

        self.numerical_features = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                                   'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
                                   'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
                                   'REGION_RATING_CLIENT_W_CITY', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
                                   'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_HOUR',
                                   'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                                   'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

        self.cyclical_features = [
            'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START']

        self.yes_no_features = ['FLAG_OWN_CAR',
                                'FLAG_OWN_REALTY', 'EMERGENCYSTATE_MODE', 'CODE_GENDER']

        self.binary_features = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'REG_REGION_NOT_LIVE_REGION',
                                'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']

    def drop_features(self, df):
        # df = df.drop(self.unused_features, axis=1)
        # df = df.drop(self.binary_features, axis=1)
        FEATURES = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
       'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
       'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE',
       'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
       'OWN_CAR_AGE', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS',
       'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
       'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',
       'ORGANIZATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'EMERGENCYSTATE_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE',
       'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
       'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE',
       'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
        new_df = df[FEATURES]
        return new_df

    def map_features(self, df):
        mapping_YN = {'N': 0, 'Y': 1, 'No': 0, 'Yes': 1}
        mapping_gender = {'M': 0, 'F': 1}

        for col in self.yes_no_features:
            df[col] = df[col].map(mapping_YN).fillna(
                0).astype(float).astype(int)

        df['CODE_GENDER'] = df['CODE_GENDER'].map(
            mapping_gender).fillna(0).astype(float).astype(int)
        df['EMERGENCYSTATE_MODE'] = df['EMERGENCYSTATE_MODE'].fillna(
            0).astype(float).astype(int)

        return df

    def one_hot_encoding(self, df):
        try:
            encoder = joblib.load('/app/Model/onehotencoder.joblib')
        except Exception as e:
            encoder = joblib.load('Model/onehotencoder.joblib')

        # Create a copy of the DataFrame to work with
        encoded_df = df.copy()
        encoded_df.to_csv('ohencoded.csv')
        # Fill missing values in categorical features

        if len(encoded_df) > 1:            
            encoded_df[self.categorical_features] = encoded_df[self.categorical_features].apply(
                lambda x: x.fillna(x.mode()[0]), axis=0)
        else:
            encoded_df[self.categorical_features] = encoded_df[self.categorical_features].apply(
                lambda x: x.fillna('No_Ans', axis=0))
            
        
        # Transform the data
        encoded = encoder.transform(encoded_df[self.categorical_features])

        # Get the feature names
        feature_names = encoder.get_feature_names_out()

        # Create a DataFrame with the encoded values
        ohe_df = pd.DataFrame(encoded.toarray(), columns=feature_names)
        ohe_df = ohe_df.astype(int)

        # If there are unseen categories, add them as new columns filled with zeros
        unseen_categories = set(feature_names) - set(ohe_df.columns)
        for category in unseen_categories:
            ohe_df[category] = 0

        # Drop the categorical features from the encoded DataFrame
        encoded_df = encoded_df.drop(columns=self.categorical_features)

        # Reset indices of the DataFrames
        encoded_df.reset_index(drop=True, inplace=True)
        ohe_df.reset_index(drop=True, inplace=True)

        # Concatenate the encoded DataFrame with the remaining features
        # return pd.concat([encoded_df, ohe_df.reindex(sorted(ohe_df.columns), axis=1)], axis=1)
        return pd.concat([encoded_df, ohe_df], axis=1)

    def regression_filler_1(self, df):
        data = df.loc[:, ["EXT_SOURCE_1",
                          "EXT_SOURCE_2", "EXT_SOURCE_3"]].dropna()
        x_filler = data.drop(columns="EXT_SOURCE_1")
        y_filler = data["EXT_SOURCE_1"]

        scaler = StandardScaler()
        x_filler_scaled = scaler.fit_transform(x_filler)

        reg_model = Ridge(alpha=0.001)
        reg_model.fit(x_filler_scaled, y_filler)

        mask = (~(df["EXT_SOURCE_2"].isna())) & (
            ~(df["EXT_SOURCE_3"].isna())) & (df["EXT_SOURCE_1"].isna())
        x_pred = df.loc[mask, ["EXT_SOURCE_2", "EXT_SOURCE_3"]]
        x_pred_scaled = scaler.transform(x_pred)
        df.loc[mask, "EXT_SOURCE_1"] = reg_model.predict(x_pred_scaled)

        return df

    def regression_filler_2(self, df):
        data = df.loc[:, ["EXT_SOURCE_1",
                          "EXT_SOURCE_2", "EXT_SOURCE_3"]].dropna()
        x_filler = data.drop(columns="EXT_SOURCE_3")
        y_filler = data["EXT_SOURCE_3"]

        scaler = StandardScaler()
        x_filler_scaled = scaler.fit_transform(x_filler)

        reg_model = Ridge(alpha=0.001)
        reg_model.fit(x_filler_scaled, y_filler)

        mask = (~(df["EXT_SOURCE_2"].isna())) & (
            ~(df["EXT_SOURCE_1"].isna())) & (df["EXT_SOURCE_3"].isna())
        x_pred = df.loc[mask, ["EXT_SOURCE_1", "EXT_SOURCE_2"]]
        x_pred_scaled = scaler.transform(x_pred)
        df.loc[mask, "EXT_SOURCE_3"] = reg_model.predict(x_pred_scaled)

        return df

    def scale_numerical_features(self, df):
        try:
            scaler = joblib.load('/app/Model/standardscaler.joblib')
        except Exception as e:
            scaler = joblib.load('Model/standardscaler.joblib')
        # Create a copy of the DataFrame to work with
        scaled_df = df.copy()

        # Fill missing values in numerical features
        scaled_df[self.numerical_features] = scaled_df[self.numerical_features].apply(
            lambda x: x.fillna(x.median()), axis=0)

        # Scaling features
        scaled_features = scaler.transform(scaled_df[self.numerical_features])

        # Create a new DataFrame with scaled values
        scaled_features_df = pd.DataFrame(
            scaled_features, columns=self.numerical_features)

        # Drop the numerical features from the scaled DataFrame
        scaled_df = scaled_df.drop(columns=self.numerical_features)

        # Reset indices of the DataFrames
        scaled_df.reset_index(drop=True, inplace=True)
        scaled_features_df.reset_index(drop=True, inplace=True)

        # Concatenate the scaled DataFrame with the remaining features
        return pd.concat([scaled_df, scaled_features_df], axis=1)

    def cyclic_encoding_day(self, df):
        mapping = {'MONDAY': 1, 'TUESDAY': 2, 'WEDNESDAY': 3,
                   'THURSDAY': 4, 'FRIDAY': 5, 'SATURDAY': 6, 'SUNDAY': 7}

        df['WEEKDAY_APPR_PROCESS_START'] = df['WEEKDAY_APPR_PROCESS_START'].map(
            mapping)

        df['DAY_WEEK_SIN'] = np.sin(
            df['WEEKDAY_APPR_PROCESS_START'] * (2 * np.pi / 7))
        df['DAY_WEEK_COS'] = np.cos(
            df['WEEKDAY_APPR_PROCESS_START'] * (2 * np.pi / 7))

        df = df.drop(columns=['WEEKDAY_APPR_PROCESS_START'])
        return df

    def cyclic_encoding_hour(self, df):
        # Convert the hour (in 24h format) to a number between 0 and 1, and multiply it by 2*pi to convert it to radians
        df['HOUR_APPR_PROCESS_START'] = pd.Series(df['HOUR_APPR_PROCESS_START'], dtype=int)
        df['HOUR_APPR_PROCESS_START_rad'] = df['HOUR_APPR_PROCESS_START'] / \
            24. * 2 * np.pi

        # Create the two new features using sine and cosine
        df['HOUR_APPR_PROCESS_START_sin'] = np.sin(
            df['HOUR_APPR_PROCESS_START_rad'])
        df['HOUR_APPR_PROCESS_START_cos'] = np.cos(
            df['HOUR_APPR_PROCESS_START_rad'])

        # Drop the original 'HOUR_APPR_PROCESS_START' column and the intermediary radians column
        df = df.drop(['HOUR_APPR_PROCESS_START_rad'], axis=1)

        return df

    def prepare_data(self):

        if 'TARGET' in self.input_df.columns:
            self.input_df.drop('TARGET', axis=1, inplace=True)

        self.input_df = self.input_df.drop(self.input_df[self.input_df['NAME_INCOME_TYPE'] == 'Unemployed'].index, inplace=False)
        self.input_df = self.input_df.drop(self.input_df[self.input_df['NAME_INCOME_TYPE'] == 'Student'].index, inplace=False)
        self.input_df = self.input_df.drop(self.input_df[self.input_df['NAME_INCOME_TYPE'] == 'Businessman'].index, inplace=False)
        self.input_df = self.input_df.drop(self.input_df[self.input_df['NAME_INCOME_TYPE'] == 'Maternity leave'].index, inplace=False)

        self.input_df.replace('NaN', np.nan, inplace=True)
        self.input_df = self.drop_features(self.input_df)
        #self.input_df = self.regression_filler_1(self.input_df)
        #self.input_df = self.regression_filler_2(self.input_df)
        self.input_df = self.map_features(self.input_df)
        self.input_df = self.one_hot_encoding(self.input_df)
        self.input_df = self.scale_numerical_features(self.input_df)
        self.input_df = self.cyclic_encoding_day(self.input_df)
        self.input_df = self.cyclic_encoding_hour(self.input_df)

        return self.input_df

    @staticmethod
    def remove_unnecessary_cols(df):
        # # Features classification
        # unused_features = ['NAME_CONTRACT_TYPE', 'SK_ID_CURR', 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE',
        #                         'WALLSMATERIAL_MODE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
        #                         'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
        #                         'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
        #                         'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
        #                         'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
        #                         'FLAG_DOCUMENT_21',
        #                         'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
        #                         'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG',
        #                         'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
        #                         'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
        #                         'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE',
        #                         'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',
        #                         'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI',
        #                         'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI',
        #                         'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI',
        #                         'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
        #                         'TOTALAREA_MODE']

        # binary_features = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'REG_REGION_NOT_LIVE_REGION',
        #                         'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']

        FEATURES = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
       'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
       'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE',
       'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
       'OWN_CAR_AGE', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS',
       'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
       'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',
       'ORGANIZATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'EMERGENCYSTATE_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE',
       'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
       'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE',
       'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
        
        if 'TARGET' in df.columns:
            df.drop('TARGET', axis=1, inplace=True)
            
        # df = df.drop(unused_features, axis=1)
        # df = df.drop(binary_features, axis=1)

        new_df = df[FEATURES]
        print('All G')

        return new_df