# Credit Analysis - TShield
![Tsheild logo](https://github.com/CamachoJon/credit_score_alternative_data/assets/42679048/4df4e077-ebeb-4006-b2c5-05fc94fe348e)

## Project Description 
Traditional credit risk analysis based on credit history, financial metrics, and personal information has limitations in coverage, timeliness, and inherent biases. It often fails to assess risk accurately for 'thin-file' or 'no-file' customers, lacks up-to-date information, perpetuates biases, overlooks relevant contextual data, and struggles to capture dynamic financial circumstances. In contrast, alternative data from non-traditional sources like digital footprints and social media can offer a more inclusive and comprehensive view of creditworthiness, despite privacy and data quality challenges.​

To address these shortcomings, this credit analysis application aims to empower financial institutions by leveraging alternative data analysis to better assess the creditworthiness of their customers.​


## Dataset used
The dataset is taken from HomeCredit Financial institution https://www.kaggle.com/c/home-credit-default-risk/data?select=credit_card_balance.csv\
Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.\


## Technologies used
* WebApp – Streamlit as FrontEnd and FastApi as backend
* Airflow – Automates data ingestion job and prediction job in a period of interval
* Azure SQL – Database to store all user information
* SHAP – prediction analysis based on individual feature


## Architecture
<img width="608" alt="image" src="https://github.com/CamachoJon/credit_score_alternative_data/assets/42679048/76c18abe-45bd-4e8f-a342-100cef216b1b">


## Machine learning model used
In this study, we explored the application of 4 different models: Logistic Regression, SVM, XGBoost, and Neural networks to predict the likelihood of credit default. Several factors are considered in the preprocessing of the data such as data imbalance, correlation, data quality, and feature selection. 
 
After experimentation and analysis, XGBoost demonstrated a higher accuracy in classification. Further research and improvement are necessary to develop robust and accurate models for practical implementation in the financial industry.


## Application Development
* We developed a web application that, using alternative data, delivers a comprehensive evaluation of a client's creditworthiness. Moreover, it offers strategic insights into the areas requiring enhancement for credit procurement, as well as the aspects demanding careful attention to mitigate the risk of a detrimental impact on their credit score.

<img width="613" alt="image" src="https://github.com/CamachoJon/credit_score_alternative_data/assets/42679048/6a2a8a90-a4d4-4393-a3a0-f4d91ac46f40">

* After meticulous feature engineering, we selected XGBoost as our model for processing new user data.

* The application generates User Profiling Reports, offering a graphical representation of key client distributions over a specific timeframe.

* It incorporates a real-time analysis feature for the historical data of approved credits.

* The application has the capability to search for a specific client, retrieve their data from the database, and provide a comprehensive report with clear graphical representations.

<img width="556" alt="image" src="https://github.com/CamachoJon/credit_score_alternative_data/assets/42679048/ca1350db-cd1b-40f6-8178-cc0dd4859b12">

* It offers the functionality to access user data in real time for inference, subsequently providing analytical output.

<img width="382" alt="image" src="https://github.com/CamachoJon/credit_score_alternative_data/assets/42679048/ad261a2e-1c48-446a-a929-45c984a8a92d">

<img width="647" alt="image" src="https://github.com/CamachoJon/credit_score_alternative_data/assets/42679048/0c9d8ebc-80f2-4c7e-b796-22194cbb8c1a">

* We have designed an automated process to assess the creditworthiness of new clients in batch operations.


