import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import time
import json
import requests
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from Frontend import UNIQ_VAL_URL, FEATURE_URL, PREDICT_URL, PAST_PREDICT_URL

st.set_page_config(layout="wide")

with st.sidebar:
    selected = option_menu(
        menu_title="Credit Scoring",
        options=["Home", "User Input", "Reports", "Prediction"]
    )

with st.container():
    if selected == "Home":
        st.title("User Profiling Reports")
        col1, col2 = st.columns(2)

        with col1:
            chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['a', 'b', 'c'])

            st.line_chart(chart_data)

        with col2:
            chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['a', 'b', 'c'])

            st.area_chart(chart_data)
        

    if selected == "User Input":
        with st.container():
            x1 = np.random.randn(200) - 2
            x2 = np.random.randn(200)
            x3 = np.random.randn(200) + 2

            # Group data together
            hist_data = [x1, x2, x3]

            group_labels = ['Group 1', 'Group 2', 'Group 3']

            # Create distplot with custom bin_size
            fig = ff.create_distplot(
                    hist_data, group_labels, bin_size=[.1, .25, .5])

            # Plot!
            st.plotly_chart(fig, use_container_width=True)
            
    if selected == "Reports":
        st.title(f"You have selected: {selected}")

    if selected == "Prediction":
        # To get unique values of categorical columns
        uv = requests.get(UNIQ_VAL_URL)
        unique_vals = json.loads(uv.json())

        # Get all features
        fs = requests.get(FEATURE_URL)
        features_set = fs.json()

        # combine all categorical, numerical, etc in one array
        combined_features = []
        for f_set in features_set.values():
            combined_features.extend(f_set)

        cat_f = features_set["cat"]
        num_f = features_set["num"]
        cyc_f = features_set["cyc"]
        bool_f = features_set["bool"]

        #  ******************** Prediction with csv or parquet ***********************
        st.subheader("Upload CSV or Parquet file with input data:")
        uploaded_file = st.file_uploader("", type=["csv", "parquet"])

        # check if file was uploaded
        if uploaded_file is not None:
            try:
                # load input data
                if uploaded_file.type == "parquet":
                    input_data = pd.read_parquet(uploaded_file)
                else:
                    input_data = pd.read_csv(uploaded_file)
                
                input_data = input_data[combined_features]
                input_data[cat_f] = input_data[cat_f].fillna("na")
                input_data[num_f] = input_data[num_f].fillna(0)
                input_data[cyc_f] = input_data[cyc_f].fillna(0)
                input_data[bool_f] = input_data[bool_f].fillna(0)
                # st.write(input_data)

                # make prediction using the API
                if st.button("Predict"):
                    response = requests.post(PREDICT_URL, json=input_data.to_dict(orient='records'))

                    if response.status_code == 200:
                        data = json.loads(response.json())
                        df = pd.DataFrame(data)
                        st.write(df)
                    else:
                        st.subheader("Error:")
                        st.write("There was an error with the API request.")

            except Exception as e:
                st.subheader("Error:")
                st.write(e)

        # ***************** Single sample prediction - with form ***********************
        st.header("Single sample prediction")

        single_sample = dict()
        with st.form(key='form'):
            #num_cols = features_set[0]  # numerical cols
            #cat_cols = features_set[1]  # categorical cols
            #ord_cols = features_set[2]  # ordinal cols

            for i, col in enumerate(unique_vals):
                col_name = cat_f[i]
                single_sample[col_name] = st.selectbox(col_name, unique_vals[col_name])


            for i in range(len(num_f)):
                col_name = num_f[i]
                val = str(st.number_input(col_name, value=0, step=1))
                single_sample[col_name] = int(val)

            slider_range = {
                "HOUR_APPR_PROCESS_START": [0, 23],
                "WEEKDAY_APPR_PROCESS_START": [1, 7]
            }

            for i in range(len(cyc_f)):
                col_name = cyc_f[i]
                val = str(st.slider(col_name, min_value=slider_range[col_name][0], max_value=slider_range[col_name][1]))
                single_sample[col_name] = int(val)

            for i in range(len(bool_f)):
                col_name = bool_f[i]
                val = str(st.number_input(col_name, value=0, step=1, max_value=1))
                single_sample[col_name] = int(val)

            submit_button = st.form_submit_button(label='Submit')

        ss_list = []
        ss_list.append(single_sample)
        df = pd.DataFrame(ss_list)

        if submit_button:
            input_data = df[combined_features]
            input_data[cat_f] = input_data[cat_f].fillna("na")
            input_data[num_f] = input_data[num_f].fillna(0)
            response = requests.post(PREDICT_URL, json=input_data.to_dict(orient='records'))

            if response.status_code == 200:
                data = json.loads(response.json())
                df = pd.DataFrame(data)
                st.write(df)
            
            else:
                st.subheader("Error:")
                st.write("There was an error with the API request.")



st.markdown('''
        <style>
            [data-testid=column] {
                border: 1px solid red;
                padding: 10px;
                width: 100%
            }
        </style>
    ''', unsafe_allow_html=True)