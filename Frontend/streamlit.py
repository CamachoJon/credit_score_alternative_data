import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

from services import user as user_service
from services import report as report_service
import json
import requests
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
# from app import UNIQ_VAL_URL, PREDICT_URL
# from Frontend import UNIQ_VAL_URL, FEATURE_URL, PREDICT_URL, PAST_PREDICT_URL

st.set_page_config(layout="wide")

with st.sidebar:
    selected = option_menu(
        menu_title="Credit Scoring",
        options=["Home", "Model Analysis", "User Report", "Prediction"]
    )

with st.container():
    if selected == "Home":
        user_data = user_service.get_all_user_data()
        st.title("User Profiling Reports")
        odf = pd.DataFrame(user_data)
        df =  odf

        if 'prev_selected_option' not in st.session_state:
            st.session_state.prev_selected_option = 7  # Initialize with the value of "Past 7 days"

        options = {
            'Past 7 days': 7,
            'Past 15 days': 15,
            'Past 1 month': 30,
            'Past 6 months': 180
        }

        # cola1, cola2, cola3, cola4 = st.columns(4)

        # with cola1:
        #     st.write(f"Total: {len(df)}")
        
        col01, col02 = st.columns(2)
        with col01:
            selected_option = st.selectbox('Select Day Range', list(options.keys()))

            if st.session_state.prev_selected_option != options[selected_option]:
                df = odf
                st.session_state.prev_selected_option = options[selected_option]

                # Convert the 'Date' column to datetime objects
                odf['DATE'] = pd.to_datetime(odf['DATE'])

                # Get the date threshold based on the selected option
                date_threshold = pd.to_datetime('today') - pd.Timedelta(days=st.session_state.prev_selected_option)

                # Filter the DataFrame based on the date_threshold
                df = odf[odf['DATE'] >= date_threshold]

        col11, col12, col13, col14 = st.columns(4)
        with col11:
            gender_counts = df['CODE_GENDER'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=gender_counts.index, values=gender_counts.values)])
            fig.update_layout(title='Gender Distribution', title_x=0.3)
            st.plotly_chart(fig, use_container_width=True)

        with col12:
            fig = px.histogram(df, x="CNT_CHILDREN")
            fig.update_yaxes(title_text="Count")
            fig.update_xaxes(title_text="Number of Children")
            st.plotly_chart(fig, use_container_width=True)

        with col13:
            fig = px.histogram(df, x="NAME_FAMILY_STATUS")
            fig.update_xaxes(title_text="Family Status")
            fig.update_yaxes(title_text="Count")
            st.plotly_chart(fig, use_container_width=True)

        with col14:
            counts = df['NAME_INCOME_TYPE'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values)])
            fig.update_layout(title='Income Type Distribution', title_x=0.3)
            st.plotly_chart(fig, use_container_width=True)

        col21, col22, col23, col24 = st.columns(4)
        with col21:
            counts = df['NAME_HOUSING_TYPE'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values)])
            fig.update_layout(title='Housing Type Distribution', title_x=0.3)
            st.plotly_chart(fig, use_container_width=True)

        with col22:
            fig = px.histogram(df, x="NAME_EDUCATION_TYPE")
            fig.update_xaxes(title_text="Education Distribution")
            fig.update_yaxes(title_text="Count")
            st.plotly_chart(fig, use_container_width=True)

        with col23:
            counts = df['NAME_TYPE_SUITE'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values)])
            fig.update_layout(title='Suite Type Distribution', title_x=0.3)
            st.plotly_chart(fig, use_container_width=True)

        with col24:
            counts = df['ORGANIZATION_TYPE'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values)])
            fig.update_layout(title='Organization Type', title_x=0.3)
            st.plotly_chart(fig, use_container_width=True)

        col31, col32 = st.columns(2)
        with col31:
            # Calculate the frequency of each weekday
            weekday_counts = df["WEEKDAY_APPR_PROCESS_START"].value_counts().reset_index()
            weekday_counts.columns = ["WEEKDAY_APPR_PROCESS_START", "Count"]

            # Order weekdays in descending order
            weekday_counts = weekday_counts.sort_values(by="Count", ascending=True)
             # Plot the frequency distribution using Plotly bar chart
            fig = px.line(weekday_counts, x="WEEKDAY_APPR_PROCESS_START", y="Count",
                 labels={"WEEKDAY_APPR_PROCESS_START": "Weekday", "Count": "Count"},
                 title="Weekday Appraisal Start Frequency Distribution",
                 text="Count")  # Display count on top of the bars

            # Customize the layout of the chart
            fig.update_layout(
                xaxis=dict(title="Weekday"),
                showlegend=False,
                bargap=0.1,
                bargroupgap=0.2,
            )

            # Show the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)
        
        with col32:
            counts = df['OCCUPATION_TYPE'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values)])
            fig.update_layout(title='Occupation Type', title_x=0.3)
            st.plotly_chart(fig, use_container_width=True)

    if selected == "Model Analysis":
        st.title("Model Analysis")
        chart_placeholder = st.empty()
        with st.container():
            while True:
                user_data = user_service.get_all_user_data()
                df = pd.DataFrame(user_data)
                
                df['DATE'] = pd.to_datetime(df['DATE'])

                # Get the current date
                current_date = datetime.now()

                # Calculate the date 15 days ago from the current date
                past_date = current_date - timedelta(days=30)

                # Filter the DataFrame to include predictions made in the past 15 days
                filtered_df = df[(df['DATE'] >= past_date) & (df['DATE'] <= current_date)]

                # Group the data by date and count the number of predictions made on each day for TARGET=0
                grouped_data_0 = filtered_df[filtered_df['TARGET'] == 0].groupby(filtered_df['DATE'].dt.date).size().reset_index(name='count_0')

                # Group the data by date and count the number of predictions made on each day for TARGET=1
                grouped_data_1 = filtered_df[filtered_df['TARGET'] == 1].groupby(filtered_df['DATE'].dt.date).size().reset_index(name='count_1')

                # Create two Plotly line charts
                fig = go.Figure()

                # Line chart for TARGET=0
                fig.add_trace(go.Scatter(x=grouped_data_0['DATE'], y=grouped_data_0['count_0'],
                                        mode='lines+markers', line=dict(color='rgb(26, 118, 255)'), line_shape="spline",
                                        marker=dict(size=8, color='rgb(26, 118, 255)', line=dict(width=2)),
                                        name='TARGET=0',  # Legend label
                                        text=grouped_data_0['count_0'],  # Display the count as text on the line chart
                                        textposition="top center"))  # Position of the text label

                # Line chart for TARGET=1
                fig.add_trace(go.Scatter(x=grouped_data_1['DATE'], y=grouped_data_1['count_1'],
                                        mode='lines+markers', line=dict(color='rgb(255, 0, 0)'), line_shape="spline",
                                        marker=dict(size=8, color='rgb(255, 0, 0)', line=dict(width=2)),
                                        name='TARGET=1',  # Legend label
                                        text=grouped_data_1['count_1'],  # Display the count as text on the line chart
                                        textposition="top center"))  # Position of the text label

                # Customize the chart layout
                fig.update_layout(
                    title='Number of Predictions (TARGET=0 and TARGET=1) in the Past 15 Days',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Number of Predictions'),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                )

                # Show the plot using Streamlit
                chart_placeholder.plotly_chart(fig, use_container_width=True)


                # now = datetime.now()
                # twelve_hours_ago = now - timedelta(hours=12)
                # df[df['DATE'] >= twelve_hours_ago]


                # fig = px.line(df, x='DATE', y='TARGET', title='Counts of 1s and 0s in the Last 12 Hours',
                #   labels={'TARGET': 'Count', 'DATE': 'Time'})

                # # Set the x-axis interval to 1 hour
                # fig.update_xaxes(
                #     dtick='HOUR',
                #     tickformat='%H:%M:%S',  # Display hours, minutes, and seconds on the x-axis
                #     ticklabelmode='period',
                #     ticklabelposition='inside'
                # )

                # if not df["DATE"].empty:
                #     # Plot the line chart using Plotly
                #     st.plotly_chart(fig, use_container_width=True)
                # else:
                #     st.warning("No data available in the last 12 hours.")


                time.sleep(3)      

    if selected == "User Report":
        st.title(f"User Report")
        colr1, colr2 = st.columns(2)

        with colr1:
            firstname = st.text_input("Customer's First Name:")

        with colr2:
            lastname = st.text_input("Customer's Last Name:")

        

        if st.button("Get User Information"):
            if firstname and lastname:
                user_info_str = user_service.get_user_data_by_name(firstname, lastname)
                user_info = json.loads(user_info_str)

                # st.write(user_info_str)
                
                gender = 'Male' if user_info[0]['CODE_GENDER'] == 'M' else 'Female'
                st.write(f"Gender : {gender}")
                st.write(f"Marital Status : {user_info[0]['NAME_FAMILY_STATUS']}")
                st.write(f"Income : {user_info[0]['AMT_INCOME_TOTAL']}")
                st.write(f"Education : {user_info[0]['NAME_EDUCATION_TYPE']}")
                st.write(f"Housing : {user_info[0]['NAME_HOUSING_TYPE']}")
                st.write(f"Occupation : {user_info[0]['OCCUPATION_TYPE']}")
                
                owns_car = 'Yes' if user_info[0]['FLAG_OWN_CAR'] == 'Y' else 'No'
                st.write(f"Owns Car : {owns_car}")

                response = requests.get(f"http://backend:8000/generate_report?name={firstname}&lastname={lastname}")
                st.download_button(label="Download PDF Report 📑", data=response.content, file_name=f"{firstname}_{lastname}_report.pdf", mime="application/pdf")

            else:
                st.warning("Both First Name & Last Name of the Customer are required to search data.")
                  
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
                border: 1px solid transparent;
                padding: 10px;
                width: 100%
            }
        </style>
    ''', unsafe_allow_html=True)