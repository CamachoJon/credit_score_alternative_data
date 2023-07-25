import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from services import user as user_service

st.set_page_config(layout="wide")

with st.sidebar:
    selected = option_menu(
        menu_title="Credit Scoring",
        options=["Home", "User Input", "Reports"]
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
            weekday_counts = weekday_counts.sort_values(by="Count", ascending=False)
             # Plot the frequency distribution using Plotly bar chart
            fig = px.bar(weekday_counts, x="WEEKDAY_APPR_PROCESS_START", y="Count",
                 labels={"WEEKDAY_APPR_PROCESS_START": "Weekday", "Count": "Count"},
                 title="Weekday Appraisal Start Frequency Distribution",
                 text="Count")  # Display count on top of the bars

            # Customize the layout of the chart
            fig.update_layout(
                xaxis=dict(title="Weekday"),
                yaxis=dict(title="Frequency"),
                showlegend=False,
                bargap=0.1,
                bargroupgap=0.2,
            )

            # Show the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)
        
        # with col32:
        #     df["OWN_CAR_AGE"] = pd.to_numeric(df["OWN_CAR_AGE"], errors="coerce")
        #     df["OWN_CAR_AGE"].fillna(-1, inplace=True)

        #     # Create 3-year intervals for the car age
        #     interval_size = 3
        #     intervals = pd.IntervalIndex.from_tuples([(i, i + interval_size) for i in range(0, int(df["OWN_CAR_AGE"].max()), interval_size)])
        #     df["Age_Interval"] = pd.cut(df["OWN_CAR_AGE"], bins=intervals)

        #     # Calculate the count of customers in each interval
        #     interval_counts = df["Age_Interval"].value_counts().reset_index()
        #     interval_counts.columns = ["Age_Interval", "Count"]

        #     # Sort intervals by their start value
        #     interval_counts = interval_counts.sort_values(by="Age_Interval")

        #     # Plot the frequency distribution using Plotly bar chart
        #     fig = px.bar(interval_counts, x="Age_Interval", y="Count",
        #                 labels={"Age_Interval": "Car Age Interval", "Count": "Count"},
        #                 title="Customer Car Age Distribution",
        #                 text="Count")  # Display count on top of the bars

        #     # Customize the layout of the chart
        #     fig.update_layout(
        #         xaxis=dict(title="Car Age Interval"),
        #         yaxis=dict(title="Count"),
        #         showlegend=False,
        #         bargap=0.1,
        #         bargroupgap=0.2,
        #     )

        #     # Show the chart in Streamlit
        #     st.plotly_chart(fig, use_container_width=True)
        

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

st.markdown('''
        <style>
            [data-testid=column] {
                border: 1px solid transparent;
                padding: 10px;
                width: 100%
            }
        </style>
    ''', unsafe_allow_html=True)