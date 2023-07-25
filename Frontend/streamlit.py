import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import time
import pandas as pd
import json

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

        # Convert JSON string to Python dictionary
        data_dict = json.loads(user_data)

        # Create a DataFrame from the dictionary
        df = pd.DataFrame([data_dict])

        st.dataframe(df.head(10))
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

st.markdown('''
        <style>
            [data-testid=column] {
                border: 1px solid red;
                padding: 10px;
                width: 100%
            }
        </style>
    ''', unsafe_allow_html=True)