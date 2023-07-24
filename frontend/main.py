import streamlit as st
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        menu_title="Credit Scoring",
        options=["Home", "Reports"]
    )


if selected == "Home":
    st.title(f"You have selected: {selected}")
if selected == "Reports":
    st.title(f"You have selected: {selected}")