import streamlit as st
import pandas as pd

try:
    data_cleaned = st.session_state.DataCleaned
except Exception as e:
    st.warning("No Data found, Please provide a dataset.")
    st.stop()
with st.sidebar:
    st.write("Your Data: ", data_cleaned.head())



st.title("A quick overview of your Data.")

st.title("Your Data:")
st.write(st.session_state.data)



