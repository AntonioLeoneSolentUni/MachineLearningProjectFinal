import streamlit as st
import pandas as pd
import numpy as np
from requests import session
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go


uploadChecker = False
# Step 0: File Upload and Basic Validation
st.title("Personal Finance Management System with Machine Learning and Interactive Features")
st.write("---")

st.write("Welcome to your finance management system!")
st.write("This system has 2 modes:")
st.subheader("Mode 1: Monthly Income")
st.write("This Works with your data and gives you advice with the help of Machine Learning models. ")
st.write("It can help you planing future expenses and having an overview where you money was spent in the Months your have provided!")
st.write("In order to make this system work you need to provide the data in a specific order.")
st.subheader("Order:")
st.code("Month,Monthly Income (£),[Every spending of money],Savings for Property (£),Monthly Outing (£)")
st.write("Make sure you have following columns correct (including writing and spelling)")
st.code("Month,Monthly Income (£),Savings for Property (£),Monthly Outing (£)")
st.write("Here an example of a data entry")
st.code("Month,Monthly Income (£),Electricity Bill (£),Gas Bill (£),Netflix (£),Amazon Prime (£),Groceries (£),Transportation (£),Water Bill (£),Sky Sports (£),Other Expenses (£),Savings for Property (£),Monthly Outing (£)")
st.code("1,4999.39,120.0,80.0,12,4.99,239.69,149.11,40.0,70.0,100.0,300.0,120.0")
st.subheader("Mode 2: Employee Data")
st.write("This Works with your Employee data and gives you advice with the help of Machine Learning models. ")
st.write("It can help you planing future expenses and having an overview where you money was spent in the Months your have provided!")
st.write("In order to make this system work you need to provide the data in a specific order.")
st.subheader("Order:")
st.code("Employee,Monthly Income (£),[Every spending of money] ,Savings for Property (£),Monthly Outing (£)")
st.write("Make sure you have following columns correct (including writing and spelling)")
st.code("Employee,Monthly Income (£),Savings for Property (£),Monthly Outing (£)")
st.write("Here an example of a data entry")
st.code("Month,Monthly Income (£),Electricity Bill (£),Gas Bill (£),Netflix (£),Amazon Prime (£),Groceries (£),Transportation (£),Water Bill (£),Sky Sports (£),Other Expenses (£),Savings for Property (£),Monthly Outing (£)")
st.code("Employee_1,4999.39,120.0,80.0,12,4.99,239.69,149.11,40.0,70.0,100.0,300.0,120.0")


st.subheader("Please select what usecase your data has.")
mode = st.radio("Is it Monthly Income or Employee data?", ("Monthly Income","Employee data"))

if mode not in st.session_state:
    st.session_state.mode = mode



if uploadChecker not in st.session_state:
    uploadFile = st.file_uploader("Upload your CSV file:", type=["csv"])
    st.session_state.uploadChecker = False

    if uploadFile is not None:
        selectedData = pd.read_csv(uploadFile)
        data = selectedData
        uncleanedData =  selectedData
        st.session_state.uncleanedData = uncleanedData
        st.session_state.uploadChecker = True




if st.session_state.uploadChecker is True:
    try:
        # Read uploaded CSV file

        if data not in st.session_state:
            st.session_state.data = data
        st.write("Data Preview:", data.head())



    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# Step 1: Data Cleaning and Preprocessing
st.subheader("Step 1: Data Cleaning and Preprocessing")

# Define target column
target_col = 'Savings for Property (£)'
requiredCol = ["Monthly Income (£)","Savings for Property (£)","Monthly Outing (£)"]
if target_col not in st.session_state:
    st.session_state.TargetCol = target_col

try:
         # Drop non-numeric columns
    non_numeric_columns = st.session_state.data.select_dtypes(exclude=["number"]).columns
    if non_numeric_columns.size > 0:
        st.write("Dropping non-numeric columns:", non_numeric_columns.tolist())
        data = data.drop(columns=non_numeric_columns)

        missing_cols = [col for col in requiredCol if col not in data.columns]

        if missing_cols:
            st.error(f"Required columns {', '.join(missing_cols)} are missing in the dataset.")
            st.stop()
    # Drop rows with missing target values and fill remaining NaNs with column means
    data_cleaned = data.dropna(subset=[target_col])
    data_cleaned.fillna(data_cleaned.mean(), inplace=True)

    if data_cleaned not in st.session_state:
        st.session_state.DataCleaned = data_cleaned



        st.switch_page('pages/EDAShowcase.py')

except Exception as e:
    st.error(f"Data cleaning error: {e}")
    st.stop()
