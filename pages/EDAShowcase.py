import numpy as np
import streamlit as st
import pandas as pd
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


try:
    data = st.session_state.DataCleaned
    target_col = st.session_state.TargetCol
except Exception as e:
    st.warning("No Data found, Please provide a dataset.")
    st.stop()
with st.sidebar:
    st.write("Your Data: ", data.head())
st.header("Exploratory Data Analysis")
st.write("This section shows you your data in Graphs.")


pieTitle = ""

df = st.session_state.data
if st.session_state.mode == 'Employee data':

    columnNames = df.columns.tolist()
    if "Employee" in columnNames:
        columnNames.remove("Employee")
    columnNames.remove('Monthly Income (£)')
    pieTitle = "Distribution of Money"
elif st.session_state.mode == 'Monthly Income':

    columnNames = df.columns.tolist()
    if "Month" in columnNames:
        columnNames.remove('Month')
    columnNames.remove('Monthly Income (£)')
    pieTitle = "Your distribution of Money over time"



# Calculation of char pie values
summedValues = {columnName: df[columnName].sum() for columnName in columnNames}
allValues = 0
pieExplode = []

for columnName in columnNames:
    allValues += summedValues[columnName]
    pieExplode.append(0.1)

pieValue = { columnValue: (summedValues[columnValue] / allValues * 100) for columnValue in columnNames}

# generating pie char

fig, distributionOfMoney  = plt.subplots(figsize=(10, 6))
distributionOfMoney.set_title(pieTitle)
distributionOfMoney.pie(pieValue.values(),labels=pieValue.keys() ,autopct='%1.1f%%', explode=pieExplode)
st.title('')
st.pyplot(fig)

if st.session_state.mode == 'Monthly Income':
    #Generating Line char
    st.subheader("Distribution of money over the whole time")
    st.line_chart(data, x='Month', y=columnNames, x_label=columnName)

# Heatmap generating for forcasting

st.title("Heatmap trend forecasting")
numeric_columns = data.select_dtypes(include=[np.number]).columns  # Identify numeric columns

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data[numeric_columns].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
st.pyplot(plt)
