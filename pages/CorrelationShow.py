import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

try:
    mode = st.session_state.mode
    data = st.session_state.DataCleaned
except Exception as e:
    st.warning("No Data found, Please provide a dataset.")
    st.stop()
with st.sidebar:
    st.write("Your Data: ", data.head())

st.header("Correlation of Dataset.")
st.write("This section can show you how your Money is spent via Bar chars and line Chars")

if mode != "Monthly Income":
    data = st.session_state.data
    st.write("You can see how your spending is correlating between your Employee and the spending of money.")
    def showBarChar():
        if selectedOptions != [] and selectedEmployee != []:
            barCharValues = {employee:
                                 {option:
                                      data[data['Employee'] == employee][option] for option in selectedOptions} for
                             employee
                             in selectedEmployee}
            barCharArray = []
            for employee in selectedEmployee:
                for option in barCharValues[employee]:
                    barCharArray.append(barCharValues[employee][option])

            st.bar_chart(barCharArray, x_label=selectedOptions, stack=False)
        else:
            st.warning("Please provide Employee data and Select an Option")

    selectedEmployee = st.multiselect("Select an Employee",options=data['Employee'])
    selectedOptions = st.multiselect("Select an Options",options=data.columns)
    st.button("Generate Barchar",on_click=showBarChar)
else:
    st.write("You can see how your spending is correlating between your expenditure and income.")

    tab1, tab2 = st.tabs(["Bar Chart","LineChart"])
    selectedOptions = st.multiselect("Select an Options",options=data.columns)

    with tab1:
        st.bar_chart(data, x='Month', y=selectedOptions, x_label=selectedOptions, stack=False)
    with tab2:
        st.line_chart(data,x='Month',y = selectedOptions, x_label=selectedOptions)



