from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

st.header("Clustering:")

st.write("Clustering is a technic to define groups which will be defined by the value 'k'. You can adjust this value"
         "by your liking, or generating a 'K Elbow' graph which show you the best K value to insert."
         "By selecting the column and providing a K value the system  generate a K cluster graph by pressing 'Generate K Cluster'"
         "By pressing 'Generate K elbow' the system shows a graph for the best K value.")

try:
    data_cleaned = st.session_state.DataCleaned
except Exception as e:
    st.warning("No Data found, Please provide a dataset.")
    st.stop()
with st.sidebar:
    st.write("Your Data: ", data_cleaned.head())

if st.session_state.mode != "Monthly Income":
    st.warning("This section is only for 'Monthly Income' data")
    st.stop()


all_columns = data_cleaned.columns.tolist()

selectedColumn1 = st.selectbox("Select first column to compare with", options= data_cleaned.columns)
selectedColumn2 = st.selectbox("Select second column to compare with", options = data_cleaned.columns)
X = data_cleaned[[selectedColumn1, selectedColumn2]]

def GenerateKCluster():
    try:
        kmeans = KMeans(n_clusters=kInput)
        kmeans.fit(X)
        labels = kmeans.predict(X)
        X['Cluster'] = labels

        plt.scatter(X[selectedColumn1], X[selectedColumn2], c=X['Cluster'], cmap='rainbow')
        plt.title('Clusters Visualization')
        plt.xlabel(selectedColumn1)
        plt.ylabel(selectedColumn2)
        st.pyplot(plt)
    except Exception as e:
        st.error("K cluster generating error: ",e)




def GenerateKElbow():
    try:
        if selectedColumn1 == selectedColumn2 or selectedColumn1 == "" or selectedColumn2 == "":
            st.warning("Please provide two columns to compare with")
        else:


            inertias = []

            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)

            plt.plot(range(1, 11), inertias, marker='o')
            plt.title('Elbow method')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')
            st.pyplot(plt)



    except Exception as e:
        st.write("Error Elbow method: ", e)
kInput = int(st.number_input("K input"))

st.button("Generate K cluster", on_click=GenerateKCluster)


# Specify the number of clusters

#def KClustering():

st.button("Generate K elbow", on_click=GenerateKElbow)




def showDistribution():
    # Distribution of the target variable
    plt.figure(figsize=(8, 6))
    sns.histplot(data_cleaned[columnToShow], kde=True, bins=30, color='skyblue')
    plt.title(f"Distribution of {columnToShow}")
    plt.xlabel(columnToShow)
    plt.ylabel("Frequency")
    st.pyplot(plt)


columnToShow = st.selectbox("Select a target Column",data_cleaned.columns)
st.button("Show Distribution", on_click=showDistribution)
