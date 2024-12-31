from email.policy import default
from unittest.mock import inplace

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
         "and experiment with. The outcome how many groups are in relation to your data and Monthly Outing")

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
exclude_column = 'Monthly Outing (£)'
selected_columns = [col for col in all_columns if col != exclude_column]



X = data_cleaned[selected_columns]
y = data_cleaned['Monthly Outing (£)']

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Specify the number of clusters

def KClustering():
    # Create a KMeans instance
    kmeans = KMeans(n_clusters=kVal, random_state=42)

    # Fit the model to the data
    kmeans.fit(X_scaled)

    # Get the cluster labels
    labels = kmeans.labels_

    # Get the coordinates of the cluster centers
    centers = kmeans.cluster_centers_

    from sklearn.decomposition import PCA

    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plotting the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=30, cmap='viridis', label='Data points')
    centers_pca = pca.transform(centers)  # Transform the centers to PCA space
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.75, marker='X',
                label='Centroids')  # Cluster centers
    plt.title('K-means Clustering with Multiple Features')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()
    st.pyplot(plt)
    st.write(kVal)

kVal = st.number_input(label="Grouping Input", placeholder = 1) # You can adjust this based on your analysis
kVal = int(kVal)
st.write(kVal)
st.button("Generate KClustering.", on_click=KClustering)



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
