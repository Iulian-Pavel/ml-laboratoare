import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
df = pd.read_csv(url, sep=';')

df = df.select_dtypes(include=[np.number]).dropna()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

kmeans = KMeans(n_clusters=3, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(df_scaled)

dbscan = DBSCAN(eps=0.5, min_samples=5)
df['dbscan_cluster'] = dbscan.fit_predict(df_scaled)

silhouette_kmeans = silhouette_score(df_scaled, df['kmeans_cluster'])

def plot_clusters():
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df['kmeans_cluster'], palette='viridis', ax=ax[0])
    ax[0].set_title("K-Means Clustering")

    sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df['dbscan_cluster'], palette='coolwarm', ax=ax[1])
    ax[1].set_title("DBSCAN Clustering")

    plt.show()

st.title("Clustering & PCA Visualization")
st.write(f"Scorul Silhouette pentru K-Means: {silhouette_kmeans:.2f}")

st.pyplot(plot_clusters())
