import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Încărcarea setului de date
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
data = pd.read_csv("day.csv")

# Vizualizarea datelor
st.write("## Vizualizare date")
st.write(data.head())

# Vizualizarea statisticilor descriptive
st.write("## Statistici descriptive")
st.write(data.describe())

# Tratarea valorilor lipsă
st.write("## Verificarea valorilor lipsă")
st.write(data.isnull().sum())

# Eliminarea outlierilor
sns.boxplot(data['cnt'])
outlier_threshold = data['cnt'].quantile(0.99)
data = data[data['cnt'] <= outlier_threshold]

# Selectarea caracteristicilor relevante
features = ['temp', 'atemp', 'hum', 'windspeed']
target = 'cnt'
X = data[features]
y = data[target]

# Împărțirea datelor în seturi de antrenament și test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizarea datelor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelul de regresie liniară
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)

# Modelul k-NN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Evaluarea modelelor
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    st.write(f"### {model_name}")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"R2 Score: {r2:.2f}")
    
evaluate_model(y_test, y_pred_lin, "Regresie Liniară")
evaluate_model(y_test, y_pred_knn, "K-Nearest Neighbors")

# Crearea interfeței cu Streamlit
st.title("Vizualizare rezultate modele de regresie")

model_option = st.selectbox("Selectează modelul", ["Regresie Liniară", "k-NN"])

if model_option == "Regresie Liniară":
    predictions = y_pred_lin
else:
    predictions = y_pred_knn

fig, ax = plt.subplots()
ax.scatter(y_test, predictions)
ax.set_xlabel("Valori reale")
ax.set_ylabel("Predicții")
ax.set_title(f"{model_option} - Predicții vs. Valori reale")
st.pyplot(fig)
