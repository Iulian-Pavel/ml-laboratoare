import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
data = pd.read_csv("bank-additional/bank-additional-full.csv", sep=';')

label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

y = data['y']
X = data.drop(columns=['y'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Regresie Logistică": LogisticRegression(),
    "Arbore Decizional": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} - Acuratețe: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("-" * 50)

st.title("Clasificare Machine Learning - Bank Marketing")
st.write("Selectați un model pentru a vedea performanța acestuia.")
model_choice = st.selectbox("Alegeți un model:", list(models.keys()))

if st.button("Rulează modelul"):
    model = models[model_choice]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"Acuratețea modelului {model_choice}: {accuracy_score(y_test, y_pred):.4f}")
    st.write("Raport de clasificare:")
    st.text(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    st.pyplot(plt)
