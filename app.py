# app.py
# Clasificación de cáncer de mama con Regresión Logística y Red Neuronal (MLP)
# Adaptado a Streamlit para interfaz web

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# Cargar dataset desde UCI
# -----------------------------
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    columns = [
        "id","diagnosis",
        "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
        "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
        "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"
    ]
    df = pd.read_csv(url, header=None, names=columns)
    df["label"] = df["diagnosis"].map({"M":1,"B":0})
    X = df.iloc[:,2:32].values
    y = df["label"].values
    return X, y

X, y = load_data()

# -----------------------------
# Funciones de entrenamiento
# -----------------------------
def train_logistic(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def train_mlp(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu", solver="adam",
                          max_iter=300, random_state=42, early_stopping=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    err = 1 - acc
    cm = confusion_matrix(y_true, y_pred)
    return {"Accuracy":acc,"Precision":prec,"Recall":rec,"F1":f1,"Error":err,"ConfusionMatrix":cm}

# -----------------------------
# Interfaz Streamlit
# -----------------------------
st.title("🩺 Clasificación de Cáncer de Mama (UCI WDBC)")
st.write("Aplicación con **Regresión Logística** y **Red Neuronal (MLP)** para diagnóstico benigno/maligno.")

# Selección de modelo
model_choice = st.radio("Selecciona el modelo:", ["Regresión Logística","Red Neuronal (MLP)"])

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

if st.button("Entrenar y Evaluar"):
    if model_choice == "Regresión Logística":
        y_pred = train_logistic(X_train, y_train, X_test, y_test)
        title = "Matriz de Confusión - Regresión Logística"
    else:
        y_pred = train_mlp(X_train, y_train, X_test, y_test)
        title = "Matriz de Confusión - Red Neuronal (MLP)"

    results = evaluate(y_test, y_pred)

    st.subheader("📊 Métricas del modelo")
    st.write(f"- Error: {results['Error']:.4f}")
    st.write(f"- Exactitud (Accuracy): {results['Accuracy']:.4f}")
    st.write(f"- Precisión (Precision): {results['Precision']:.4f}")
    st.write(f"- Exhaustividad (Recall): {results['Recall']:.4f}")
    st.write(f"- F1-Score: {results['F1']:.4f}")

    st.subheader(title)
    fig, ax = plt.subplots()
    sns.heatmap(results["ConfusionMatrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benigno (0)","Maligno (1)"],
                yticklabels=["Benigno (0)","Maligno (1)"], ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    st.pyplot(fig)

st.caption("Dataset: Breast Cancer Wisconsin (Diagnostic) - 569 instancias, 30 atributos numéricos.")
