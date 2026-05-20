# 🤖 Breast Cancer Classifier — Clasificación de Cáncer de Mama

> **ES** | [EN below](#en---breast-cancer-classifier)

Aplicación web interactiva que compara dos modelos de Machine Learning para clasificar tumores de mama como benignos o malignos, usando el dataset UCI Breast Cancer Wisconsin (Diagnostic).

## ¿Qué hace?

- Entrena y compara **Regresión Logística** vs **Red Neuronal MLP**
- Muestra métricas de evaluación: accuracy, precision, recall, F1
- Visualiza la **matriz de confusión** de forma dinámica
- Interfaz web construida con **Streamlit** — sin necesidad de código para usarla

## Tecnologías

| Herramienta | Uso |
|---|---|
| Python 3 | Lenguaje principal |
| Streamlit | Interfaz web interactiva |
| scikit-learn | Modelos ML y métricas |
| pandas / numpy | Procesamiento de datos |

## Dataset

- **Fuente:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **569 muestras** con 30 características numéricas
- **Variable objetivo:** Benigno (0) o Maligno (1)

## Cómo ejecutarlo

```bash
# 1. Clona el repositorio
git clone https://github.com/JakelinB/Taller_MachineLearning.git
cd Taller_MachineLearning

# 2. Instala las dependencias
pip install -r requirements.txt

# 3. Lanza la app
streamlit run app.py
```

## Lo que aprendí

- Preprocesamiento de datos reales con valores atípicos
- Diferencias prácticas entre modelos lineales y redes neuronales
- Cómo exponer un modelo ML como aplicación web con Streamlit

---

## EN — Breast Cancer Classifier

Interactive web app that compares two ML models (Logistic Regression vs MLP Neural Network) for classifying breast tumors as benign or malignant, using the UCI WDBC dataset.

### Stack
`Python` · `Streamlit` · `scikit-learn` · `pandas` · `numpy`

### Run it

```bash
git clone https://github.com/JakelinB/Taller_MachineLearning.git
cd Taller_MachineLearning
pip install -r requirements.txt
streamlit run app.py
```

---
*Desarrollado por [Jakelin Bedoya](https://www.linkedin.com/in/jakelin-bedoya-becerra) · Medellín, Colombia*
