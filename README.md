# 🩺 Clasificación de Cáncer de Mama (UCI WDBC)

Este proyecto implementa una aplicación web interactiva para la **clasificación de cáncer de mama** utilizando el dataset **Breast Cancer Wisconsin (Diagnostic)** de la UCI Machine Learning Repository.  
La aplicación compara dos enfoques de aprendizaje automático:

- **Regresión Logística**: modelo estadístico que estima la probabilidad de malignidad.  
- **Red Neuronal Multicapa (MLP)**: arquitectura con capas ocultas que captura relaciones no lineales más complejas.  

La interfaz está desarrollada con **Streamlit**, lo que permite entrenar modelos, visualizar métricas y mostrar la matriz de confusión de manera sencilla y dinámica.

---

## 📊 Dataset

- **Fuente**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  
- **Instancias**: 569 muestras  
- **Atributos**: 30 características numéricas (radio, textura, perímetro, área, suavidad, compacidad, concavidad, simetría, dimensión fractal, etc.)  
- **Variable objetivo**: `diagnosis` → Benigno (0) o Maligno (1)  

---

