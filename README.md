# Predicción de la Calidad del Aire (NO₂) mediante Machine Learning

> **Proyecto académico de Inteligencia Artificial**  
> Implementación de modelos predictivos para estimar la concentración de dióxido de nitrógeno (NO₂) en el aire urbano, utilizando técnicas de Machine Learning y análisis de datos ambientales.

---

## Autores:

- Juan Jose Vasquez Gomez
- Santiago Alvarez Peña

---

## Recursos Importantes:

- Articulo: [Predicción de calidad del aire en entornos urbanos](https://drive.google.com/file/d/1d9T75G_UDFeVqf1j4EwvcN8HN86xjFMr/view?usp=sharing)
- Video demostrativo: [Ejecución y explicación de código](https://drive.google.com/file/d/1G5I-Azd-9_8uCmUdscjrNNf8Fx0yZZ-N/view?usp=sharing)

---

El objetivo de este proyecto es desarrollar un modelo capaz de **predecir la concentración de NO₂ (µg/m³)** a partir de variables químicas y meteorológicas, como temperatura, humedad, presión atmosférica y concentraciones de otros gases contaminantes.

Este trabajo combina **análisis exploratorio, preprocesamiento, modelado predictivo** y **evaluación de desempeño**, buscando aportar herramientas para el monitoreo ambiental inteligente.

---

## Descripción técnica

El modelo se entrena con el **Air Quality Dataset (UCI / Kaggle)**, que contiene mediciones horarias de gases contaminantes recolectados por sensores en entornos urbanos.

Los algoritmos utilizados son:
- **Random Forest Regressor**
- **XGBoost Regressor**
  
Ambos permiten capturar relaciones no lineales entre variables meteorológicas y contaminantes, optimizando la predicción de concentraciones futuras de NO₂.

---

## Estructura del repositorio

/Entrega4-IA/  
│  
├── data/  
│ ├── AirQualityUCI.csv # Dataset original  
│ └── processed/  
│ └── air_quality_split.npz # Datos limpios y divididos  
│  
├── media/ # Imagenes y gráficos por cada fase  
│ ├── eda/  
│ ├── explainability/  
│ └── training/  
│  
├── src/  
│ ├── eda_preprocessing.py # Limpieza y análisis exploratorio  
│ ├── train_models.py # Entrenamiento y evaluación  
│ └── explainability.py # Análisis de resultados  
│  
├── requirements.txt # Dependencias del proyecto  
└── README.md # Documentación principal  

---

## Instalación y ejecución

### 1 Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/Proyecto-CalidadAire.git
cd Proyecto-CalidadAire
```
### 2. Crear entorno virtual (Opcional)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```
### 3. Instalar dependencias requeridas

`pip install -r requirements.txt`

### 4. Ejecutar los módulos principales (En orden)
- Preprocesamiento y limpieza
`python src/eda_preprocessing.py`

- Entrenamiento de modelos
`python src/train_models.py`

- Evaluación e interpretación
`python src/explainability.py`

## Análisis de resultados

| Modelo            | MAE (µg/m³) | RMSE (µg/m³) |   R² |
| :---------------- | ----------: | -----------: | ---: |
| **Random Forest** |      29.939 |       38.023 | 0.45 |
| **XGBoost**       |      29.225 |       38.302 | 0.43 |
  
El modelo XGBoost logró una mejor precisión global y menor error de predicción, mostrando mayor capacidad de generalización.

### Importancia de las variables

- Las variables que mostraron mayor influencia sobre el NO₂ fueron:

- PT08.S4(NO2): Sensor químico de NO₂.

- C6H6(GT): Concentración de benceno, correlacionado con emisiones de tráfico.

- T: Temperatura ambiente, que afecta la dispersión de contaminantes.
  
Esto concuerda con la literatura sobre relaciones entre temperatura, tráfico y concentración de gases urbanos.

### Comparación de predicciones

El modelo XGBoost sigue más fielmente el comportamiento real del NO₂, especialmente en picos de concentración, mientras que Random Forest presenta mayor suavizado.

### Distribución de errores

Ambos modelos presentan errores centrados alrededor de cero, sin sesgo evidente, lo cual indica un ajuste adecuado sin sobreentrenamiento significativo.

## Dependencias principales

- Python 3.11-3.14

| Librería       | Uso principal                  |
| -------------- | ------------------------------ |
| `pandas`       | Manipulación de datos          |
| `numpy`        | Cálculos numéricos             |
| `matplotlib`   | Visualización                  |
| `seaborn`      | Gráficos estadísticos          |
| `scikit-learn` | Modelos y métricas             |
| `xgboost`      | Modelo predictivo avanzado     |

## Posibles mejoras futuras

- Implementar una red LSTM para predicciones temporales secuenciales.

- Integrar datos climáticos externos (viento, lluvia) para mejorar la precisión.

- Desplegar un dashboard interactivo con Streamlit o Dash.

- Aplicar técnicas de Explainable AI (SHAP / LIME) para mayor interpretabilidad.
