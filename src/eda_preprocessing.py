# Autores: Juan Jose Vasquez, Santiago Alvarez

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. CARGA DEL DATASET LOCAL

DATA_PATH = os.path.join("../data", "AirQualityUCI.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("El archivo 'AirQualityUCI.csv' no se está en la carpeta /data/")

# Cargar el archivo CSV con configuración correcta
df = pd.read_csv(DATA_PATH, sep=',', decimal='.', low_memory=False)

print(f"✅ Dataset cargado correctamente con {df.shape[0]} filas y {df.shape[1]} columnas.\n")

# Limpiar nombres de columnas
df.columns = df.columns.str.strip()
print("📋 Columnas disponibles:", list(df.columns), "\n")

# 2. LIMPIEZA DE DATOS

# Eliminar columnas completamente vacías
df.dropna(axis=1, how='all', inplace=True)

# Reemplazar valores faltantes (-200) con NaN
df.replace(-200, np.nan, inplace=True)

# Eliminar filas con más del 30% de valores nulos
df.dropna(thresh=df.shape[1]*0.7, inplace=True)

# Convertir columnas numéricas correctamente antes de interpolar
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# Identificar columnas numéricas
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Interpolación solo en las columnas numéricas
df[num_cols] = df[num_cols].interpolate(method='linear')

print(f"Datos limpips. Nuevas dimensiones: {df.shape}\n")

# 3. ANÁLISIS EXPLORATORIO BÁSICO (EDA)

target = 'NO2(GT)'

# Estadísticas descriptivas
print("Resumen estadístico de la variable objetivo:")
print(df[target].describe(), "\n")

# Histograma del contaminante NO₂
plt.figure(figsize=(8, 5))
sns.histplot(df[target], bins=40, kde=True, color="steelblue")
plt.title("Distribución de NO₂ (µg/m³)")
plt.xlabel("Concentración de NO₂ (µg/m³)")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

# Matriz de correlación general
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("Matriz de correlación de variables numéricas")
plt.tight_layout()
plt.show()

# 4. SELECCIÓN DE VARIABLES Y NORMALIZACIÓN

features = [
    "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "CO(GT)", "C6H6(GT)",
    "T", "RH", "AH"
]

# Variables predictoras y objetivo
X = df[features]
y = df[target]

# Normalizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. DIVISIÓN TEMPORAL DE DATOS (70/15/15)

train_size = int(len(X_scaled) * 0.7)
val_size = int(len(X_scaled) * 0.85)

X_train, X_val, X_test = X_scaled[:train_size], X_scaled[train_size:val_size], X_scaled[val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:val_size], y[val_size:]

print(f"División de datos completada:")
print(f"   - Entrenamiento: {len(X_train)} registros")
print(f"   - Validación:    {len(X_val)} registros")
print(f"   - Prueba:        {len(X_test)} registros\n")

# 6. GUARDAR DATOS PROCESADOS (opcional)

os.makedirs("../data/processed", exist_ok=True)
np.savez("../data/processed/air_quality_split.npz",
         X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test)

print("Archivos procesados guardados en /data/processed/air_quality_split.npz")
