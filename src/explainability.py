# Autores: Juan Jose Vasquez, Santiago Alvarez

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. CARGA DE DATOS Y MODELOS

DATA_PATH = os.path.join("../data", "processed", "air_quality_split.npz")
MODELS_PATH = "models"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("No se encontró el archivo procesado. Ejecuta 'eda_preprocessing.py' primero.")

data = np.load(DATA_PATH)
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

rf_model = joblib.load(os.path.join(MODELS_PATH, "random_forest_model.pkl"))
xgb_model = joblib.load(os.path.join(MODELS_PATH, "xgboost_model.pkl"))

feature_names = [
    "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "CO(GT)", "C6H6(GT)",
    "T", "RH", "AH"
]

print("Modelos y datos cargados correctamente.\n")

# 2. PREDICCIÓN Y EVALUACIÓN

def evaluar_modelo(nombre, modelo, X, y):
    """Evalúa el modelo y devuelve métricas + predicciones."""
    preds = modelo.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    print(f"{nombre}")
    print(f"   MAE  = {mae:.3f}")
    print(f"   RMSE = {rmse:.3f}")
    print(f"   R²   = {r2:.3f}\n")
    return preds, mae, rmse, r2

rf_preds, _, _, _ = evaluar_modelo("Random Forest", rf_model, X_test, y_test)
xgb_preds, _, _, _ = evaluar_modelo("XGBoost", xgb_model, X_test, y_test)

# 3. IMPORTANCIA DE VARIABLES

# --- Random Forest ---
rf_importances = rf_model.feature_importances_
# --- XGBoost ---
xgb_importances = xgb_model.feature_importances_

# Plot comparativo
plt.figure(figsize=(10, 6))
x = np.arange(len(feature_names))
width = 0.35
plt.bar(x - width/2, rf_importances, width, label="Random Forest")
plt.bar(x + width/2, xgb_importances, width, label="XGBoost")
plt.xticks(x, feature_names, rotation=45, ha='right')
plt.title("Comparación de importancia de variables")
plt.ylabel("Importancia relativa")
plt.legend()
plt.tight_layout()
plt.show()

# 4. PREDICCIÓN VS REAL (COMPARACIÓN VISUAL)

plt.figure(figsize=(10, 5))
plt.plot(y_test[:200].tolist(), label="Real", color="black", linewidth=1.5)
plt.plot(rf_preds[:200].tolist(), label="Random Forest", color="blue", alpha=0.7)
plt.plot(xgb_preds[:200].tolist(), label="XGBoost", color="red", alpha=0.7)
plt.title("Comparación de predicciones (primeras 200 muestras)")
plt.xlabel("Índice de muestra")
plt.ylabel("Concentración NO₂ (µg/m³)")
plt.legend()
plt.tight_layout()
plt.show()

# 5. DISTRIBUCIÓN DE ERRORES

rf_errors = y_test - rf_preds
xgb_errors = y_test - xgb_preds

plt.figure(figsize=(8, 5))
sns.histplot(rf_errors, color="blue", label="Random Forest", kde=True, alpha=0.5)
sns.histplot(xgb_errors, color="red", label="XGBoost", kde=True, alpha=0.5)
plt.title("Distribución de errores (residuales)")
plt.xlabel("Error (y_real - y_pred)")
plt.ylabel("Frecuencia")
plt.legend()
plt.tight_layout()
plt.show()

