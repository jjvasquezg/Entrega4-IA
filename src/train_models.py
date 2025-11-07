# Autores: Juan Jose Vasquez, Santiago Alvarez

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. CARGA DE DATOS PROCESADOS

DATA_PATH = os.path.join("../data", "processed", "air_quality_split.npz")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("No se encontró el archivo procesado. "
                            "Ejecuta primero 'eda_preprocessing.py' para generarlo.")

data = np.load(DATA_PATH)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

print("Datos cargados correctamente:")
print(f"   Entrenamiento: {X_train.shape}")
print(f"   Validación:    {X_val.shape}")
print(f"   Prueba:        {X_test.shape}\n")

# 2. ENTRENAMIENTO DE MODELOS

# --- Modelo 1: Random Forest ---
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# --- Modelo 2: XGBoost ---
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

print("Entrenamiento completado para ambos modelos.\n")

# 3. EVALUACIÓN DE MODELOS

def evaluar_modelo(nombre, modelo, X, y):
    """Calcula y muestra las métricas de desempeño."""
    preds = modelo.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    print(f"{nombre}:")
    print(f"   MAE  = {mae:.3f}")
    print(f"   RMSE = {rmse:.3f}")
    print(f"   R²   = {r2:.3f}\n")
    return preds, mae, rmse, r2

# Evaluar los dos modelos sobre el conjunto de prueba
rf_preds, rf_mae, rf_rmse, rf_r2 = evaluar_modelo("Random Forest", rf_model, X_test, y_test)
xgb_preds, xgb_mae, xgb_rmse, xgb_r2 = evaluar_modelo("XGBoost", xgb_model, X_test, y_test)

# 4. COMPARACIÓN VISUAL DE RESULTADOS

plt.figure(figsize=(10, 5))
plt.plot(y_test[:200].tolist(), label="Real", color="black", linewidth=1.5)
plt.plot(rf_preds[:200].tolist(), label="Random Forest", color="blue", alpha=0.7)
plt.plot(xgb_preds[:200].tolist(), label="XGBoost", color="red", alpha=0.7)
plt.title("Comparación de predicciones (primeras 200 muestras)")
plt.xlabel("Índice de muestra")
plt.ylabel("Concentración de NO₂ (µg/m³)")
plt.legend()
plt.tight_layout()
plt.show()

# 5. GUARDAR MODELOS ENTRENADOS

os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/random_forest_model.pkl")
joblib.dump(xgb_model, "models/xgboost_model.pkl")

print("Modelos guardados en la carpeta src/models/")
