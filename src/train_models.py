# src/train_models.py
# Entrenamiento multi-modelo a nivel PRODUCTO

import pandas as pd
import numpy as np
import warnings
import joblib
from datetime import datetime
from sqlalchemy import text
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from .utils import get_engine, log

warnings.filterwarnings('ignore')

# ============================================================
# CLASE PARA ENTRENAMIENTO POR PRODUCTO
# ============================================================

class ModeloProducto:
    def __init__(self, categoria, producto):
        self.categoria = categoria
        self.producto = producto
        self.modelos = {}
        self.mape_scores = {}
        self.mejor_modelo = None
        self.mejor_mape = float('inf')
        self.scaler = StandardScaler()

    def preparar_features(self, df):
        """Prepara features para ML"""
        df = df.copy()
        
        # Features básicas
        features = ['lag_1_mes', 'lag_3_meses', 'lag_6_meses', 'lag_12_meses',
                   'media_3_meses', 'media_6_meses', 'mes', 'es_invierno', 'es_verano']
        
        # Features cíclicas para mes
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        features.extend(['mes_sin', 'mes_cos'])
        
        return df[features].fillna(0), df['target_1_mes'].values

    def entrenar_ridge(self, X_train, X_test, y_train, y_test):
        """Ridge Regression"""
        try:
            X_train_s = self.scaler.fit_transform(X_train)
            X_test_s = self.scaler.transform(X_test)
            
            model = Ridge(alpha=10.0)
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            
            mape = np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1))) * 100
            self.modelos['Ridge'] = {'modelo': model, 'mape': mape, 'preds': preds}
            self.mape_scores['Ridge'] = mape
            return mape
        except Exception as e:
            log(f"      Ridge Error: {e}")
            return float('inf')

    def entrenar_xgboost(self, X_train, X_test, y_train, y_test):
        """XGBoost"""
        try:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            mape = np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1))) * 100
            self.modelos['XGBoost'] = {'modelo': model, 'mape': mape, 'preds': preds}
            self.mape_scores['XGBoost'] = mape
            return mape
        except Exception as e:
            log(f"      XGBoost Error: {e}")
            return float('inf')

    def entrenar_sarima(self, serie, test_size=6):
        """SARIMA para series temporales"""
        try:
            if len(serie) < 24:
                raise ValueError("Datos insuficientes")
            
            train = serie[:-test_size]
            test = serie[-test_size:]
            
            model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                          enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            forecast = results.forecast(steps=test_size)
            
            mape = np.mean(np.abs((test - forecast) / np.maximum(test, 1))) * 100
            self.modelos['SARIMA'] = {'modelo': results, 'mape': mape, 'preds': forecast}
            self.mape_scores['SARIMA'] = mape
            return mape
        except Exception as e:
            log(f"      SARIMA Error: {e}")
            return float('inf')

    def entrenar_prophet(self, df):
        """Prophet"""
        try:
            df_prophet = df[['fecha', 'cantidad_total']].rename(
                columns={'fecha': 'ds', 'cantidad_total': 'y'})
            
            split = len(df_prophet) - 6
            train = df_prophet.iloc[:split]
            test = df_prophet.iloc[split:]
            
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(train)
            
            future = model.make_future_dataframe(periods=6, freq='M')
            forecast = model.predict(future)
            preds = forecast['yhat'].iloc[-6:].values
            
            mape = np.mean(np.abs((test['y'].values - preds) / np.maximum(test['y'].values, 1))) * 100
            self.modelos['Prophet'] = {'modelo': model, 'mape': mape, 'preds': preds}
            self.mape_scores['Prophet'] = mape
            return mape
        except Exception as e:
            log(f"      Prophet Error: {e}")
            return float('inf')

    def seleccionar_mejor(self):
        if self.mape_scores:
            self.mejor_modelo = min(self.mape_scores, key=self.mape_scores.get)
            self.mejor_mape = self.mape_scores[self.mejor_modelo]
        return self.mejor_modelo

    def predecir_siguiente(self, df):
        """Predice el siguiente mes"""
        if not self.mejor_modelo:
            return None
        
        ganador = self.modelos[self.mejor_modelo]
        ultima_fecha = df['fecha'].iloc[-1]
        siguiente_fecha = ultima_fecha + pd.DateOffset(months=1)
        
        if self.mejor_modelo == 'Ridge':
            X, _ = self.preparar_features(df)
            X_last = self.scaler.transform(X.iloc[[-1]])
            pred = ganador['modelo'].predict(X_last)[0]
        elif self.mejor_modelo == 'XGBoost':
            X, _ = self.preparar_features(df)
            pred = ganador['modelo'].predict(X.iloc[[-1]])[0]
        elif self.mejor_modelo == 'SARIMA':
            pred = ganador['modelo'].forecast(steps=1)[0]
        elif self.mejor_modelo == 'Prophet':
            future = ganador['modelo'].make_future_dataframe(periods=1, freq='M')
            forecast = ganador['modelo'].predict(future)
            pred = forecast['yhat'].iloc[-1]
        else:
            pred = df['cantidad_total'].iloc[-1]
        
        return {
            'fecha_prediccion': siguiente_fecha,
            'categoria': self.categoria,
            'producto': self.producto,
            'modelo': self.mejor_modelo,
            'prediccion': max(0, pred),
            'mape': self.mejor_mape
        }


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def entrenar_modelos_producto():
    engine = get_engine()
    log("=" * 80)
    log("ENTRENAMIENTO MULTI-MODELO A NIVEL PRODUCTO")
    log("=" * 80)
    
    # 1. Preparar datos de entrenamiento por producto
    log("Preparando datos de entrenamiento por producto...")
    
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE gold.ml_training_producto"))
        
        # Generar datos agregados por producto
        conn.execute(text("""
            INSERT INTO gold.ml_training_producto (fecha, categoria, producto, cantidad_total, mes, anio, trimestre)
            SELECT 
                fecha,
                categoria,
                producto,
                SUM(cantidad) as cantidad_total,
                EXTRACT(MONTH FROM fecha) as mes,
                EXTRACT(YEAR FROM fecha) as anio,
                EXTRACT(QUARTER FROM fecha) as trimestre
            FROM silver.ventas_consolidada
            GROUP BY fecha, categoria, producto
            ORDER BY categoria, producto, fecha
        """))
        
        # Calcular lags
        conn.execute(text("""
            UPDATE gold.ml_training_producto t
            SET 
                lag_1_mes = (
                    SELECT cantidad_total FROM gold.ml_training_producto t2
                    WHERE t2.categoria = t.categoria AND t2.producto = t.producto
                    AND t2.fecha = t.fecha - INTERVAL '1 month'
                ),
                lag_3_meses = (
                    SELECT AVG(cantidad_total) FROM gold.ml_training_producto t2
                    WHERE t2.categoria = t.categoria AND t2.producto = t.producto
                    AND t2.fecha BETWEEN t.fecha - INTERVAL '3 months' AND t.fecha - INTERVAL '1 month'
                ),
                lag_6_meses = (
                    SELECT AVG(cantidad_total) FROM gold.ml_training_producto t2
                    WHERE t2.categoria = t.categoria AND t2.producto = t.producto
                    AND t2.fecha BETWEEN t.fecha - INTERVAL '6 months' AND t.fecha - INTERVAL '1 month'
                ),
                lag_12_meses = (
                    SELECT AVG(cantidad_total) FROM gold.ml_training_producto t2
                    WHERE t2.categoria = t.categoria AND t2.producto = t.producto
                    AND t2.fecha BETWEEN t.fecha - INTERVAL '12 months' AND t.fecha - INTERVAL '1 month'
                ),
                media_3_meses = lag_3_meses,
                media_6_meses = lag_6_meses,
                target_1_mes = (
                    SELECT cantidad_total FROM gold.ml_training_producto t2
                    WHERE t2.categoria = t.categoria AND t2.producto = t.producto
                    AND t2.fecha = t.fecha + INTERVAL '1 month'
                )
        """))
        
        conn.execute(text("COMMIT"))
    
    # 2. Cargar datos
    df = pd.read_sql("""
        SELECT * FROM gold.ml_training_producto
        WHERE target_1_mes IS NOT NULL
        ORDER BY categoria, producto, fecha
    """, engine)
    
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    productos = df[['categoria', 'producto']].drop_duplicates()
    log(f"Productos a entrenar: {len(productos)}")
    
    # 3. Entrenar por producto
    resultados = []
    
    for _, row in productos.iterrows():
        cat = row['categoria']
        prod = row['producto']
        log(f"\n📦 Entrenando: {prod} ({cat})")
        
        df_prod = df[(df['categoria'] == cat) & (df['producto'] == prod)].copy()
        
        if len(df_prod) < 12:
            log(f"   ⚠️ Saltando: solo {len(df_prod)} meses de datos")
            continue
        
        modelo = ModeloProducto(cat, prod)
        
        # Preparar features
        X, y = modelo.preparar_features(df_prod)
        split = len(X) - 6
        
        if split <= 12:
            log(f"   ⚠️ Datos insuficientes para validación")
            continue
        
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Entrenar modelos
        log("   Entrenando modelos...")
        modelo.entrenar_ridge(X_train, X_test, y_train, y_test)
        modelo.entrenar_xgboost(X_train, X_test, y_train, y_test)
        modelo.entrenar_sarima(df_prod['cantidad_total'].values)
        modelo.entrenar_prophet(df_prod)
        
        ganador = modelo.seleccionar_mejor()
        log(f"   🏆 Mejor modelo: {ganador} (MAPE: {modelo.mejor_mape:.1f}%)")
        
        # Generar predicción
        pred = modelo.predecir_siguiente(df_prod)
        if pred:
            resultados.append(pred)
        
        # Guardar modelo
        joblib.dump(modelo, f'modelo_{cat}_{prod}.joblib')
    
    # 4. Guardar predicciones
    if resultados:
        df_pred = pd.DataFrame(resultados)
        df_pred.to_sql('predicciones_producto', engine, schema='gold', if_exists='replace', index=False)
        log(f"\n✅ Guardadas {len(df_pred)} predicciones por producto")
    
    log("\n" + "=" * 80)
    log("ENTRENAMIENTO COMPLETADO")
    log("=" * 80)


if __name__ == "__main__":
    entrenar_modelos_producto()