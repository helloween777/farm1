# src/utils.py
# Utilidades compartidas para todas las capas (Bronze, Silver, Gold, ML)

import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================
# Conexión a Supabase
# ============================
def get_engine():
    """Retorna conexión a Supabase"""
    DATABASE_URL = (
        "postgresql+psycopg2://"
        "postgres.xaedxfohrqtxwrlzljvh:"
        "Proyecto_Productivo"
        "@aws-0-us-west-2.pooler.supabase.com:5432/postgres"
    )
    return create_engine(DATABASE_URL, pool_pre_ping=True)


# ============================
# Funciones de Batch y Hash
# ============================
def generar_batch_id():
    """Genera identificador único para el lote de carga"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def calcular_hash_md5(row_dict):
    """
    Calcula hash MD5 de los datos de negocio para detectar cambios.
    No incluye campos de auditoria (load_type, batch_id, fecha_ingesta).
    """
    campos_ordenados = sorted(row_dict.keys())
    valores = []

    for campo in campos_ordenados:
        valor = row_dict[campo]
        if pd.isna(valor):
            valores.append('')
        else:
            valores.append(str(valor))

    contenido = '|'.join(valores)
    return hashlib.md5(contenido.encode()).hexdigest()


def limpiar_dataframe(df, columnas_requeridas):
    """
    Limpia y valida un dataframe antes de carga.
    """
    df = df.copy()

    # Eliminar columnas que no existen en el destino
    columnas_existentes = [col for col in columnas_requeridas if col in df.columns]
    df = df[columnas_existentes]

    # Manejar valores nulos
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('')
        elif 'int' in str(df[col].dtype):
            df[col] = df[col].fillna(0)
        elif 'float' in str(df[col].dtype):
            df[col] = df[col].fillna(0.0)

    return df


# ============================
# Funciones de métricas ML
# ============================
def calcular_mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calcular_mae(y_true, y_pred):
    """Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)


def calcular_rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ============================
# Funciones de transformación de fechas
# ============================
def agregar_features_temporales(df, col_fecha="fecha"):
    """Agrega columnas de año, mes, trimestre, etc. a un DataFrame evitando duplicados"""
    df = df.copy()
    df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce")

    if "anio" not in df.columns:
        df["anio"] = df[col_fecha].dt.year
    if "mes" not in df.columns:
        df["mes"] = df[col_fecha].dt.month
    if "trimestre" not in df.columns:
        df["trimestre"] = df[col_fecha].dt.quarter
    if "dia_del_anio" not in df.columns:
        df["dia_del_anio"] = df[col_fecha].dt.dayofyear
    if "es_invierno" not in df.columns:
        df["es_invierno"] = df["mes"].isin([12, 1, 2])
    if "es_verano" not in df.columns:
        df["es_verano"] = df["mes"].isin([6, 7, 8])

    return df


# ============================
# Logging simple
# ============================
def log(msg):
    """Imprime mensajes con formato estándar"""
    print(f"[INFO] {msg}")