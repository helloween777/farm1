# src/bronze.py
# Capa Bronze - Carga de datos con soporte FULL (H) y DELTA (D)

from sqlalchemy import create_engine, text
import pandas as pd
from .utils import get_engine, log, generar_batch_id, calcular_hash_md5, limpiar_dataframe

# ============================================================
# FUNCIONES DE CARGA BRONZE
# ============================================================

def cargar_stock_huancayo(df, load_type='D', batch_id=None):
    """
    Carga datos de stock a bronze.stock_huancayo

    Args:
        df: DataFrame con columnas del CSV stock_huancayo.csv
        load_type: 'H' (Full - trunca e inserta) o 'D' (Delta - upsert)
        batch_id: Identificador del lote (opcional)
    """
    if batch_id is None:
        batch_id = generar_batch_id()

    log(f"[STOCK] Iniciando carga modo {load_type}, batch: {batch_id}")

    columnas_negocio = [
        'fecha_actualizacion', 'almacen', 'categoria', 'producto',
        'stock_actual', 'stock_minimo', 'stock_maximo', 'punto_reorden',
        'lote', 'fecha_vencimiento', 'estado', 'ubicacion_almacen', 'dias_inventario'
    ]

    df_clean = limpiar_dataframe(df, columnas_negocio)

    engine = get_engine()
    registros_procesados = 0

    with engine.connect() as conn:
        trans = conn.begin()

        try:
            # Modo Full: Truncar tabla
            if load_type == 'H':
                conn.execute(text("TRUNCATE TABLE bronze.stock_huancayo"))
                log(f"[STOCK] Tabla truncada (modo H)")

            # Preparar statement
            sql = text("""
                INSERT INTO bronze.stock_huancayo (
                    fecha_actualizacion, almacen, categoria, producto,
                    stock_actual, stock_minimo, stock_maximo, punto_reorden,
                    lote, fecha_vencimiento, estado, ubicacion_almacen,
                    dias_inventario, load_type, batch_id, hash_fila,
                    source_system, fecha_ingesta
                ) VALUES (
                    :fecha_actualizacion, :almacen, :categoria, :producto,
                    :stock_actual, :stock_minimo, :stock_maximo, :punto_reorden,
                    :lote, :fecha_vencimiento, :estado, :ubicacion_almacen,
                    :dias_inventario, :load_type, :batch_id, :hash_fila,
                    'stock_huancayo.csv', NOW()
                )
                ON CONFLICT (fecha_actualizacion, almacen, categoria, producto, load_type)
                DO UPDATE SET
                    stock_actual = EXCLUDED.stock_actual,
                    stock_minimo = EXCLUDED.stock_minimo,
                    stock_maximo = EXCLUDED.stock_maximo,
                    punto_reorden = EXCLUDED.punto_reorden,
                    lote = EXCLUDED.lote,
                    fecha_vencimiento = EXCLUDED.fecha_vencimiento,
                    estado = EXCLUDED.estado,
                    ubicacion_almacen = EXCLUDED.ubicacion_almacen,
                    dias_inventario = EXCLUDED.dias_inventario,
                    hash_fila = EXCLUDED.hash_fila,
                    fecha_ingesta = NOW()
                WHERE bronze.stock_huancayo.hash_fila != EXCLUDED.hash_fila
            """)

            for _, row in df_clean.iterrows():
                row_dict = row.to_dict()
                hash_val = calcular_hash_md5(row_dict)

                conn.execute(sql, {
                    'fecha_actualizacion': row_dict.get('fecha_actualizacion'),
                    'almacen': row_dict.get('almacen'),
                    'categoria': row_dict.get('categoria'),
                    'producto': row_dict.get('producto'),
                    'stock_actual': int(row_dict.get('stock_actual', 0)),
                    'stock_minimo': int(row_dict.get('stock_minimo', 0)),
                    'stock_maximo': int(row_dict.get('stock_maximo', 0)) if row_dict.get('stock_maximo') else None,
                    'punto_reorden': int(row_dict.get('punto_reorden', 0)) if row_dict.get('punto_reorden') else None,
                    'lote': str(row_dict.get('lote', '')),
                    'fecha_vencimiento': row_dict.get('fecha_vencimiento'),
                    'estado': row_dict.get('estado', 'normal'),
                    'ubicacion_almacen': row_dict.get('ubicacion_almacen', ''),
                    'dias_inventario': int(row_dict.get('dias_inventario', 0)),
                    'load_type': load_type,
                    'batch_id': batch_id,
                    'hash_fila': hash_val
                })
                registros_procesados += 1

            trans.commit()
            log(f"[STOCK] Procesados {registros_procesados} registros")
            return batch_id

        except Exception as e:
            trans.rollback()
            log(f"[STOCK] ERROR: {str(e)}")
            raise


def cargar_ventas_productos(df, load_type='D', batch_id=None):
    """
    Carga datos de ventas detalladas a bronze.ventas_productos
    """
    if batch_id is None:
        batch_id = generar_batch_id()

    log(f"[VENTAS_PROD] Iniciando carga modo {load_type}, batch: {batch_id}")

    columnas_negocio = ['fecha', 'categoria', 'producto', 'cantidad']
    df_clean = limpiar_dataframe(df, columnas_negocio)

    engine = get_engine()

    with engine.connect() as conn:
        trans = conn.begin()

        try:
            if load_type == 'H':
                conn.execute(text("TRUNCATE TABLE bronze.ventas_productos"))
                log(f"[VENTAS_PROD] Tabla truncada (modo H)")

            sql = text("""
                INSERT INTO bronze.ventas_productos (
                    fecha, categoria, producto, cantidad,
                    load_type, batch_id, hash_fila, source_system, fecha_ingesta
                ) VALUES (
                    :fecha, :categoria, :producto, :cantidad,
                    :load_type, :batch_id, :hash_fila, 'ventas_productos.csv', NOW()
                )
                ON CONFLICT (fecha, categoria, producto, load_type)
                DO UPDATE SET
                    cantidad = EXCLUDED.cantidad,
                    hash_fila = EXCLUDED.hash_fila,
                    fecha_ingesta = NOW()
                WHERE bronze.ventas_productos.hash_fila != EXCLUDED.hash_fila
            """)

            registros = 0
            for _, row in df_clean.iterrows():
                row_dict = row.to_dict()
                hash_val = calcular_hash_md5(row_dict)

                conn.execute(sql, {
                    'fecha': row_dict.get('fecha'),
                    'categoria': row_dict.get('categoria'),
                    'producto': row_dict.get('producto'),
                    'cantidad': float(row_dict.get('cantidad', 0)),
                    'load_type': load_type,
                    'batch_id': batch_id,
                    'hash_fila': hash_val
                })
                registros += 1

            trans.commit()
            log(f"[VENTAS_PROD] Procesados {registros} registros")
            return batch_id

        except Exception as e:
            trans.rollback()
            log(f"[VENTAS_PROD] ERROR: {str(e)}")
            raise


def cargar_ventas_categoria(df, load_type='D', batch_id=None):
    """
    Carga datos agregados por categoria a bronze.ventas_categoria
    Corresponde al archivo salesmonthly.csv
    """
    if batch_id is None:
        batch_id = generar_batch_id()

    log(f"[VENTAS_CAT] Iniciando carga modo {load_type}, batch: {batch_id}")

    # Transformar de wide a long
    if 'datum' in df.columns:
        df_melted = df.melt(
            id_vars=['datum'],
            var_name='categoria',
            value_name='cantidad_total'
        )
        df_melted.rename(columns={'datum': 'fecha'}, inplace=True)
    elif 'fecha' in df.columns:
        df_melted = df.melt(
            id_vars=['fecha'],
            var_name='categoria',
            value_name='cantidad_total'
        )
    else:
        raise ValueError("CSV debe tener columna 'datum' o 'fecha'")

    columnas_negocio = ['fecha', 'categoria', 'cantidad_total']
    df_clean = limpiar_dataframe(df_melted, columnas_negocio)

    engine = get_engine()

    with engine.connect() as conn:
        trans = conn.begin()

        try:
            if load_type == 'H':
                conn.execute(text("TRUNCATE TABLE bronze.ventas_categoria"))
                log(f"[VENTAS_CAT] Tabla truncada (modo H)")

            sql = text("""
                INSERT INTO bronze.ventas_categoria (
                    fecha, categoria, cantidad_total,
                    load_type, batch_id, hash_fila, source_system, fecha_ingesta
                ) VALUES (
                    :fecha, :categoria, :cantidad_total,
                    :load_type, :batch_id, :hash_fila, 'salesmonthly.csv', NOW()
                )
                ON CONFLICT (fecha, categoria, load_type)
                DO UPDATE SET
                    cantidad_total = EXCLUDED.cantidad_total,
                    hash_fila = EXCLUDED.hash_fila,
                    fecha_ingesta = NOW()
                WHERE bronze.ventas_categoria.hash_fila != EXCLUDED.hash_fila
            """)

            registros = 0
            for _, row in df_clean.iterrows():
                row_dict = row.to_dict()
                hash_val = calcular_hash_md5(row_dict)

                conn.execute(sql, {
                    'fecha': row_dict.get('fecha'),
                    'categoria': row_dict.get('categoria'),
                    'cantidad_total': float(row_dict.get('cantidad_total', 0)),
                    'load_type': load_type,
                    'batch_id': batch_id,
                    'hash_fila': hash_val
                })
                registros += 1

            trans.commit()
            log(f"[VENTAS_CAT] Procesados {registros} registros")
            return batch_id

        except Exception as e:
            trans.rollback()
            log(f"[VENTAS_CAT] ERROR: {str(e)}")
            raise


# ============================================================
# FUNCION DE CARGA ORQUESTADA
# ============================================================

def ejecutar_carga_bronze(archivo_csv, tipo_tabla, load_type='D', batch_id=None):
    """
    Orquesta la carga de cualquiera de los 3 archivos.

    Args:
        archivo_csv: Ruta al archivo CSV
        tipo_tabla: 'stock', 'ventas_productos' o 'ventas_categoria'
        load_type: 'H' o 'D'
        batch_id: Opcional, para agrupar cargas relacionadas
    """
    log(f"\n{'='*60}")
    log(f"CARGA BRONZE: {tipo_tabla} | Modo: {load_type}")
    log(f"{'='*60}")

    # Leer CSV
    try:
        df = pd.read_csv(archivo_csv, encoding='utf-8')
        log(f"Archivo leido: {len(df)} filas, {len(df.columns)} columnas")
    except Exception as e:
        log(f"Error leyendo CSV: {e}")
        return None

    # Ejecutar carga segun tipo
    if tipo_tabla == 'stock':
        return cargar_stock_huancayo(df, load_type, batch_id)
    elif tipo_tabla == 'ventas_productos':
        return cargar_ventas_productos(df, load_type, batch_id)
    elif tipo_tabla == 'ventas_categoria':
        return cargar_ventas_categoria(df, load_type, batch_id)
    else:
        raise ValueError(f"Tipo de tabla no valido: {tipo_tabla}")


# ============================================================
# EJECUCION PRINCIPAL
# ============================================================

def main():
    # 1. Conexión a Supabase
    engine = get_engine()
    log("Conectado a Supabase - Capa Bronze")

    # Ejemplo 1: Carga INICIAL (H - Full) de todos los datos historicos
    log("\n" + "="*60)
    log("EJEMPLO 1: CARGA FULL (H) - Inicial o reinicio completo")
    log("="*60)

    batch_h = generar_batch_id()

    try:
        # Cargar los 3 archivos en modo Full con el mismo batch_id
        ejecutar_carga_bronze('data/stock_huancayo.csv', 'stock', load_type='H', batch_id=batch_h)
        ejecutar_carga_bronze('data/ventas_productos.csv', 'ventas_productos', load_type='H', batch_id=batch_h)
        ejecutar_carga_bronze('data/salesmonthly.csv', 'ventas_categoria', load_type='H', batch_id=batch_h)

        log(f"\nCarga H completada. Batch ID: {batch_h}")

    except Exception as e:
        log(f"Error en carga H: {e}")
        raise

    # Ejemplo 2: Carga DELTA (D) - Solo nuevos datos o actualizaciones
    log("\n" + "="*60)
    log("EJEMPLO 2: CARGA DELTA (D) - Actualizacion diaria")
    log("="*60)

    try:
        df_stock_delta = pd.read_csv('data/stock_huancayo.csv').tail(10)
        batch_d = generar_batch_id()

        ejecutar_carga_bronze('data/stock_huancayo_delta.csv', 'stock', load_type='D', batch_id=batch_d)

        log(f"\nCarga D completada. Batch ID: {batch_d}")

    except Exception as e:
        log(f"Error en carga D: {e}")

    # Verificacion final
    log("\n" + "="*60)
    log("VERIFICACION FINAL EN BASE DE DATOS")
    log("="*60)

    with engine.connect() as conn:
        tablas = ['ventas_productos', 'ventas_categoria', 'stock_huancayo']

        for tabla in tablas:
            result = conn.execute(text(f"SELECT load_type, COUNT(*) FROM bronze.{tabla} GROUP BY load_type"))
            rows = result.fetchall()
            log(f"\nTabla bronze.{tabla}:")
            for row in rows:
                log(f"  Modo {row[0]}: {row[1]} registros")


if __name__ == "__main__":
    main()