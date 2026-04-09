# src/gold.py
# Capa Gold - Generación de datos para ML, alertas y recomendaciones

from sqlalchemy import create_engine, text
from datetime import datetime
from .utils import get_engine, log


def generar_streamlit_manual(conn, load_type):
    """Genera productos_formato_streamlit manualmente si la funcion SQL falla"""
    if load_type == 'H':
        conn.execute(text("TRUNCATE TABLE gold.productos_formato_streamlit"))

    query = """
        INSERT INTO gold.productos_formato_streamlit (
            producto_clave, cantidad_total, categoria,
            producto_base, presentacion, fecha_desde, fecha_hasta, ultima_carga_type
        )
        SELECT
            producto || ' ' ||
                CASE
                    WHEN producto LIKE '%jarabe%' THEN 'x 120 ml'
                    WHEN producto LIKE '%ampolla%' OR producto LIKE '%inyectable%' THEN 'x 10 amp'
                    WHEN producto LIKE '%inhalador%' OR producto LIKE '%spray%' THEN 'x 200 dosis'
                    ELSE 'x 100 tab'
                END as producto_clave,
            ROUND(SUM(cantidad))::INTEGER,
            categoria,
            producto,
            CASE
                WHEN producto LIKE '%jarabe%' THEN 'x 120 ml'
                WHEN producto LIKE '%ampolla%' OR producto LIKE '%inyectable%' THEN 'x 10 amp'
                WHEN producto LIKE '%inhalador%' OR producto LIKE '%spray%' THEN 'x 200 dosis'
                ELSE 'x 100 tab'
            END,
            MIN(fecha),
            MAX(fecha),
            :load_type
        FROM silver.ventas_consolidada
        GROUP BY categoria, producto
        ON CONFLICT (producto_clave) DO UPDATE SET
            cantidad_total = CASE
                WHEN :load_type = 'H' THEN EXCLUDED.cantidad_total
                ELSE gold.productos_formato_streamlit.cantidad_total + EXCLUDED.cantidad_total
            END,
            fecha_hasta = GREATEST(gold.productos_formato_streamlit.fecha_hasta, EXCLUDED.fecha_hasta),
            ultima_carga_type = :load_type,
            ultima_actualizacion = NOW()
    """
    result = conn.execute(text(query), {'load_type': load_type})
    log(f"      Insertados/Actualizados (manual): {result.rowcount}")


def generar_recomendaciones_manual(conn, load_type):
    """Genera recomendaciones basadas en stock vs demanda"""
    if load_type == 'H':
        conn.execute(text("TRUNCATE TABLE gold.recomendaciones_stock"))

    query = """
        WITH demanda_reciente AS (
            SELECT
                categoria,
                AVG(cantidad_total) as demanda_mensual_promedio
            FROM silver.categoria_mensual
            WHERE fecha >= (SELECT MAX(fecha) - INTERVAL '3 months' FROM silver.categoria_mensual)
            GROUP BY categoria
        ),
        stock_actual AS (
            SELECT DISTINCT ON (categoria, producto, almacen)
                categoria, producto, almacen,
                stock_actual, stock_minimo, punto_reorden, dias_inventario
            FROM silver.stock_silver
            ORDER BY categoria, producto, almacen, fecha_actualizacion DESC
        )
        INSERT INTO gold.recomendaciones_stock (
            categoria, producto, almacen, fecha_actual,
            stock_actual, stock_minimo, demanda_predicha_3m,
            recomendacion, cantidad_sugerida, prioridad
        )
        SELECT
            s.categoria,
            s.producto,
            s.almacen,
            CURRENT_DATE,
            s.stock_actual,
            s.stock_minimo,
            COALESCE(d.demanda_mensual_promedio * 3, s.stock_minimo * 2) as demanda_3m,
            CASE
                WHEN s.stock_actual <= 0 THEN 'REPOSICION URGENTE'
                WHEN s.stock_actual <= s.stock_minimo THEN 'REPOSICION NECESARIA'
                WHEN s.stock_actual <= s.punto_reorden THEN 'REPOSICION PREVENTIVA'
                ELSE 'STOCK ADECUADO'
            END,
            GREATEST(0,
                CASE
                    WHEN s.stock_actual <= s.punto_reorden
                    THEN (COALESCE(d.demanda_mensual_promedio * 3, s.stock_minimo * 2) - s.stock_actual)
                    ELSE 0
                END
            )::INTEGER,
            CASE
                WHEN s.stock_actual <= 0 THEN 1
                WHEN s.stock_actual <= s.stock_minimo THEN 2
                WHEN s.stock_actual <= s.punto_reorden THEN 3
                ELSE 4
            END
        FROM stock_actual s
        LEFT JOIN demanda_reciente d ON s.categoria = d.categoria
        WHERE s.stock_actual <= s.punto_reorden * 1.5
        ON CONFLICT DO NOTHING
    """
    result = conn.execute(text(query))
    log(f"      Recomendaciones generadas: {result.rowcount}")


def ejecutar_capa_gold(load_type='H'):
    """
    Ejecuta el procesamiento de capa Gold.

    Args:
        load_type: 'H' (Full) o 'D' (Delta)
    """
    log(f"\n{'='*60}")
    log(f"CAPA GOLD - Modo: {load_type}")
    log(f"{'='*60}")

    batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    engine = get_engine()

    with engine.connect() as conn:
        trans = conn.begin()

        try:
            # 1. Generar datos de entrenamiento para ML
            log("\n[1/4] Generando ml_training_data...")
            try:
                result = conn.execute(
                    text("SELECT * FROM gold.generar_ml_training_data(:load_type)"),
                    {'load_type': load_type}
                )
                row = result.fetchone()
                log(f"      Registros: {row[0]}, Tipo: {row[1]}")
            except Exception as e:
                log(f"      Error: {e}")
                # Fallback: Insertar manualmente si la funcion falla
                if load_type == 'H':
                    conn.execute(text("TRUNCATE TABLE gold.ml_training_data"))

                query = """
                    INSERT INTO gold.ml_training_data (
                        fecha, categoria, cantidad_total,
                        lag_1_mes, lag_3_meses, lag_12_meses,
                        media_3_meses, media_6_meses,
                        mes, trimestre, es_invierno, es_verano,
                        target_1_mes, fecha_procesamiento
                    )
                    SELECT
                        fecha, categoria, cantidad_total,
                        lag_1_mes, lag_3_meses, lag_12_meses,
                        media_3_meses, media_6_meses,
                        mes, trimestre,
                        mes IN (6,7,8),
                        mes IN (12,1,2,3),
                        LEAD(cantidad_total, 1) OVER (PARTITION BY categoria ORDER BY fecha),
                        NOW()
                    FROM silver.categoria_mensual
                    ORDER BY categoria, fecha
                    ON CONFLICT (fecha, categoria) DO UPDATE SET
                        cantidad_total = EXCLUDED.cantidad_total,
                        target_1_mes = EXCLUDED.target_1_mes,
                        fecha_procesamiento = NOW()
                """
                result = conn.execute(text(query))
                log(f"      Insertados/Actualizados (manual): {result.rowcount}")

            # 2. Generar formato para Streamlit
            log("\n[2/4] Generando productos_formato_streamlit...")
            try:
                result = conn.execute(
                    text("SELECT * FROM gold.generar_productos_formato_streamlit(NULL, NULL, :load_type)"),
                    {'load_type': load_type}
                )
                rows = result.fetchall()
                log(f"      Productos generados: {len(rows)}")
                if len(rows) > 0:
                    log(f"      Ejemplo: {rows[0][0]} = {rows[0][1]}")
            except Exception as e:
                log(f"      Error en funcion SQL: {e}")
                log("      Ejecutando generacion manual...")
                generar_streamlit_manual(conn, load_type)

            # 3. Generar alertas de stock
            log("\n[3/4] Generando alertas_stock_out...")
            try:
                result = conn.execute(text("SELECT * FROM gold.generar_alertas_stock_out()"))
                row = result.fetchone()
                if row and row[0] > 0:
                    log(f"      Alertas generadas: {row[0]}")
                    log(f"      Ejemplo: {row[1]} - {row[2]}")
                else:
                    log("      No se generaron alertas nuevas (todo OK o ya existen)")
            except Exception as e:
                log(f"      Error: {e}")

            # 4. Generar recomendaciones
            log("\n[4/4] Generando recomendaciones_stock...")
            generar_recomendaciones_manual(conn, load_type)

            trans.commit()
            log(f"\n[OK] Capa Gold completada. Batch: {batch_id}")

        except Exception as e:
            trans.rollback()
            log(f"\n[ERROR] {str(e)}")
            raise


def verificar_gold():
    """Verifica el estado de las tablas Gold"""
    log("\n" + "="*60)
    log("VERIFICACION TABLAS GOLD")
    log("="*60)

    tablas = {
        'ml_training_data': 'fecha, categoria',
        'predicciones_demanda': 'fecha_prediccion',
        'recomendaciones_stock': 'categoria, prioridad',
        'productos_formato_streamlit': 'producto_clave',
        'alertas_stock_out': 'estado_alerta'
    }

    engine = get_engine()

    with engine.connect() as conn:
        for tabla, cols in tablas.items():
            try:
                count = conn.execute(text(f"SELECT COUNT(*) FROM gold.{tabla}")).scalar()
                log(f"\n{tabla}: {count} registros")

                if count > 0:
                    result = conn.execute(text(f"SELECT * FROM gold.{tabla} LIMIT 1"))
                    row = result.fetchone()
                    log(f"  Ejemplo: {dict(row._mapping) if row else 'N/A'}")
            except Exception as e:
                log(f"{tabla}: Error - {str(e)[:50]}")


# ============================================================
# EJECUCION PRINCIPAL
# ============================================================

def main():
    # Carga FULL (primera vez)
    ejecutar_capa_gold(load_type='H')
    verificar_gold()

    # Para carga DELTA diaria (descomentar cuando sea necesario):
    # ejecutar_capa_gold(load_type='D')


if __name__ == "__main__":
    main()