# src/silver.py
# Capa Silver - Procesamiento usando funciones SQL almacenadas

from sqlalchemy import create_engine, text
from datetime import datetime
from .utils import get_engine, log


def ejecutar_procesamiento_silver(load_type='D', batch_id_bronze=None):
    """
    Ejecuta el procesamiento de capa Silver usando las funciones SQL.

    Args:
        load_type: 'H' (Full) o 'D' (Delta)
        batch_id_bronze: Si es modo D, filtrar solo ese batch de Bronze.
                          Si es None, procesa todo lo pendiente de tipo D.
    """
    log(f"\n{'='*60}")
    log(f"CAPA SILVER - Modo: {load_type}")
    log(f"{'='*60}")

    batch_silver = datetime.now().strftime('%Y%m%d_%H%M%S')
    engine = get_engine()

    with engine.connect() as conn:
        trans = conn.begin()

        try:
            # 1. Procesar ventas_productos -> ventas_consolidada
            log("\n[1/3] Procesando ventas_productos...")
            result = conn.execute(
                text("SELECT * FROM silver.procesar_ventas_productos_silver(:load_type, :batch_id)"),
                {
                    'load_type': load_type,
                    'batch_id': batch_id_bronze
                }
            )
            row = result.fetchone()
            log(f"      Registros: {row[0]}, Insertados/Actualizados: {row[1]}, Tipo: {row[2]}")

            # 2. Procesar ventas_categoria -> categoria_mensual (con lags)
            log("\n[2/3] Procesando categoria_mensual...")
            result = conn.execute(
                text("SELECT * FROM silver.procesar_categoria_mensual_silver(:load_type, :batch_id)"),
                {
                    'load_type': load_type,
                    'batch_id': batch_id_bronze
                }
            )
            row = result.fetchone()
            log(f"      Registros: {row[0]}, Insertados/Actualizados: {row[1]}, Tipo: {row[2]}")

            # 3. Procesar stock_huancayo -> stock_silver
            log("\n[3/3] Procesando stock...")
            result = conn.execute(
                text("SELECT * FROM silver.procesar_stock_silver(:load_type, :batch_id)"),
                {
                    'load_type': load_type,
                    'batch_id': batch_id_bronze
                }
            )
            row = result.fetchone()
            log(f"      Registros: {row[0]}, Insertados/Actualizados: {row[1]}, Tipo: {row[2]}")

            trans.commit()
            log(f"\n[OK] Silver completado. Batch Silver: {batch_silver}")

        except Exception as e:
            trans.rollback()
            log(f"\n[ERROR] {str(e)}")
            raise


def verificar_silver():
    """Verifica el estado actual de las tablas Silver"""
    log("\n" + "="*60)
    log("VERIFICACION TABLAS SILVER")
    log("="*60)

    queries = {
        'ventas_consolidada': "SELECT source_load_type, COUNT(*) FROM silver.ventas_consolidada GROUP BY source_load_type",
        'categoria_mensual': "SELECT source_load_type, COUNT(*) FROM silver.categoria_mensual GROUP BY source_load_type",
        'stock_silver': "SELECT source_load_type, COUNT(*) FROM silver.stock_silver GROUP BY source_load_type"
    }

    engine = get_engine()

    with engine.connect() as conn:
        for tabla, sql in queries.items():
            try:
                result = conn.execute(text(sql))
                rows = result.fetchall()
                log(f"\n{tabla}:")
                for row in rows:
                    log(f"  Modo {row[0]}: {row[1]} registros")

                # Verificar integridad bronze_id
                result = conn.execute(text(f"SELECT COUNT(*) FROM silver.{tabla} WHERE bronze_id IS NULL"))
                nulls = result.fetchone()[0]
                if nulls > 0:
                    log(f"  [ADVERTENCIA] {nulls} registros sin bronze_id")

            except Exception as e:
                log(f"{tabla}: Error - {e}")


# ============================================================
# EJECUCION PRINCIPAL
# ============================================================

def main():
    # EJEMPLO 1: Carga FULL (H) - Primera vez o reconstrucción completa
    log("EJECUCION 1: CARGA FULL")
    ejecutar_procesamiento_silver(load_type='H')
    verificar_silver()

    # EJEMPLO 2: Carga DELTA (D) - Procesar solo un batch específico de Bronze
    # Descomentar cuando sea necesario:
    # log("\nEJECUCION 2: CARGA DELTA")
    # ejecutar_procesamiento_silver(load_type='D', batch_id_bronze='20260408_174459')
    # verificar_silver()


if __name__ == "__main__":
    main()