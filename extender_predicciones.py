# extender_predicciones.py
import sys
import os

# Agregar la carpeta src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import get_engine
import pandas as pd
from datetime import datetime

engine = get_engine()

df_ultimas = pd.read_sql("""
    SELECT DISTINCT ON (categoria) 
        categoria, modelo, prediccion, intervalo_inferior, intervalo_superior
    FROM gold.predicciones_demanda
    ORDER BY categoria, fecha_prediccion DESC
""", engine)

fechas = pd.date_range('2020-01-01', datetime.now(), freq='M')
nuevas = []

for _, row in df_ultimas.iterrows():
    for fecha in fechas:
        nuevas.append({
            'fecha_prediccion': fecha,
            'categoria': row['categoria'],
            'modelo': row['modelo'],
            'prediccion': row['prediccion'],
            'intervalo_inferior': row['intervalo_inferior'],
            'intervalo_superior': row['intervalo_superior']
        })

if nuevas:
    df_nuevas = pd.DataFrame(nuevas)
    df_nuevas.to_sql('predicciones_demanda', engine, schema='gold', if_exists='append', index=False)
    print(f'✅ Insertadas {len(df_nuevas)} predicciones')
else:
    print('❌ No se generaron predicciones')