# streamlit_app.py
# Dashboard de Predicción de Demanda Farmacéutica - Con 3 CSVs y Capa Gold

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# ============================================================
# CONFIGURACION
# ============================================================
st.set_page_config(page_title="Predicción de Demanda", page_icon="📊", layout="wide")
st.title("📊 Sistema de Predicción de Demanda y Reabastecimiento")
st.markdown("---")

# ============================================================
# CONEXION A SUPABASE
# ============================================================
@st.cache_resource
def get_connection():
    try:
        db_url = st.secrets["database"]["url"]
    except:
        db_url = (
            "postgresql+psycopg2://"
            "postgres.xaedxfohrqtxwrlzljvh:"
            "Proyecto_Productivo"
            "@aws-0-us-west-2.pooler.supabase.com:5432/postgres"
        )
    return create_engine(db_url)

engine = get_connection()

# ============================================================
# FUNCIÓN PARA EXTENDER PREDICCIONES CON ESTACIONALIDAD
# ============================================================
def extender_predicciones_con_estacionalidad(df_pred, df_historico):
    """
    Extiende predicciones más allá de los datos reales usando:
    - Tendencia histórica
    - Estacionalidad anual (ciclo)
    - Ruido controlado
    """
    if df_pred.empty:
        return df_pred
    
    df_pred = df_pred.copy().sort_values('fecha_prediccion')
    
    if len(df_pred) == 0:
        return df_pred
    
    ultima_fecha_pred = df_pred['fecha_prediccion'].max()
    fecha_actual = datetime.now()
    
    # Si ya tenemos predicciones hasta hoy, no extender
    if ultima_fecha_pred >= fecha_actual:
        df_pred['es_extendida'] = False
        return df_pred
    
    # Calcular tendencia histórica (últimos 12 meses de datos reales)
    if df_historico is not None and len(df_historico) >= 6:
        df_hist = df_historico[df_historico['fecha'] >= fecha_actual - timedelta(days=365)].copy()
        if len(df_hist) >= 6:
            inicio_periodo = df_hist['cantidad'].iloc[:6].mean()
            fin_periodo = df_hist['cantidad'].iloc[-6:].mean()
            if inicio_periodo > 0:
                crecimiento_anual = (fin_periodo / inicio_periodo) - 1
                crecimiento_anual = max(-0.15, min(0.15, crecimiento_anual))
            else:
                crecimiento_anual = 0.02
        else:
            crecimiento_anual = 0.02
    else:
        crecimiento_anual = 0.02
    
    # Generar meses faltantes
    fechas_faltantes = pd.date_range(
        start=ultima_fecha_pred + pd.DateOffset(months=1), 
        end=fecha_actual, 
        freq='M'
    )
    
    if len(fechas_faltantes) == 0:
        df_pred['es_extendida'] = False
        return df_pred
    
    ultimo_valor = df_pred.iloc[-1]['prediccion']
    ultima_categoria = df_pred.iloc[-1]['categoria']
    ultimo_producto = df_pred.iloc[-1]['producto']
    ultimo_modelo = df_pred.iloc[-1]['modelo']
    
    predicciones_extendidas = []
    
    for i, fecha in enumerate(fechas_faltantes):
        # Tiempo transcurrido en años
        anos_desde_base = (fecha.year - ultima_fecha_pred.year) + (fecha.month - ultima_fecha_pred.month) / 12.0
        
        # 1. Tendencia
        tendencia = (1 + crecimiento_anual) ** anos_desde_base
        
        # 2. Estacionalidad mensual
        mes = fecha.month
        # Patrón: picos en invierno (julio-agosto) y primavera (marzo-abril)
        estacionalidad = 1 + 0.12 * np.sin(2 * np.pi * (mes - 7) / 12)
        estacionalidad += 0.06 * np.cos(2 * np.pi * (mes - 3) / 6)
        estacionalidad = max(0.85, min(1.15, estacionalidad))
        
        # 3. Ruido aleatorio controlado
        np.random.seed(i + int(fecha.year) * 12 + fecha.month)
        ruido = 1 + np.random.normal(0, 0.025)
        
        # 4. Combinar factores
        prediccion_extendida = ultimo_valor * tendencia * estacionalidad * ruido
        
        predicciones_extendidas.append({
            'fecha_prediccion': fecha,
            'categoria': ultima_categoria,
            'producto': ultimo_producto,
            'modelo': ultimo_modelo,
            'prediccion': max(0, prediccion_extendida),
            'intervalo_inferior': max(0, prediccion_extendida * 0.85),
            'intervalo_superior': prediccion_extendida * 1.15,
            'es_extendida': True
        })
    
    # Marcar predicciones originales
    df_pred['es_extendida'] = False
    
    # Combinar
    if predicciones_extendidas:
        df_extendido = pd.DataFrame(predicciones_extendidas)
        df_pred = pd.concat([df_pred, df_extendido], ignore_index=True)
    
    return df_pred

# ============================================================
# CARGAR DATOS DESDE GOLD (3 CSVs integrados)
# ============================================================
@st.cache_data(ttl=300)
def cargar_datos():
    # 1. Productos disponibles (desde ventas consolidadas)
    query_productos = """
        SELECT DISTINCT categoria, producto 
        FROM silver.ventas_consolidada 
        ORDER BY categoria, producto
    """
    df_productos = pd.read_sql(query_productos, engine)
    
    # 2. Ventas históricas (desde silver)
    query_ventas = """
        SELECT fecha, categoria, producto, cantidad 
        FROM silver.ventas_consolidada 
        ORDER BY categoria, producto, fecha
    """
    df_ventas = pd.read_sql(query_ventas, engine)
    df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])
    
    # 3. Predicciones por CATEGORÍA (para referencia)
    query_predicciones = """
        SELECT * FROM gold.predicciones_demanda 
        ORDER BY categoria, fecha_prediccion
    """
    df_predicciones = pd.read_sql(query_predicciones, engine)
    if not df_predicciones.empty:
        df_predicciones['fecha_prediccion'] = pd.to_datetime(df_predicciones['fecha_prediccion'])
    
    # 4. PREDICCIONES POR PRODUCTO
    try:
        query_pred_producto = """
            SELECT * FROM gold.predicciones_producto 
            ORDER BY categoria, producto, fecha_prediccion
        """
        df_pred_producto = pd.read_sql(query_pred_producto, engine)
        if not df_pred_producto.empty:
            df_pred_producto['fecha_prediccion'] = pd.to_datetime(df_pred_producto['fecha_prediccion'])
    except Exception as e:
        st.warning(f"Tabla gold.predicciones_producto no encontrada. Usando predicciones por categoría.")
        df_pred_producto = pd.DataFrame()
    
    # 5. Recomendaciones de stock
    query_recomendaciones = """
        SELECT * FROM gold.recomendaciones_stock 
        ORDER BY prioridad, demanda_predicha_3m DESC
    """
    df_recomendaciones = pd.read_sql(query_recomendaciones, engine)
    
    # 6. Stock actual (último estado)
    query_stock = """
        SELECT DISTINCT ON (categoria, producto, almacen)
            fecha_actualizacion, almacen, categoria, producto,
            stock_actual, stock_minimo, stock_maximo, punto_reorden,
            dias_inventario
        FROM silver.stock_silver
        ORDER BY categoria, producto, almacen, fecha_actualizacion DESC
    """
    df_stock = pd.read_sql(query_stock, engine)
    
    return df_productos, df_ventas, df_predicciones, df_pred_producto, df_recomendaciones, df_stock

try:
    df_productos, df_ventas, df_predicciones, df_pred_producto, df_recomendaciones, df_stock = cargar_datos()
    st.success(f"✅ Conectado | {len(df_productos)} productos | {len(df_predicciones)} predicciones categoría | {len(df_pred_producto)} predicciones producto")
except Exception as e:
    st.error(f"Error de conexión: {e}")
    st.stop()

# ============================================================
# SIDEBAR - VERSIÓN CORREGIDA
# ============================================================
st.sidebar.header("🔍 Filtros")

categorias = df_productos['categoria'].unique().tolist()
categoria_sel = st.sidebar.selectbox("1. Seleccionar Categoría", categorias)

productos_categoria = df_productos[df_productos['categoria'] == categoria_sel]['producto'].tolist()
producto_sel = st.sidebar.selectbox("2. Seleccionar Producto", productos_categoria)

# Mostrar info de stock en sidebar
stock_prod = df_stock[(df_stock['categoria'] == categoria_sel) & (df_stock['producto'] == producto_sel)]
if not stock_prod.empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📦 Stock Actual")
    st.sidebar.metric("Stock", f"{stock_prod.iloc[0]['stock_actual']:.0f} unid.")
    st.sidebar.metric("Stock Mínimo", f"{stock_prod.iloc[0]['stock_minimo']:.0f} unid.")
    if stock_prod.iloc[0]['stock_actual'] < stock_prod.iloc[0]['stock_minimo']:
        st.sidebar.warning("⚠️ Stock por debajo del mínimo")

# Mostrar predicción por PRODUCTO en sidebar (NO categoría)
pred_producto_sidebar = df_pred_producto[
    (df_pred_producto['categoria'] == categoria_sel) & 
    (df_pred_producto['producto'] == producto_sel)
].sort_values('fecha_prediccion')

if not pred_producto_sidebar.empty:
    ultima_pred = pred_producto_sidebar.iloc[-1]
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Predicción")
    st.sidebar.metric("Próximo mes", f"{ultima_pred['prediccion']:.0f} tabletas")
    st.sidebar.caption(f"Modelo: {ultima_pred['modelo']} | MAPE: {ultima_pred['mape']:.1f}%")
else:
    # Fallback a predicción por categoría
    pred_cat_sidebar = df_predicciones[df_predicciones['categoria'] == categoria_sel].sort_values('fecha_prediccion')
    if not pred_cat_sidebar.empty:
        ultima_pred = pred_cat_sidebar.iloc[-1]
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 Predicción")
        st.sidebar.metric("Próximo mes", f"{ultima_pred['prediccion']:.0f} (estimado)")
        st.sidebar.caption(f"Modelo: {ultima_pred['modelo']}")

# ============================================================
# DATOS DEL PRODUCTO
# ============================================================
df_prod = df_ventas[
    (df_ventas['categoria'] == categoria_sel) & 
    (df_ventas['producto'] == producto_sel)
].copy().sort_values('fecha')

if len(df_prod) == 0:
    st.error(f"No hay datos para {producto_sel}")
    st.stop()

# Obtener predicciones para este producto específico
pred_producto = df_pred_producto[
    (df_pred_producto['categoria'] == categoria_sel) & 
    (df_pred_producto['producto'] == producto_sel)
].copy().sort_values('fecha_prediccion')

# Obtener predicción de categoría (fallback)
pred_cat = df_predicciones[df_predicciones['categoria'] == categoria_sel].copy().sort_values('fecha_prediccion')

# Decidir qué predicción usar
if not pred_producto.empty:
    predicciones_base = pred_producto
    tipo_prediccion = "por producto"
elif not pred_cat.empty:
    predicciones_base = pred_cat
    tipo_prediccion = "por categoría (aproximado)"
else:
    predicciones_base = pd.DataFrame()
    tipo_prediccion = "ninguna"

# EXTENDER PREDICCIONES CON ESTACIONALIDAD
if not predicciones_base.empty:
    predicciones_usar = extender_predicciones_con_estacionalidad(predicciones_base, df_prod)
    # Actualizar tipo_prediccion si se extendió
    if predicciones_usar['es_extendida'].any():
        tipo_prediccion = f"{tipo_prediccion} (extendido con estacionalidad)"
else:
    predicciones_usar = pd.DataFrame()

# Última fecha real
ultima_fecha_real = df_prod['fecha'].max()

# Métricas del producto
ultimo_valor = df_prod['cantidad'].iloc[-1]
promedio_3m = df_prod['cantidad'].tail(3).mean()
promedio_12m = df_prod['cantidad'].tail(12).mean() if len(df_prod) >= 12 else promedio_3m
tendencia_pct = ((promedio_3m - promedio_12m) / promedio_12m * 100) if promedio_12m > 0 else 0

# ============================================================
# METRICAS PRINCIPALES
# ============================================================
st.subheader(f"📦 {producto_sel} ({categoria_sel})")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Categoría", categoria_sel)
with col2:
    if not predicciones_usar.empty:
        st.metric("Modelo", f"{predicciones_usar.iloc[-1]['modelo']} ({tipo_prediccion})")
    else:
        st.metric("Modelo", "Pendiente")
with col3:
    st.metric("Último Mes", f"{ultimo_valor:.0f}")
with col4:
    st.metric("Promedio 3M", f"{promedio_3m:.0f}", delta=f"{tendencia_pct:+.1f}%")
with col5:
    if not predicciones_usar.empty:
        ultima_prediccion = predicciones_usar[predicciones_usar['fecha_prediccion'] <= datetime.now()].iloc[-1]['prediccion'] if len(predicciones_usar[predicciones_usar['fecha_prediccion'] <= datetime.now()]) > 0 else predicciones_usar.iloc[-1]['prediccion']
        st.metric("Predicción", f"{ultima_prediccion:.0f}")
    else:
        st.metric("Predicción", "N/A")

st.markdown("---")

# ============================================================
# GRAFICO HISTORICO + PREDICCIONES CONTINUAS
# ============================================================
st.subheader(f"📈 Evolución de Demanda - {producto_sel}")

fig = go.Figure()

# Datos históricos (línea azul sólida)
fig.add_trace(go.Scatter(
    x=df_prod['fecha'],
    y=df_prod['cantidad'],
    mode='lines+markers',
    name='Histórico (datos reales)',
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=4)
))

# Predicciones (línea roja discontinua)
if not predicciones_usar.empty:
    # Separar originales y extendidas para diferente estilo
    pred_originales = predicciones_usar[predicciones_usar['es_extendida'] == False] if 'es_extendida' in predicciones_usar.columns else predicciones_usar
    pred_extendidas = predicciones_usar[predicciones_usar['es_extendida'] == True] if 'es_extendida' in predicciones_usar.columns else pd.DataFrame()
    
    if not pred_originales.empty:
        fig.add_trace(go.Scatter(
            x=pred_originales['fecha_prediccion'],
            y=pred_originales['prediccion'],
            mode='lines+markers',
            name=f'Predicción ({tipo_prediccion})',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(color='red', size=4, symbol='circle')
        ))
    
    if not pred_extendidas.empty:
        fig.add_trace(go.Scatter(
            x=pred_extendidas['fecha_prediccion'],
            y=pred_extendidas['prediccion'],
            mode='lines+markers',
            name=f'Predicción extendida (estacionalidad)',
            line=dict(color='orange', width=2, dash='dot'),
            marker=dict(color='orange', size=3, symbol='diamond')
        ))
    
    # Intervalos de confianza
    if 'intervalo_inferior' in predicciones_usar.columns and 'intervalo_superior' in predicciones_usar.columns:
        fig.add_trace(go.Scatter(
            x=predicciones_usar['fecha_prediccion'].tolist() + predicciones_usar['fecha_prediccion'].tolist()[::-1],
            y=predicciones_usar['intervalo_superior'].tolist() + predicciones_usar['intervalo_inferior'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.15)',
            line=dict(color='rgba(255,0,0,0)'),
            name='Intervalo 90%'
        ))

# Línea vertical divisoria (separación real/predicción)
fig.add_vline(x=ultima_fecha_real, line_dash="dash", line_color="gray", opacity=0.7)

fig.update_layout(
    title=f'{producto_sel} - Demanda Real y Predicción',
    xaxis_title='Fecha',
    yaxis_title='Cantidad Vendida',
    height=500,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TABLA DE RESULTADOS Y GUARDADO DE PROYECCIONES
# ============================================================
st.markdown("---")
st.subheader("📋 Tabla de Resultados y Proyecciones")

# Preparar datos para la tabla
if not predicciones_usar.empty:
    # Obtener últimos meses reales
    ultimos_meses = df_prod.tail(4).copy()
    
    # Crear lista para la tabla
    tabla_datos = []
    
    # Agregar meses históricos
    for _, row in ultimos_meses.iterrows():
        tabla_datos.append({
            "MEDICAMENTO": f"{producto_sel} x 100 tab",
            "MES": row['fecha'].strftime('%B'),
            "CANTIDAD": round(row['cantidad'], 2),
            "ESTADO": "Histórico"
        })
    
    # Agregar predicción del próximo mes
    if not predicciones_usar.empty:
        ultima_pred = predicciones_usar[predicciones_usar['fecha_prediccion'] >= datetime.now()].iloc[0] if len(predicciones_usar[predicciones_usar['fecha_prediccion'] >= datetime.now()]) > 0 else predicciones_usar.iloc[-1]
        tabla_datos.append({
            "MEDICAMENTO": f"{producto_sel} x 100 tab",
            "MES": ultima_pred['fecha_prediccion'].strftime('%B'),
            "CANTIDAD": round(ultima_pred['prediccion'], 2),
            "ESTADO": "Proyección"
        })
    
    df_tabla_resultados = pd.DataFrame(tabla_datos)
    st.dataframe(df_tabla_resultados, use_container_width=True, hide_index=True)
    
    # Botón para guardar proyección
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("💾 Guardar Proyección en Base de Datos", type="primary"):
            try:
                with engine.connect() as conn:
                    trans = conn.begin()
                    
                    # Insertar en gold.proyecciones_guardadas
                    conn.execute(text("""
                        INSERT INTO gold.proyecciones_guardadas 
                        (categoria, producto, mes, cantidad_predicha, modelo_utilizado, confirmada)
                        VALUES (:categoria, :producto, :mes, :cantidad, :modelo, :confirmada)
                    """), {
                        'categoria': categoria_sel,
                        'producto': producto_sel,
                        'mes': ultima_pred['fecha_prediccion'],
                        'cantidad': float(ultima_pred['prediccion']),
                        'modelo': ultima_pred['modelo'],
                        'confirmada': True
                    })
                    
                    trans.commit()
                    st.success(f"✅ Proyección guardada correctamente para {producto_sel}")
            except Exception as e:
                st.error(f"❌ Error al guardar: {e}")
    
    with col_btn2:
        if st.button("📊 Ver Historial de Proyecciones"):
            try:
                df_historial = pd.read_sql(f"""
                    SELECT * FROM gold.proyecciones_guardadas 
                    WHERE producto = '{producto_sel}'
                    ORDER BY fecha_registro DESC
                    LIMIT 10
                """, engine)
                if not df_historial.empty:
                    st.dataframe(df_historial[['fecha_registro', 'mes', 'cantidad_predicha', 'modelo_utilizado']])
                else:
                    st.info("No hay proyecciones guardadas para este producto")
            except Exception as e:
                st.info("Tabla de historial aún sin datos")
    
    with col_btn3:
        if st.button("📤 Exportar a CSV"):
            csv = df_tabla_resultados.to_csv(index=False)
            st.download_button(
                label="Descargar CSV",
                data=csv,
                file_name=f"proyeccion_{producto_sel}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
else:
    st.info("No hay predicciones disponibles para mostrar")

# ============================================================
# RECOMENDACIONES DE STOCK
# ============================================================
st.markdown("---")
st.subheader("⚠️ Alertas y Recomendaciones")

recom_prod = df_recomendaciones[df_recomendaciones['producto'] == producto_sel]

if not recom_prod.empty:
    for _, rec in recom_prod.iterrows():
        prioridad_color = {
            1: ("🔴", "#ff4b4b"),
            2: ("🟠", "#ffa500"),
            3: ("🟡", "#ffcc00"),
            4: ("🟢", "#00cc66")
        }.get(rec['prioridad'], ("⚪", "#cccccc"))
        
        st.markdown(f"""
        <div style='border-left: 4px solid {prioridad_color[1]}; padding: 10px; margin: 10px 0; background-color: #f0f2f6; border-radius: 5px;'>
            <b>{prioridad_color[0]} {rec['recomendacion']}</b><br>
            📦 Stock actual: <b>{rec['stock_actual']:.0f}</b> | Mínimo: {rec['stock_minimo']:.0f}<br>
            📊 Demanda predicha 3M: <b>{rec['demanda_predicha_3m']:.0f}</b><br>
            🛒 Cantidad sugerida: <b>{rec['cantidad_sugerida']:.0f}</b> unidades
        </div>
        """, unsafe_allow_html=True)
else:
    if not stock_prod.empty and stock_prod.iloc[0]['stock_actual'] < stock_prod.iloc[0]['stock_minimo']:
        st.warning(f"⚠️ **Stock bajo!** Actual: {stock_prod.iloc[0]['stock_actual']:.0f} | Mínimo: {stock_prod.iloc[0]['stock_minimo']:.0f}")
    else:
        st.info("✅ Sin alertas activas para este producto")

# ============================================================
# TABLA HISTORICA
# ============================================================
with st.expander("📋 Ver historial de ventas"):
    df_tabla = df_prod[['fecha', 'cantidad']].copy()
    df_tabla['fecha'] = df_tabla['fecha'].dt.strftime('%Y-%m-%d')
    df_tabla.columns = ['Fecha', 'Cantidad']
    st.dataframe(df_tabla, use_container_width=True)

# ============================================================
# COMPARATIVO DE PRODUCTOS EN CATEGORIA
# ============================================================
st.markdown("---")
st.subheader(f"📊 Comparativo de Productos en {categoria_sel}")

df_cat_todos = df_ventas[df_ventas['categoria'] == categoria_sel].copy()

fig_comp = go.Figure()
for prod in df_cat_todos['producto'].unique()[:10]:
    df_p = df_cat_todos[df_cat_todos['producto'] == prod]
    fig_comp.add_trace(go.Scatter(
        x=df_p['fecha'],
        y=df_p['cantidad'],
        mode='lines',
        name=prod,
        opacity=0.7,
        line=dict(width=3 if prod == producto_sel else 1)
    ))

fig_comp.update_layout(
    title=f'Todos los productos de {categoria_sel}',
    xaxis_title='Fecha',
    yaxis_title='Cantidad',
    height=400
)

st.plotly_chart(fig_comp, use_container_width=True)

# ============================================================
# RESUMEN DE STOCK POR CATEGORIA
# ============================================================
st.markdown("---")
st.subheader("📊 Resumen de Stock por Categoría")

df_stock_cat = df_stock[df_stock['categoria'] == categoria_sel].copy()

if not df_stock_cat.empty:
    total_stock = df_stock_cat['stock_actual'].sum()
    productos_bajos = df_stock_cat[df_stock_cat['stock_actual'] < df_stock_cat['stock_minimo']].shape[0]
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Stock Total Categoría", f"{total_stock:.0f} unid.")
    with col_b:
        st.metric("Productos con Stock Bajo", productos_bajos)
    with col_c:
        st.metric("Productos en Categoría", len(df_stock_cat))
    
    st.dataframe(
        df_stock_cat[['producto', 'almacen', 'stock_actual', 'stock_minimo']]
        .sort_values('stock_actual')
        .style.apply(lambda x: ['background-color: #ffcccc' if x['stock_actual'] < x['stock_minimo'] else '' for _ in x], axis=1),
        use_container_width=True
    )
else:
    st.info("No hay información de stock para esta categoría")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Sistema de Predicción de Demanda - Arquitectura Medallion | Datos desde capa Gold")