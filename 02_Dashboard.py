import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Quejas Telecomunicaciones México",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv(
        'df_quejas_telecom_final.csv',
        parse_dates=['fecha_ingreso']
    )

df = load_data()

st.title("Análisis de Quejas en Telecomunicaciones (PROFECO)")
st.markdown("**Periodo 2022–2025 | Dashboard Analítico y Visual**")

# Bloque 1

st.header("Panorama General")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total de quejas", f"{len(df):,}")

with col2:
    tasa = df['resuelta'].mean() * 100
    st.metric("Tasa de conciliación", f"{tasa:.1f}%")

with col3:
    tiempo = df['dias_resolucion'].median()
    st.metric("Mediana días resolución", f"{tiempo:.0f}")

# Evolución temporal

quejas_mes = (
    df.groupby(df['fecha_ingreso'].dt.to_period('M'))
      .size()
      .reset_index(name='quejas')
)
quejas_mes['fecha_ingreso'] = quejas_mes['fecha_ingreso'].astype(str)

fig = px.line(
    quejas_mes,
    x='fecha_ingreso',
    y='quejas',
    title='Evolución mensual de quejas'
)
st.plotly_chart(fig, use_container_width=True)



# Bloque 2: proveedores

st.header("Comparación entre proveedores")

proveedor_sel = st.multiselect(
    "Selecciona proveedor",
    df['proveedor_top'].unique(),
    default=['TELCEL', 'MOVISTAR', 'AT&T', 'TELMEX']
)

df_f = df[df['proveedor_top'].isin(proveedor_sel)]

fig = px.bar(
    df_f['proveedor_top'].value_counts().reset_index(),
    x='proveedor_top',
    y='count',
    title='Número de quejas por proveedor'
)
st.plotly_chart(fig, use_container_width=True)

# heatmap x problema 

tabla = pd.crosstab(df_f['proveedor_top'], df_f['problema_clasificado'])

fig = px.imshow(
    tabla,
    text_auto=True,
    title='Distribución de problemas por proveedor'
)
st.plotly_chart(fig, use_container_width=True)

# Bloque 3 : tipos de problemas
st.header("Tipos de Problemas")

fig = px.pie(
    df,
    names='problema_clasificado',
    title='Distribución de problemas'
)
st.plotly_chart(fig, use_container_width=True)

# Top motivos

top_motivos = (
    df.groupby('motivo_reclamacion')
      .size()
      .sort_values(ascending=False)
      .head(10)
      .reset_index(name='conteo')
)

fig = px.bar(
    top_motivos,
    x='conteo',
    y='motivo_reclamacion',
    orientation='h',
    title='Top 10 motivos de reclamación'
)
st.plotly_chart(fig, use_container_width=True)

# Bloque 4: análisis economico

st.header("Impacto Económico")

fig = px.scatter(
    df,
    x='monto_reclamado',
    y='monto_recuperado',
    color='proveedor_top',
    title='Monto reclamado vs recuperado',
    log_x=True,
    log_y=True
)
st.plotly_chart(fig, use_container_width=True)


#recuperación promedio
recup = (
    df.groupby('proveedor_top')['porcentaje_recuperado']
      .mean()
      .reset_index()
)

fig = px.bar(
    recup,
    x='proveedor_top',
    y='porcentaje_recuperado',
    title='Porcentaje promedio recuperado'
)
st.plotly_chart(fig, use_container_width=True)


#bloque 5 pca

st.header("Similitud entre proveedores (PCA)")



perfil = df.groupby('proveedor_top').agg({
    'dias_resolucion': 'mean',
    'resuelta': 'mean',
    'monto_reclamado': 'mean'
})

X = StandardScaler().fit_transform(perfil)

pca = PCA(n_components=2)
coords = pca.fit_transform(X)

perfil_pca = pd.DataFrame(
    coords,
    columns=['PC1', 'PC2'],
    index=perfil.index
).reset_index()

fig = px.scatter(
    perfil_pca,
    x='PC1',
    y='PC2',
    text='proveedor_top',
    title='PCA de proveedores'
)
st.plotly_chart(fig, use_container_width=True)

# Correlaciones

st.header("Correlaciones")

corr = df[['monto_reclamado', 'dias_resolucion', 'resuelta']].corr()

fig = px.imshow(
    corr,
    text_auto=True,
    title='Matriz de correlación'
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
 **Nota importante:**  
La correlación no implica causalidad.  
Estas relaciones indican asociación estadística, no causa directa.
""")
