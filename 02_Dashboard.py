import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Quejas Telecomunicaciones", layout="wide")

@st.cache_data
def load_main_data():
    return pd.read_csv(
        "df_telecom_final.csv",
        parse_dates=["fecha_ingreso"]
    )

@st.cache_data
def load_poblacion():
    return pd.read_csv("poblacion_edos.csv")

df = load_main_data()
df_pob = load_poblacion()

st.title("Análisis de Quejas en Telecomunicaciones entre 2022 y 2025 (PROFECO)")

def multiselect_all(label, options, key):
    """
    Multiselect con botón real de seleccionar / deseleccionar todos
    Funciona correctamente en apps desplegadas
    """

    if key not in st.session_state:
        st.session_state[key] = options

    col1, col2 = st.columns([3, 1])

    with col1:
        selection = st.multiselect(
            label,
            options,
            default=st.session_state[key],
            key=f"{key}_multiselect"
        )

    with col2:
        if st.button("Todos", key=f"{key}_all"):
            st.session_state[key] = options
            st.rerun()

        if st.button("Ninguno", key=f"{key}_none"):
            st.session_state[key] = []
            st.rerun()

    st.session_state[key] = selection
    return selection


## BLOQUE 1 : Panorama general

st.header("Bloque 1: Comparación de compañías")

# Filtros
f1, f2, f3, f4 = st.columns(4)

with f1:
    anios_b1 = multiselect_all(
        "Año",
        sorted(df["anio"].dropna().unique()),
        key="b1_anio"
    )

with f2:
    estados_b1 = multiselect_all(
        "Estado",
        sorted(df["estado"].dropna().unique()),
        key="b1_estado"
    )

with f3:
    proveedores_b1 = multiselect_all(
        "Proveedor",
        sorted(df["proveedor_top"].dropna().unique()),
        key="b1_proveedor"
    )

with f4:
    top_n = st.slider(
        "Top N proveedores",
        min_value=1,
        max_value=10,
        value=5
    )


# Filtros

df_b1 = df.copy()

if anios_b1:
    df_b1 = df_b1[df_b1["anio"].isin(anios_b1)]

if estados_b1:
    df_b1 = df_b1[df_b1["estado"].isin(estados_b1)]

if proveedores_b1:
    df_b1 = df_b1[df_b1["proveedor_top"].isin(proveedores_b1)]

#kpis
k1, k2, k3 = st.columns(3)

k1.metric("Total de quejas", f"{len(df_b1):,}")

k2.metric(
    "Tasa de conciliación",
    f"{df_b1['resuelta'].mean()*100:.1f}%"
)

k3.metric(
    "Mediana días de resolución",
    f"{df_b1['dias_resolucion'].median():.0f}"
)
# Gráfica 1 Evolución mensua
top_prov = (
    df_b1["proveedor_top"]
    .value_counts()
    .head(top_n)
    .index
)

df_line = (
    df_b1[df_b1["proveedor_top"].isin(top_prov)]
    .groupby([
        df_b1["fecha_ingreso"].dt.to_period("M"),
        "proveedor_top"
    ])
    .size()
    .reset_index(name="quejas")
)

df_line["fecha_ingreso"] = df_line["fecha_ingreso"].astype(str)

fig1 = px.line(
    df_line,
    x="fecha_ingreso",
    y="quejas",
    color="proveedor_top",
    title="Evolución mensual de quejas"
)

st.plotly_chart(fig1, use_container_width=True)

# Gráfica 2 — Quejas vs conciliadas
df_bar = (
    df_b1.groupby("proveedor_top")
    .agg(
        total_quejas=("resuelta", "count"),
        conciliadas=("resuelta", "sum")
    )
    .reset_index()
)

df_bar = df_bar.melt(
    id_vars="proveedor_top",
    value_vars=["total_quejas", "conciliadas"],
    var_name="tipo",
    value_name="cantidad"
)

fig2 = px.bar(
    df_bar,
    x="proveedor_top",
    y="cantidad",
    color="tipo",
    barmode="group",
    title="Quejas totales vs conciliadas"
)

st.plotly_chart(fig2, use_container_width=True)

# Griaficas 3 y 4
c1, c2 = st.columns(2)

with c1:
    fig3 = px.pie(
        df_b1,
        names="tipo_servicio",
        title="Inconformidades por tipo de servicio"
    )
    st.plotly_chart(fig3, use_container_width=True)

with c2:
    fig4 = px.pie(
        df_b1,
        names="estado_procesal",
        title="Estatus de inconformidades"
    )
    st.plotly_chart(fig4, use_container_width=True)

# Gráfica 5 Tiempo de atención

df_time = (
    df_b1.groupby("proveedor_top")["dias_resolucion"]
    .median()
    .reset_index()
)

fig5 = px.bar(
    df_time,
    x="proveedor_top",
    y="dias_resolucion",
    title="Mediana de días de resolución por compañía"
)

st.plotly_chart(fig5, use_container_width=True)




## BLOQUE 2 — Comparación entre estados

st.header("Bloque 2: Comparación entre estados")

anio_sel = st.selectbox(
    "Selecciona año",
    sorted(df["anio"].dropna().unique())
)

proveedores_b2 = multiselect_all(
    "Proveedor",
    sorted(df["proveedor_top"].unique()),
    key="b2_proveedor"
)

df_b2 = df[
    (df["anio"] == anio_sel) &
    (df["proveedor_top"].isin(proveedores_b2))
]

# Gráfica 6: Tabla por 100,000 hab
tabla_estados = df_b2.groupby("estado").size().reset_index(name="quejas")

pob_col = str(anio_sel)
tabla_estados = tabla_estados.merge(
    df_pob[["estado", pob_col]],
    on="estado",
    how="left"
)

tabla_estados["quejas_100k"] = (
    tabla_estados["quejas"] / tabla_estados[pob_col] * 100000
)

tabla_estados = tabla_estados.sort_values(
    "quejas_100k",
    ascending=False
)

st.subheader("Inconformidades por estado (por 100,000 habitantes)")
st.dataframe(tabla_estados)

#Gráfica 7: Top proovedores por edo
estado_sel = st.selectbox(
    "Selecciona estado",
    sorted(df_b2["estado"].unique())
)

top5 = (
    df_b2[df_b2["estado"] == estado_sel]
    .groupby("proveedor_top")
    .size()
    .sort_values(ascending=False)
    .head(5)
    .reset_index(name="quejas")
)

st.subheader("Top 5 proveedores con más inconformidades")
st.dataframe(top5)


## BLOQUE 3 Motivos de reclamación
st.header("Bloque 3: Motivos de reclamación")

prov_b3 = multiselect_all(
    "Proveedor",
    sorted(df["proveedor_top"].unique()),
    key="b3_proveedor"
)

problema_sel = st.selectbox(
    "Problema clasificado",
    sorted(df["problema_clasificado"].unique())
)

top_n_motivos = st.slider(
    "Número de motivos a mostrar",
    min_value=5,
    max_value=20,
    value=10
)

df_b3 = df[
    (df["proveedor_top"].isin(prov_b3)) &
    (df["problema_clasificado"] == problema_sel)
]


# Gráfica 8 motivos y % de conciliación

tabla_motivos = (
    df_b3.groupby("motivo_reclamacion")
    .agg(
        frecuencia=("resuelta", "count"),
        tasa_conciliacion=("resuelta", "mean")
    )
    .reset_index()
    .sort_values("frecuencia", ascending=False)
    .head(top_n_motivos)
)

tabla_motivos["tasa_conciliacion"] *= 100

st.dataframe(tabla_motivos)

