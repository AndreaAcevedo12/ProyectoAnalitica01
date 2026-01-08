import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import pearsonr


st.set_page_config(page_title="Quejas Telecomunicaciones", layout="wide")

# CARGA DE DATOS


@st.cache_data
def load_main_data():
    return pd.read_csv(
        "df_telecom_final.csv",
        parse_dates=["fecha_ingreso"]
    )

@st.cache_data
def load_poblacion():
    return pd.read_csv("poblacion_edos.csv")

@st.cache_data
def load_perfiles():
    return pd.read_csv("perfil_proveedores_raw.csv", index_col=0)

@st.cache_data
def load_distancias():
    return pd.read_csv("distancia_euclidiana_proveedores.csv", index_col=0)

@st.cache_data
def load_pca():
    return pd.read_csv("pca_proveedores.csv", index_col=0)

@st.cache_data
def load_mds():
    return pd.read_csv("mds_proveedores.csv", index_col=0)

@st.cache_data
def load_corr():
    return pd.read_csv("correlacion_pearson_proveedores.csv", index_col=0)

df = load_main_data()
df_pob = load_poblacion()

perfil_df = load_perfiles()
dist_df = load_distancias()
pca_df = load_pca()
mds_df = load_mds()
corr_df = load_corr()

st.title("Análisis de Quejas en Telecomunicaciones (PROFECO 2022–2025)")

def limpiar_clave(series):
    if series is None: return None
    # Mayúsculas, sin espacios y sin acentos para asegurar el merge interno
    return series.astype(str).str.upper().str.strip().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

# 1. Claves para DF Principal
df["_key_estado"] = limpiar_clave(df["estado"])
df["_key_prov"] = limpiar_clave(df["nombre_comercial"])

# 2. Claves para DF Población
col_estado_pob = [c for c in df_pob.columns if "estado" in c.lower() or "entidad" in c.lower()][0]
df_pob["_key_estado"] = limpiar_clave(df_pob[col_estado_pob])

# Detectar columna de población (Año más reciente)
cols_anios = [c for c in df_pob.columns if str(c).strip() in ['2025','2024','2023','2022','2020']]
cols_anios.sort(reverse=True)
col_pob_target = cols_anios[0] if cols_anios else df_pob.select_dtypes('number').columns[-1]
df_pob["_pob_uso"] = df_pob[col_pob_target]

# 3. Claves para Perfil (Usuarios)
if "proveedor_top" not in perfil_df.columns:
    perfil_df = perfil_df.reset_index()
    if "proveedor_top" not in perfil_df.columns:
         perfil_df.rename(columns={perfil_df.columns[0]: "proveedor_top"}, inplace=True)

perfil_df["_key_prov"] = limpiar_clave(perfil_df["proveedor_top"])


# PESTAÑAS


tab1, tab2, tab3 = st.tabs([
    "Análisis descriptivo",
    "Análisis económico e inferencial",
    "Análisis multivariado de proveedores"
])


# TAB 1 — DASHBOARD ORIGINAL (BLOQUES 1, 2 y 3)

with tab1:
    st.markdown("### Panorama general de quejas")
    
    # Filtros
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            f_min, f_max = df["fecha_ingreso"].min(), df["fecha_ingreso"].max()
            date_range = st.date_input("Periodo", [f_min, f_max])
        with c2:
            all_states = sorted(df["estado"].unique())
            sel_states = st.multiselect("Estados", all_states, default=all_states)
        with c3:
            all_provs = sorted(df["nombre_comercial"].unique())
            top_5 = df["nombre_comercial"].value_counts().head(5).index.tolist()
            sel_provs = st.multiselect("Proveedores", all_provs, default=top_5)

    # Aplicar filtros
    if len(date_range) == 2:
        mask = (
            (df["fecha_ingreso"].dt.date >= date_range[0]) & 
            (df["fecha_ingreso"].dt.date <= date_range[1]) &
            (df["estado"].isin(sel_states)) &
            (df["nombre_comercial"].isin(sel_provs))
        )
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()

    # KPIs
    total_q = len(df_filtered)
    try:
        if "conciliada" in df_filtered.columns:
            conciliadas = df_filtered["conciliada"].sum()
        else:
            conciliadas = df_filtered["estado_procesal"].astype(str).str.contains("Conciliada|Favor", case=False).sum()
        pct_concil = (conciliadas / total_q * 100) if total_q > 0 else 0
    except:
        pct_concil = 0
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Quejas", f"{total_q:,}")
    k2.metric("Proveedores", f"{df_filtered['nombre_comercial'].nunique()}")
    k3.metric("% Conciliación", f"{pct_concil:.1f}%")
    
    st.markdown("---")

    # Bloque 1: Evolución
    
    # Bloque 1: Evolución temporal (DIARIA)

    st.subheader("Tendencia temporal")
    
    # CONTROLES
    col_ctrl_1, col_ctrl_2 = st.columns(2)
    
    with col_ctrl_1:
        tipo_suavizado = st.selectbox(
            "Tipo de suavizado",
            ["Sin suavizar", "Media móvil", "Mediana móvil"]
        )
    
    with col_ctrl_2:
        ventana = st.slider(
            "Ventana (días)",
            min_value=3,
            max_value=30,
            value=7
        )
    
    # AGREGACIÓN DIARIA
    df_evo = (
        df_filtered
        .groupby(["dia", "nombre_comercial", "_key_prov"])
        .size()
        .reset_index(name="conteo")
        .sort_values("dia")
    )
    
    # APLICAR SUAVIZADO
    if tipo_suavizado != "Sin suavizar":
        df_evo["conteo_suave"] = (
            df_evo
            .groupby("nombre_comercial")["conteo"]
            .transform(
                lambda x: (
                    x.rolling(window=ventana, center=True).mean()
                    if tipo_suavizado == "Media móvil"
                    else x.rolling(window=ventana, center=True).median()
                )
            )
        )
        y_plot = "conteo_suave"
        y_label = f"Quejas ({tipo_suavizado.lower()})"
    else:
        y_plot = "conteo"
        y_label = "Quejas diarias"
    
    # GRÁFICAS
    col_evo_1, col_evo_2 = st.columns(2)
    
    # 1 Volumen absoluto
    with col_evo_1:
        st.markdown("**1. Volumen absoluto (diario)**")
    
        fig_abs = px.line(
            df_evo,
            x="dia",
            y=y_plot,
            color="nombre_comercial",
            title="Evolución diaria de quejas",
            labels={"dia": "Fecha", y_plot: y_label}
        )
    
        fig_abs.update_layout(legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_abs, use_container_width=True)
    
    #  2 Tasa por usuarios
    with col_evo_2:
        st.markdown("**2. Tasa diaria ponderada por usuarios**")
    
        df_evo_rel = pd.merge(
            df_evo,
            perfil_df[["_key_prov", "usuarios_totales"]],
            on="_key_prov",
            how="left"
        )
    
        df_evo_rel["usuarios_totales"] = (
            pd.to_numeric(df_evo_rel["usuarios_totales"], errors="coerce")
            .fillna(1)
        )
    
        df_evo_rel["tasa"] = (
            df_evo_rel[y_plot] / df_evo_rel["usuarios_totales"] * 10000
        )
    
        df_plot_rel = df_evo_rel[df_evo_rel["usuarios_totales"] > 100]
    
        if not df_plot_rel.empty:
            fig_rel = px.line(
                df_plot_rel,
                x="dia",
                y="tasa",
                color="nombre_comercial",
                title="Tasa diaria por 10,000 usuarios",
                labels={"dia": "Fecha", "tasa": "Tasa"}
            )
    
            fig_rel.update_layout(legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig_rel, use_container_width=True)
        else:
            st.info("No hay información suficiente de usuarios para calcular la tasa.")


    # Bloque 2: Ranking geográfico
    
    st.markdown("---")
    c_rank_head, c_rank_opt = st.columns([2, 1])
    with c_rank_head:
        st.subheader("Ranking por estado")
    with c_rank_opt:
        # Checkbox para quitar CDMX
        ocultar_cdmx_rank = st.checkbox("Ocultar CDMX", value=False, key="rank_exclude")

    try:
        # 1. Agrupar
        quejas_edo = df_filtered["_key_estado"].value_counts().reset_index()
        quejas_edo.columns = ["_key_estado", "quejas"]
        
        # 2. Merge con población
        df_ranking = pd.merge(quejas_edo, df_pob, on="_key_estado", how="left")
        
        # 3. Calcular tasa
        df_ranking["_pob_uso"] = pd.to_numeric(df_ranking["_pob_uso"], errors='coerce').fillna(1)
        df_ranking["tasa_100k"] = (df_ranking["quejas"] / df_ranking["_pob_uso"]) * 100000
        
        # 4. Recuperar nombre
        lookup_names = df_filtered[["_key_estado", "estado"]].drop_duplicates().set_index("_key_estado")
        df_ranking["nombre_estado"] = df_ranking["_key_estado"].map(lookup_names["estado"]).fillna(df_ranking["_key_estado"])
        
        # 5. Filtro para ocultar la cdmx
        if ocultar_cdmx_rank:
            df_ranking = df_ranking[~df_ranking["_key_estado"].str.contains("CIUDAD", regex=True)]

        # 6. Ordenar y graficar
        df_ranking = df_ranking.sort_values("tasa_100k", ascending=True) 
        
        # Ajustamos altura dinámica según la cantidad de estados
        altura_grafica = max(400, len(df_ranking) * 25)

        fig_rank = px.bar(
            df_ranking,
            x="tasa_100k",
            y="nombre_estado",
            orientation='h',
            text_auto=".1f",
            color="tasa_100k",
            color_continuous_scale="Reds",
            title=f"Incidencia (Quejas x 100k hab) - Base: {col_pob_target}",
            labels={"tasa_100k": "Tasa", "nombre_estado": ""}
        )
        fig_rank.update_layout(height=altura_grafica)
        st.plotly_chart(fig_rank, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error calculando el ranking: {e}")

    # Bloque 3: Resolución y matriz de calor

    st.markdown("---")
    r2_c1, r2_c2 = st.columns(2)

    with r2_c1:
        st.subheader("Resolución (%)")
        
        df_stack = df_filtered.groupby(["nombre_comercial", "estado_procesal"]).size().reset_index(name="conteo")
        totals = df_stack.groupby("nombre_comercial")["conteo"].transform("sum")
        df_stack["porcentaje"] = (df_stack["conteo"] / totals) * 100
        
        order_prov = df_filtered["nombre_comercial"].value_counts().index
        
        fig_stack = px.bar(
            df_stack, y="nombre_comercial", x="porcentaje", color="estado_procesal",
            orientation='h', text_auto=".0f", category_orders={"nombre_comercial": order_prov}
        )
        fig_stack.update_layout(barmode='stack', legend=dict(orientation="h", y=-0.2), yaxis_title=None)
        st.plotly_chart(fig_stack, use_container_width=True)

    with r2_c2:
        st.subheader("Mapa de calor de quejas por estado y proveedor")
        
        # Controles del Heatmap
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            exclude_cdmx_heat = st.checkbox("Ocultar CDMX", value=False, key="heat_exclude")
        with col_h2:
            # Opción para normalizar o no
            modo_heatmap = st.radio("Métrica", ["Conteo total", "Tasa x 100k"], horizontal=True)
        
        # Filtro de datos
        df_heat_s = df_filtered.copy()
        if exclude_cdmx_heat:
            df_heat_s = df_heat_s[~df_heat_s["_key_estado"].str.contains("CIUDAD", regex=True)]
            
        top_p = df_heat_s["nombre_comercial"].value_counts().head(10).index
        top_e = df_heat_s["estado"].value_counts().head(10).index
        
        df_heat = df_heat_s[
            (df_heat_s["nombre_comercial"].isin(top_p)) &
            (df_heat_s["estado"].isin(top_e))
        ]
        
        if not df_heat.empty:
            matriz = pd.crosstab(df_heat["nombre_comercial"], df_heat["estado"])
            
            if modo_heatmap == "Tasa x 100k":
                # Lógica de Normalización
                pob_lookup = df_pob.set_index("_key_estado")["_pob_uso"]
                cols_clean = [limpiar_clave(pd.Series([x]))[0] for x in matriz.columns]
                vals_pob = pob_lookup.reindex(cols_clean).fillna(1).values
                matriz_final = matriz.div(vals_pob, axis=1) * 100000
                fmt = ".1f"
                titulo_leg = "Tasa"
            else:
                # Lógica de Conteo Puro
                matriz_final = matriz
                fmt = "d"
                titulo_leg = "Quejas"
            
            fig_heat = px.imshow(
                matriz_final,
                text_auto=fmt,
                aspect="auto",
                color_continuous_scale="Viridis" if modo_heatmap == "Conteo Total" else "Magma",
                labels=dict(x="Estado", y="Proveedor", color=titulo_leg),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Datos insuficientes para la selección actual.")

    # Bloque 4: motivos dee reclamación
    st.markdown("---")
    st.subheader("Detalle de motivos de reclamación")

    # Crear columna año auxiliar si no existe
    if "año" not in df.columns:
        df["año"] = df["fecha_ingreso"].dt.year

    # Listas para filtros
    unique_provs = sorted(df["nombre_comercial"].unique())
    unique_years = sorted(df["año"].unique())
    unique_problems = sorted(df["problema_clasificado"].dropna().unique())

    # Contenedor de filtros específicos para este bloque
    c_f_prov, c_f_anio, c_f_prob = st.columns(3)
    
    with c_f_prov:
        sel_prov_top = st.selectbox("Seleccionar proveedor", unique_provs, index=0)
    with c_f_anio:
        sel_anio_top = st.selectbox("Seleccionar año", unique_years, index=len(unique_years)-1)
    with c_f_prob:
        sel_prob_clas = st.selectbox("Seleccionar problema", unique_problems, index=0)

    col_g1, col_g2 = st.columns(2)

    # Gráfica 1: Pastel - Inconformidades por tipo de servicio
    with col_g1:
        st.markdown(f"**Inconformidades por tipo de servicio ({sel_anio_top})**")
        
        # Filtros: proveedor + año
        df_pie = df[
            (df["nombre_comercial"] == sel_prov_top) &
            (df["año"] == sel_anio_top)
        ]
        
        if not df_pie.empty:
            pie_data = df_pie["tipo_servicio"].value_counts().reset_index()
            pie_data.columns = ["tipo_servicio", "conteo"]
            
            fig_pie = px.pie(
                pie_data, 
                names="tipo_servicio", 
                values="conteo",
                title=f"Distribución servicios - {sel_prov_top}",
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("No hay datos para la combinación Proveedor + Año seleccionada.")

    # Gráfica 2: Barras - Top motivos de reclamación
    with col_g2:
        st.markdown(f"**Top motivos: {sel_prob_clas}**")

        df_bar = df[
            (df["nombre_comercial"] == sel_prov_top) &
            (df["problema_clasificado"] == sel_prob_clas)
        ]
        
        if not df_bar.empty:
            # Top 10 motivos
            bar_data = df_bar["motivo_reclamacion"].value_counts().head(10).reset_index()
            bar_data.columns = ["motivo_reclamacion", "frecuencia"]
            
            # Ordenar para que la barra mayor salga arriba
            bar_data = bar_data.sort_values("frecuencia", ascending=True)

            fig_bar = px.bar(
                bar_data, 
                x="frecuencia", 
                y="motivo_reclamacion", 
                orientation='h',
                text_auto=True,
                title=f"Top 10 Motivos de reclamación",
                labels={"frecuencia": "Frecuencia", "motivo_reclamacion": ""}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No hay datos para la combinación Proveedor + Problema seleccionada.")

# TAB 2 — BLOQUE 4: ANÁLISIS ECONÓMICO E INFERENCIAL

with tab2:

    st.header("Análisis económico e inferencial")

    st.markdown(
        """
        Se evalúa la **existencia de relaciones lineales**
        entre variables económicas y operativas mediante:
        
        - Diagramas de dispersión
        - Correlación de Pearson
        - Prueba de hipótesis con t de Student (α = 0.05)
        """
    )

     #  Filtro por proveedor 
    proveedores = sorted(df["proveedor_top"].dropna().unique())
    proveedores_filtro = ["Todos"] + proveedores

    sel_proveedor_eco = st.selectbox(
        "Selecciona Top Proveedor",
        proveedores_filtro,
        index=0
    )


    # Preparar dataset económico
    eco_df = df[[
        "proveedor_top",
        "monto_reclamado",
        "monto_recuperado",
        "resuelta",
        "dias_resolucion"
    ]].dropna()

    if sel_proveedor_eco != "Todos":
        eco_df = eco_df[eco_df["proveedor_top"] == sel_proveedor_eco]


    eco_df["porcentaje_resolucion"] = eco_df["resuelta"] * 100

    def analizar_correlacion(x, y, x_label, y_label):
        r, p = pearsonr(x, y)

        fig = px.scatter(
            x=x,
            y=y,
            labels={"x": x_label, "y": y_label},
            title=f"{y_label} vs {x_label}"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        **Coeficiente de correlación (r):** {r:.3f}  
        **p-value:** {p:.4f}
        """)

        if p <= 0.05:
            st.success(
                "Con un nivel de significancia de 0.05, se **rechaza la hipótesis nula**. "
                "Existe evidencia estadística de **correlación lineal**."
            )
        else:
            st.warning(
                "Con un nivel de significancia de 0.05, **no se rechaza la hipótesis nula**. "
                "No se encontró evidencia suficiente de correlación lineal."
            )

        st.divider()

        
    sufijo_titulo = (
    f" — {sel_proveedor_eco}"
    if sel_proveedor_eco != "Todos"
    else " — Todos los proveedores"
    )

    # 1 Monto reclamado vs monto recuperado
    st.subheader("Monto reclamado vs monto recuperado" + sufijo_titulo)


    analizar_correlacion(
        eco_df["monto_reclamado"],
        eco_df["monto_recuperado"],
        "Monto reclamado",
        "Monto recuperado"
    )
    # 2 Porcentaje de resolución vs días de resolución
    st.subheader("Monto reclamado vs días de resolución" + sufijo_titulo)

    analizar_correlacion(
        eco_df["dias_resolucion"],
        eco_df["monto_reclamado"],
        "Días de resolución",
        "Monto reclamado"
    )

    # 3 Porcentaje de resolución vs días de resolución
    st.subheader("Monto recuperado vs días de resolución" + sufijo_titulo)

    analizar_correlacion(
        eco_df["dias_resolucion"],
        eco_df["monto_recuperado"],
        "Días de resolución",
        "Monto recuperado"
    )
    # a partir de aqui

    st.header("Análisis económico agregado por proveedor")

    usuarios_por_proveedor = {
        "TELCEL": 162420073,
        "MOVISTAR": 31934419,
        "MEGACABLE": 12009829,
        "IZZI": 15489758,
        "TOTALPLAY": 12241992,
        "AT&T": 42184165,
        "TELMEX": 19510076
    }

    top_prov = (
        df["proveedor_top"]
        .value_counts()
        .head(7)
        .index
    )

    df_eco = df[df["proveedor_top"].isin(top_prov)].copy()

    # CHECKBOXES DE NORMALIZACIÓN


    col_a, col_b = st.columns(2)

    with col_a:
        normalizar_montos = st.checkbox(
            "Normalizar montos por cada 100,000 usuarios"
        )

    with col_b:
        normalizar_casos = st.checkbox(
            "Normalizar número de casos por cada 100 quejas"
        )

    # AGREGACIÓN DE MONTOS
    eco_agregado = (
        df_eco.groupby("proveedor_top")
        .agg(
            monto_reclamado_total=("monto_reclamado", "sum"),
            monto_recuperado_total=("monto_recuperado", "sum")
        )
        .reset_index()
    )

    eco_agregado["usuarios"] = eco_agregado["proveedor_top"].map(
        usuarios_por_proveedor
    )

    eco_agregado = eco_agregado.dropna(subset=["usuarios"])

    if normalizar_montos:
        eco_agregado["monto_reclamado_total"] = (
            eco_agregado["monto_reclamado_total"]
            / eco_agregado["usuarios"] * 100000
        )
        eco_agregado["monto_recuperado_total"] = (
            eco_agregado["monto_recuperado_total"]
            / eco_agregado["usuarios"] * 100000
        )

    eco_long = eco_agregado.melt(
        id_vars="proveedor_top",
        value_vars=[
            "monto_reclamado_total",
            "monto_recuperado_total"
        ],
        var_name="tipo_monto",
        value_name="monto"
    )

    # GRÁFICA 1 — MONTOS

    st.subheader("Montos reclamados vs montos recuperados por proveedor")

    st.plotly_chart(
        px.bar(
            eco_long,
            x="proveedor_top",
            y="monto",
            color="tipo_monto",
            barmode="group",
            title="Comparación económica por proveedor"
        ),
        use_container_width=True
    )
    total_quejas_prov = (
        df_eco.groupby("proveedor_top")
        .size()
        .to_dict()
    )


    # CASO A: monto reclamado > 0 y recuperado = 0

    st.subheader("Casos con monto reclamado y recuperación igual a $0")
    
    df_no_recupera = df_eco[
        (df_eco["monto_reclamado"] > 0) &
        (df_eco["monto_recuperado"] == 0)
    ]
    
    casos_no_rec = (
        df_no_recupera.groupby("proveedor_top")
        .size()
        .reset_index(name="casos")
    )
    
    casos_no_rec["total_quejas"] = casos_no_rec["proveedor_top"].map(
        total_quejas_prov
    )
    
    if normalizar_casos:
        casos_no_rec["valor"] = (
            casos_no_rec["casos"]
            / casos_no_rec["total_quejas"] * 100
        )
        eje_x = "valor"
        titulo = "Porcentaje de quejas con monto reclamado > 0 y recuperación = 0"
    else:
        casos_no_rec["valor"] = casos_no_rec["casos"]
        eje_x = "valor"
        titulo = "Número de casos con monto reclamado > 0 y recuperación = 0"
    
    casos_no_rec = casos_no_rec.sort_values(eje_x, ascending=True)
    
    st.plotly_chart(
        px.bar(
            casos_no_rec,
            x=eje_x,
            y="proveedor_top",
            orientation="h",
            title=titulo
        ),
        use_container_width=True
    )


    # CASO B: monto reclamado = 0 y recuperado > 0

    st.subheader("Casos con recuperación positiva y monto reclamado igual a $0")
    
    df_no_reclamo = df_eco[
        (df_eco["monto_reclamado"] == 0) &
        (df_eco["monto_recuperado"] > 0)
    ]
    
    casos_no_reclamo = (
        df_no_reclamo.groupby("proveedor_top")
        .size()
        .reset_index(name="casos")
    )
    
    casos_no_reclamo["total_quejas"] = casos_no_reclamo["proveedor_top"].map(
        total_quejas_prov
    )
    
    if normalizar_casos:
        casos_no_reclamo["valor"] = (
            casos_no_reclamo["casos"]
            / casos_no_reclamo["total_quejas"] * 100
        )
        eje_x = "valor"
        titulo = "Porcentaje de quejas con monto reclamado = 0 y recuperación > 0"
    else:
        casos_no_reclamo["valor"] = casos_no_reclamo["casos"]
        eje_x = "valor"
        titulo = "Número de casos con monto reclamado = 0 y recuperación > 0"
    
    casos_no_reclamo = casos_no_reclamo.sort_values(eje_x, ascending=True)
    
    st.plotly_chart(
        px.bar(
            casos_no_reclamo,
            x=eje_x,
            y="proveedor_top",
            orientation="h",
            title=titulo
        ),
        use_container_width=True
    )









    


# TAB 3 — BLOQUE 5: ANÁLISIS MULTIVARIADO


with tab3:

    st.header("Análisis multivariado de proveedores")

    st.markdown(
        """
        En esta sección se utilizan representaciones multivariadas para comparar
        el comportamiento agregado de los principales proveedores de telecomunicaciones.
        Las técnicas empleadas tienen un propósito exploratorio y descriptivo.
        """
    )

    # 1. PERFIL DE PROVEEDORES
    st.subheader("Perfil agregado de proveedores")
    st.markdown(
        "Cada proveedor se representa como un vector numérico que resume su comportamiento promedio."
    )
    st.dataframe(perfil_df)

    # 2. MATRIZ DE DISTANCIAS
    st.subheader("Matriz de distancias euclidianas")
    st.markdown(
        "Valores pequeños indican proveedores con perfiles similares; valores grandes indican mayor disimilitud."
    )

    st.plotly_chart(
        px.imshow(
            dist_df,
            text_auto=".2f",
            title="Distancia euclidiana entre proveedores"
        ),
        use_container_width=True
    )

    # 3. PCA / MDS
    st.subheader("Proyección en dos dimensiones")

    metodo = st.radio(
        "Selecciona método de proyección",
        ["PCA", "MDS"]
    )

    if metodo == "PCA":
        st.markdown(
            "PCA preserva la mayor varianza posible del conjunto de variables originales."
        )
        fig_proj = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            text=pca_df.index,
            title="Proyección PCA de proveedores"
        )
    else:
        st.markdown(
            "MDS preserva las distancias originales entre proveedores."
        )
        fig_proj = px.scatter(
            mds_df,
            x="Dim1",
            y="Dim2",
            text=mds_df.index,
            title="Proyección MDS basada en distancias"
        )

    st.plotly_chart(fig_proj, use_container_width=True)

    # 4. CORRELACIÓN
    st.subheader("Matriz de correlación (Pearson)")
    st.markdown(
        "La correlación de Pearson evalúa relaciones lineales entre variables continuas."
    )

    st.plotly_chart(
        px.imshow(
            corr_df,
            text_auto=".2f",
            title="Correlación de Pearson entre variables"
        ),
        use_container_width=True
    )























