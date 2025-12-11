import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set(style="whitegrid")

st.set_page_config(
    page_title="Volatilidad del Tipo de Cambio",
    layout="wide"
)

# --------------------------------------------------------------------
# CONSTANTES: TIMELINE + MAPA DE MESES
# --------------------------------------------------------------------
IMAGES = [
    "https://raw.githubusercontent.com/LuisCamposD/timeline_s1/main/timeline_images/img1.jpg",
    "https://raw.githubusercontent.com/LuisCamposD/timeline_s1/main/timeline_images/img2.PNG",
    "https://raw.githubusercontent.com/LuisCamposD/timeline_s1/main/timeline_images/img3.jpg",
    "https://raw.githubusercontent.com/LuisCamposD/timeline_s1/main/timeline_images/img4.png",
    "https://raw.githubusercontent.com/LuisCamposD/timeline_s1/main/timeline_images/img5.png",
]

CAPTIONS = [
    "Años 80–90: enfoque básico",
    "Años 2000: apertura comercial y mayor exposición al dólar",
    "2008–2012: crisis y gestión del riesgo",
    "2013–2019: digitalización, BI y monitoreo diario del tipo de cambio",
    "2020 en adelante: disrupciones globales, analítica avanzada e IA",
]

TIMELINE = [
    {
        "titulo": "1️⃣ Años 80–90: tipo de cambio y compras casi desconectados",
        "resumen": (
            "En esta etapa el análisis de la volatilidad era mínimo. "
            "El tipo de cambio se veía como un dato macro, no como un insumo clave "
            "para las decisiones de logística."
        ),
        "bullets": [
            "Planeación de compras principalmente basada en experiencia y listas de precios históricas.",
            "Poca apertura comercial: menor participación de proveedores internacionales.",
            "El tipo de cambio se revisaba esporádicamente, no todos los días.",
            "No existían políticas claras sobre quién asumía el riesgo cambiario (proveedor vs empresa).",
        ],
    },
    {
        "titulo": "2️⃣ Años 2000: apertura comercial y mayor exposición al dólar",
        "resumen": (
            "Con la globalización y el aumento de importaciones, el tipo de cambio empieza "
            "a impactar directamente los costos logísticos."
        ),
        "bullets": [
            "Más compras en dólares (equipos, repuestos, tecnología, mobiliario importado).",
            "Compras empieza a comparar cotizaciones en distintas monedas, pero el análisis es manual (Excel básico).",
            "Se empiezan a usar tipos de cambio referenciales para presupuestos, pero sin escenarios de volatilidad.",
            "Mayor sensibilidad en los márgenes: variaciones de centavos ya impactan el costo total de los proyectos.",
        ],
    },
    {
        "titulo": "3️⃣ 2008–2012: crisis financiera y prioridad al riesgo cambiario",
        "resumen": (
            "La crisis global y los saltos bruscos del tipo de cambio obligan a formalizar "
            "la gestión del riesgo cambiario en compras y contratos."
        ),
        "bullets": [
            "Logística y Finanzas comienzan a trabajar juntos para definir TC de referencia y bandas de variación.",
            "Aparecen cláusulas específicas: ajuste de precio por tipo de cambio, vigencia corta de cotizaciones.",
            "Se analizan escenarios básicos: ¿qué pasa si el dólar sube 5%, 10% durante el proyecto?",
            "Compras prioriza cerrar rápidamente órdenes de compra críticas para evitar descalce entre aprobación y pago.",
        ],
    },
    {
        "titulo": "4️⃣ 2013–2019: digitalización, BI y monitoreo diario del tipo de cambio",
        "resumen": (
            "Las empresas adoptan ERPs, dashboards y reportes automáticos. "
            "El tipo de cambio se vuelve un indicador operativo para logística."
        ),
        "bullets": [
            "Dashboards de compras que muestran el impacto del tipo de cambio en el presupuesto y en el costo por contrato.",
            "Actualización diaria del tipo de cambio en sistemas (ERP) y en las plantillas de cuadros comparativos.",
            "Uso de modelos estadísticos simples para proyectar TC anual y armar presupuestos más realistas.",
            "Compras empieza a definir estrategias: adelantar o postergar compras según tendencias de tipo de cambio.",
        ],
    },
    {
        "titulo": "5️⃣ 2020 en adelante: disrupciones globales, analítica avanzada e IA",
        "resumen": (
            "Con la pandemia y los choques globales, la volatilidad del tipo de cambio se combina con "
            "rupturas de cadena de suministro. Compras necesita decisiones más inteligentes y rápidas."
        ),
        "bullets": [
            "Uso de analítica avanzada e IA para simular escenarios de tipo de cambio y su efecto en costos logísticos.",
            "Modelos que recomiendan: comprar ahora vs esperar, cambiar de proveedor, negociar en otra moneda o ajustar incoterms.",
            "Integración de datos de mercado (TC, commodities, fletes internacionales) con datos internos de consumo y stock.",
            "El rol de Compras/Logística evoluciona: de ejecutor de órdenes a gestor estratégico del riesgo cambiario y de suministro.",
        ],
    },
]

MAPA_MESES = {
    "Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Ago": 8, "Set": 9, "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12,
}

# ---------- 1. Cargar modelo, imputer, scaler, variables y datos ----------
@st.cache_resource
def cargar_recursos():
    # 1) Cargamos todo lo entrenado en Colab
    modelo = joblib.load("gbr_mejor_modelo_tc.pkl")
    selected_vars = joblib.load("selected_vars_volatilidad.pkl")
    imputer = joblib.load("imputer_volatilidad.pkl")
    scaler = joblib.load("scaler_volatilidad.pkl")

    # CSV real del repo
    df = pd.read_csv("datos_tc_limpios.csv")

    # ========== 1) Detectar columna de tipo de cambio (tc_col) ==========
    posibles_tc = [
        "TC",
        "tc",
        "TC_venta",
        "tc_venta",
        "Tipo de cambio - TC Sistema bancario SBS (S/ por US$) - Venta",
        "Tipo de cambio - TC Sistema bancario SBS (S/ por US$) - Venta ",
        "Tipo_de_cambio",
    ]

    tc_col = None
    for col in posibles_tc:
        if col in df.columns:
            tc_col = col
            break

    if tc_col is None:
        for col in df.columns:
            nombre = col.lower()
            if "tipo de cambio" in nombre or nombre == "tc":
                tc_col = col
                break

    if tc_col is None:
        raise KeyError(
            f"No se encontró columna de Tipo de Cambio en el CSV. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # ========== 2) Fecha y ordenamiento ==========
    if "fecha" not in df.columns:
        if "anio" in df.columns and "mes" in df.columns:
            df["mes_num"] = df["mes"].map(MAPA_MESES)
            df["fecha"] = pd.to_datetime(
                dict(year=df["anio"], month=df["mes_num"], day=1)
            )
        else:
            df["fecha"] = pd.date_range(start="2000-01-01", periods=len(df), freq="M")
    else:
        df["fecha"] = pd.to_datetime(df["fecha"])

    if "mes_num" not in df.columns and "mes" in df.columns:
        df["mes_num"] = df["mes"].map(MAPA_MESES)

    df = df.sort_values("fecha").reset_index(drop=True)

    # ========== 3) Rendimientos logarítmicos ==========
    df_mod = df.copy()
    df_mod["Rendimientos_log"] = np.log(df_mod[tc_col] / df_mod[tc_col].shift(1))
    df_mod = df_mod.dropna(subset=["Rendimientos_log"])

    return modelo, imputer, scaler, selected_vars, df, df_mod, tc_col


st.write("🔄 Inicializando app y cargando recursos...")

try:
    modelo, imputer, scaler, selected_vars, df, df_mod, tc_col = cargar_recursos()
except Exception as e:
    st.error("❌ Error cargando los recursos (modelo, datos o transformaciones).")
    st.exception(e)
    st.stop()

# ---------- 2. Sidebar: navegación ----------
st.sidebar.title("Menú")
pagina = st.sidebar.radio(
    "Ir a:",
    ["Inicio y línea de tiempo", "EDA", "Modelo y predicciones"]
)

# ---------- 3. Página: Inicio y línea de tiempo ----------
if pagina == "Inicio y línea de tiempo":
    st.title("Volatilidad del Tipo de Cambio de Venta (TC)")
    st.subheader("Introducción")

    st.write("""
    En este proyecto analizamos **la volatilidad del tipo de cambio de venta (TC)**,
    construyendo un modelo que predice los **rendimientos logarítmicos** del TC a partir de
    variables macroeconómicas (precios de metales, PBI, reservas, intervenciones del BCRP, etc.).

    Trabajamos con datos mensuales y respetamos la estructura temporal de la serie
    (entrenamos con los primeros períodos y probamos con los últimos).
    """)

    st.subheader("Problemática")
    st.write("""
    Para áreas de **logística, finanzas y planificación**, la volatilidad del tipo de cambio es clave:
    impacta directamente en el costo de importaciones, contratos en dólares y cobertura de riesgos.

    El objetivo es:
    - **Cuantificar** cómo se mueve el TC de un mes a otro (rendimientos logarítmicos).
    - **Identificar variables explicativas** relevantes mediante selección por **Forward**.
    - **Construir un modelo** (Gradient Boosting Regressor) que permita **simular escenarios**
      y anticipar movimientos del tipo de cambio.
    """)

    # Timeline
    st.markdown("---")
    st.subheader("Timeline: Evolución del análisis de la volatilidad del tipo de cambio")

    st.write(
        "Mueve el slider para ver cómo, a lo largo de los años, ha evolucionado el análisis "
        "de la volatilidad del tipo de cambio y su impacto en el área de Compras y Logística."
    )

    step = st.slider(
        "Selecciona la etapa del timeline:",
        min_value=1,
        max_value=len(TIMELINE),
        value=1,
        step=1,
        key="timeline_slider",
    )

    idx = step - 1
    item = TIMELINE[idx]

    st.subheader(item["titulo"])

    st.image(
        IMAGES[idx],
        caption=CAPTIONS[idx],
        use_container_width=True,
    )

    st.markdown(f"**Resumen:** {item['resumen']}")

    st.markdown("**¿Qué pasa en esta etapa?**")
    for bullet in item["bullets"]:
        st.markdown(f"- {bullet}")

    if len(TIMELINE) > 1:
        st.progress(idx / (len(TIMELINE) - 1))
    else:
        st.progress(1.0)

    # Histórico TC
    st.markdown("---")
    st.subheader("Histórico del tipo de cambio")

    df_tc = df.sort_values("fecha").copy()

    col1, col2 = st.columns([3, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_tc["fecha"], df_tc[tc_col], marker="o", linewidth=1)
        ax.set_title("Evolución del Tipo de Cambio de Venta")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("TC (S/.)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.write("**Resumen rápido:**")
        st.write(f"- Observaciones: {len(df_tc)}")
        st.write(f"- TC mínimo: {df_tc[tc_col].min():.4f}")
        st.write(f"- TC máximo: {df_tc[tc_col].max():.4f}")
        st.write(f"- TC promedio: {df_tc[tc_col].mean():.4f}")

        st.info("""
        La línea de tiempo nos permite ubicar:
        - Periodos de mayor estabilidad.
        - Picos de volatilidad que pueden asociarse a shocks externos o internos.
        """)

# ---------- 4. Página: EDA ----------
elif pagina == "EDA":
    st.title("Análisis Exploratorio de Datos (EDA)")

    st.subheader("Vista general del dataset")
    st.write(f"**Filas:** {df.shape[0]}  |  **Columnas:** {df.shape[1]}")
    st.dataframe(df.head())

    st.subheader("Tipos de datos")
    st.write(df.dtypes)

    st.markdown("---")
    st.subheader("Valores faltantes")

    missing = df.isna().sum()
    st.write(missing)

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(df.isnull(), cbar=False, ax=ax)
    ax.set_title("Mapa de valores faltantes")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Distribución del tipo de cambio")

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.boxplot(x=df[tc_col], ax=ax)
    ax.set_title("Boxplot del Tipo de Cambio de Venta")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Matriz de correlaciones")

    numeric_cols = df.select_dtypes(include=["number"])
    corr = numeric_cols.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Matriz de correlación (variables numéricas)")
    st.pyplot(fig)

    st.info("""
    El EDA nos ayuda a:
    - Ver estructura de los datos (tipos, nulos, outliers).
    - Identificar posibles relaciones entre el TC y variables explicativas.
    """)

# ---------- 5. Página: Modelo y predicciones ----------
elif pagina == "Modelo y predicciones":
    st.title("Modelo de Volatilidad y Predicciones")

    # 5.1 Performance del modelo
    st.subheader("Performance del modelo")

    X = df_mod[selected_vars]
    y = df_mod["Rendimientos_log"]

    train_size = int(len(X) * 0.8)
    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    # ---- Métricas en TEST (out-of-sample) ----
    X_test_imp = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imp)
    y_pred_test = modelo.predict(X_test_scaled)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    # 🔹 Sesgo promedio del modelo en test
    bias = (y_pred_test - y_test).mean()

    # ---- Métricas en TODO el histórico (in-sample) ----
    X_all_imp = imputer.transform(X)
    X_all_scaled = scaler.transform(X_all_imp)
    y_pred_all = modelo.predict(X_all_scaled)

    mae_all = mean_absolute_error(y, y_pred_all)
    rmse_all = np.sqrt(mean_squared_error(y, y_pred_all))
    r2_all = r2_score(y, y_pred_all)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Conjunto de prueba (20% final de la serie)**")
        st.metric("R2 (test)", f"{r2_test:.4f}")
        st.metric("MAE (test)", f"{mae_test:.6f}")
        st.metric("RMSE (test)", f"{rmse_test:.6f}")

    with col2:
        st.markdown("**Todo el histórico (train + test)**")
        st.metric("R2 (in-sample)", f"{r2_all:.4f}")
        st.metric("MAE (in-sample)", f"{mae_all:.6f}")
        st.metric("RMSE (in-sample)", f"{rmse_all:.6f}")

    # Gráfico test
    st.markdown("### Rendimientos logarítmicos en el conjunto de prueba")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(y_test.values, label="Real", alpha=0.8)
    ax.plot(y_pred_test, label="Predicho", alpha=0.8)
    ax.set_title("Rendimientos logarítmicos: real vs predicho (test)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # 5.2 Predicción futura
    st.markdown("---")
    st.subheader("Predicción de varios meses hacia adelante")

    st.write("""
    Selecciona el *año y el mes de inicio* para proyectar el tipo de cambio varios meses hacia adelante.
    El modelo usará el último registro de datos como base y calculará un **escenario acotado** del TC esperado.
    """)

    df_ordenado = df.sort_values("fecha").reset_index(drop=True)

    if "mes" in df_ordenado.columns:
        meses_nombres = sorted(
            list({m for m in df_ordenado["mes"].unique()} | set(MAPA_MESES.keys())),
            key=lambda m: MAPA_MESES.get(m, 13),
        )
        ultimo_mes_nombre = df_ordenado["mes"].iloc[-1]
    else:
        meses_nombres = list(MAPA_MESES.keys())
        ultimo_mes_nombre = "Dic"

    ultimo_anio = int(df_ordenado["anio"].iloc[-1])
    idx_mes_default = meses_nombres.index(ultimo_mes_nombre)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        anio_input = st.number_input(
            "Año de inicio de la predicción",
            min_value=ultimo_anio,
            max_value=ultimo_anio + 10,
            value=ultimo_anio,
            step=1,
        )
    with col_b:
        mes_nombre = st.selectbox(
            "Mes de inicio",
            options=meses_nombres,
            index=idx_mes_default,
        )
        mes_inicio = MAPA_MESES[mes_nombre]
    with col_c:
        num_meses = st.slider("Número de meses a predecir", 1, 24, 5)

    if st.button("Calcular predicción"):
        ultimo_X = df_mod[selected_vars].iloc[-1].copy()
        ultimo_tc = df_ordenado[tc_col].iloc[-1]

        meses_futuro = []
        mes_actual = mes_inicio
        anio_actual = int(anio_input)

        for _ in range(num_meses):
            meses_futuro.append((anio_actual, mes_actual))
            mes_actual += 1
            if mes_actual > 12:
                mes_actual = 1
                anio_actual += 1

        df_futuro = pd.DataFrame(meses_futuro, columns=["anio", "mes_num"])

        for col in selected_vars:
            if col == "anio":
                df_futuro[col] = df_futuro["anio"]
            else:
                df_futuro[col] = ultimo_X[col]

        # Imputar + escalar
        X_fut_imp = imputer.transform(df_futuro[selected_vars])
        X_fut_scaled = scaler.transform(X_fut_imp)

        # Predicción cruda
        rendimientos_raw = modelo.predict(X_fut_scaled)

        # 🔹 1) Corregimos sesgo usando el desempeño en test
        rendimientos_corr = rendimientos_raw - bias

        # 🔹 2) Limitamos rendimientos mensuales a un rango razonable ±5%
        rendimientos_pred = np.clip(rendimientos_corr, -0.05, 0.05)

        # Reconstrucción del tipo de cambio
        tc_pred = []
        tc_actual = ultimo_tc
        for r in rendimientos_pred:
            tc_actual = tc_actual * np.exp(r)
            tc_pred.append(tc_actual)

        df_futuro["TC_predicho"] = tc_pred

        mes_dict_inv = {v: k for k, v in MAPA_MESES.items()}
        df_futuro["mes"] = df_futuro["mes_num"].map(mes_dict_inv)

        st.write("### Escenario de predicción futura")
        st.dataframe(df_futuro[["anio", "mes", "TC_predicho"]])

        fig, ax = plt.subplots(figsize=(10, 4))
        x_hist = np.arange(len(df_ordenado))
        ax.plot(x_hist, df_ordenado[tc_col], label="TC real (histórico)")

        x_fut = np.arange(len(df_ordenado), len(df_ordenado) + num_meses)
        ax.plot(
            x_fut,
            df_futuro["TC_predicho"],
            label=f"TC predicho ({num_meses} meses desde {mes_nombre}/{int(anio_input)})",
            marker="o",
        )

        ax.set_title(
            f"Predicción del Tipo de Cambio - {num_meses} meses desde {mes_nombre}/{int(anio_input)}"
        )
        ax.set_xlabel("Meses")
        ax.set_ylabel("Tipo de cambio (S/ por US$)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
