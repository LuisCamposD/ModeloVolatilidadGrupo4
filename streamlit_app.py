import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer


st.set_page_config(
    page_title="Volatilidad del Tipo de Cambio",
    layout="wide"
)

# ---------- 1. Cargar modelo, imputer, variables y datos ----------
@st.cache_resource
def cargar_recursos():
    # Cargar modelo entrenado y variables seleccionadas
    modelo = joblib.load("gbr_mejor_modelo_tc.pkl")
    selected_vars = joblib.load("selected_vars_volatilidad.pkl")

    # Cargar datos
    df = pd.read_csv("datos_tc_limpios.csv")

    # Crear columna fecha si no existe
    if "fecha" not in df.columns:
        if "anio" in df.columns and "mes" in df.columns:
            mapa_meses = {
                "Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4,
                "May": 5, "Jun": 6, "Jul": 7, "Ago": 8,
                "Set": 9, "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12
            }
            df_mes_num = df["mes"].map(mapa_meses)

            df["fecha"] = pd.to_datetime(
                dict(year=df["anio"], month=df_mes_num, day=1)
            )
        else:
            df["fecha"] = pd.date_range(start="2000-01-01", periods=len(df), freq="M")

    # Crear rendimiento logarítmico para análisis y métricas
    df_mod = df.copy()
    df_mod["Rendimientos_log"] = np.log(df_mod["TC"] / df_mod["TC"].shift(1))
    df_mod = df_mod.dropna(subset=["Rendimientos_log"])

    # === Aquí re-entrenamos el imputador con los mismos datos que en Colab ===
    X = df_mod[selected_vars]
    train_size = int(len(X) * 0.8)
    X_train = X.iloc[:train_size]

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)

    return modelo, imputer, selected_vars, df, df_mod


modelo, imputer, selected_vars, df, df_mod = cargar_recursos()

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

    st.markdown("---")
    st.subheader("Línea de tiempo del tipo de cambio")

    df_tc = df.sort_values("fecha").copy()

    col1, col2 = st.columns([3, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_tc["fecha"], df_tc["TC"], marker="o", linewidth=1)
        ax.set_title("Evolución del Tipo de Cambio de Venta")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("TC (S/.)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.write("**Resumen rápido:**")
        st.write(f"- Observaciones: {len(df_tc)}")
        st.write(f"- TC mínimo: {df_tc['TC'].min():.4f}")
        st.write(f"- TC máximo: {df_tc['TC'].max():.4f}")
        st.write(f"- TC promedio: {df_tc['TC'].mean():.4f}")

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
    st.subheader("Distribución del tipo de cambio (TC)")

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.boxplot(x=df["TC"], ax=ax)
    ax.set_title("Boxplot del Tipo de Cambio de Venta (TC)")
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

    st.subheader("Performance del modelo en el conjunto de prueba")

    # Reconstruir X/y igual que en el Colab
    X = df_mod[selected_vars]
    y = df_mod["Rendimientos_log"]

    train_size = int(len(X) * 0.8)
    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    X_test_imp = imputer.transform(X_test)

    y_pred = modelo.predict(X_test_imp)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("R2", f"{r2:.4f}")
        st.metric("MAE", f"{mae:.6f}")
        st.metric("RMSE", f"{rmse:.6f}")

    with col2:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(y_test.values, label="Real", alpha=0.8)
        ax.plot(y_pred, label="Predicho", alpha=0.8)
        ax.set_title("Rendimientos logarítmicos: real vs predicho")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Simulador de predicciones")

    st.write("""
    Ajusta las variables explicativas y el modelo estimará el **rendimiento logarítmico** del TC.
    Luego proyectamos un TC esperado a partir del último valor observado.
    """)

    desc = df_mod[selected_vars].describe()

    with st.form("form_pred"):
        entrada = {}
        for col in selected_vars:
            min_val = float(desc.loc["min", col])
            max_val = float(desc.loc["max", col])
            mean_val = float(desc.loc["mean", col])

            entrada[col] = st.slider(
                col,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )

        enviado = st.form_submit_button("Calcular predicción")

    if enviado:
        x_new = pd.DataFrame([entrada])
        x_new_imp = imputer.transform(x_new)
        rend_log_pred = modelo.predict(x_new_imp)[0]

        st.write(f"**Rendimiento logarítmico predicho:** {rend_log_pred:.6f}")

        ultimo_tc = df_mod["TC"].iloc[-1]
        tc_esperado = ultimo_tc * np.exp(rend_log_pred)

        st.write(f"Último tipo de cambio observado: **{ultimo_tc:.4f}**")
        st.write(f"Tipo de cambio estimado para el siguiente período: **{tc_esperado:.4f}**")

        st.success("Predicción calculada con éxito.")
