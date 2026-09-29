"""
Interfaz web sencilla de la herramienta de predicción de inasistencia.

Ejecutar desde la carpeta del proyecto:
    streamlit run app_streamlit.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import funciones_inasistencia as fi
import prediccion_diaria as pdiaria

st.set_page_config(page_title="Riesgo de inasistencia", layout="wide")


@st.cache_resource
def cargar_modelo():
    return joblib.load(pdiaria.RUTA_MODELO)


@st.cache_resource
def cargar_fuentes():
    return fi.cargar_fuentes(".")


paquete = cargar_modelo()
st.title("Citas de mañana priorizadas por riesgo de inasistencia")
st.caption(f"Modelo {paquete['nombre_modelo']} · versión {paquete['version']} · entrenado hasta {paquete['entrenado_hasta']}")

fecha = st.date_input("Fecha de las citas", value=pd.Timestamp.today().normalize() + pd.Timedelta(days=1))
if st.button("Generar lista", type="primary"):
    with st.spinner("Calculando variables y probabilidades…"):
        # Se guarda en session_state para que la tabla no desaparezca al mover los filtros
        st.session_state["resultado"] = (fecha, *pdiaria.predecir_citas(fecha, ".", paquete, cargar_fuentes()))

if "resultado" in st.session_state:
    fecha, evaluadas, no_evaluadas = st.session_state["resultado"]
    st.subheader(f"Citas del {fecha}")
    if evaluadas.empty:
        st.warning("No hay citas de consulta agendadas para esa fecha.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Citas evaluadas", f"{len(evaluadas):,}")
        col2.metric("Riesgo alto (top 10 %)", int((evaluadas["nivel_riesgo"] == "Alto").sum()))
        col3.metric("Inasistencias esperadas", f"{evaluadas['probabilidad_inasistencia'].sum():.0f}")
        col4.metric("No evaluadas", len(no_evaluadas))

        niveles = st.multiselect("Nivel de riesgo", ["Alto", "Medio", "Bajo"], default=["Alto"])
        sedes = st.multiselect("Sede", sorted(evaluadas["sede"].unique()), default=sorted(evaluadas["sede"].unique()))
        vista = evaluadas[evaluadas["nivel_riesgo"].isin(niveles) & evaluadas["sede"].isin(sedes)].copy()
        vista["probabilidad_inasistencia"] = vista["probabilidad_inasistencia"] * 100
        st.dataframe(
            vista[["prioridad", "nivel_riesgo", "probabilidad_inasistencia", "id_paciente", "hora_cita",
                   "especialidad", "sede", "edad", "motivos"]],
            column_config={"probabilidad_inasistencia": st.column_config.ProgressColumn(
                "Probabilidad", format="%.0f%%", min_value=0, max_value=100)},
            hide_index=True, use_container_width=True,
        )

        ruta = pdiaria.guardar_excel(evaluadas, no_evaluadas, fecha, "salidas")
        st.download_button("Descargar Excel", data=Path(ruta).read_bytes(), file_name=Path(ruta).name,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
