"""
Herramienta diaria de predicción de inasistencia a citas.

Uso (se corre el día ANTERIOR a las citas):
    python prediccion_diaria.py --fecha 2026-08-27
    python prediccion_diaria.py --fecha 2026-08-27 --evaluar     # (demo) compara con lo que pasó realmente

Qué hace:
1. Carga el modelo entrenado (modelos/modelo_inasistencia.joblib).
2. Carga los archivos fuente y se queda SOLO con lo que se conocía el día anterior:
   - agenda: citas asignadas hasta D-1
   - RIPS y medicamentos: registros hasta D-2 (lo de D-1 aún no está completo)
3. Etiqueta el historial y calcula las variables con las MISMAS funciones del entrenamiento
   (funciones_inasistencia.py).
4. Predice la probabilidad de inasistencia de cada cita de mañana.
5. Guarda un Excel priorizado en salidas/prediccion_AAAA-MM-DD.xlsx con nivel de riesgo y motivos.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import funciones_inasistencia as fi

RUTA_MODELO = Path("modelos/modelo_inasistencia.joblib")


# ---------------------------------------------------------------------------
# Explicación sencilla para el equipo que hace las llamadas
# ---------------------------------------------------------------------------
def motivos_riesgo(fila):
    """Reglas simples (basadas en el EDA) para explicar por qué una cita tiene riesgo."""
    motivos = []
    if fila["estado_ultima_cita"] == "No asistió":
        motivos.append("Faltó a su última cita")
    if fila["n_citas_previas"] >= 3 and fila["tasa_inasistencia_suavizada"] >= 0.45:
        motivos.append(f"Falta con frecuencia ({fila['n_inasistidas_previas']} de {fila['n_citas_previas']})")
    if fila.get("estado_ultima_cita_esp") == "No asistió":
        motivos.append("Faltó a la última cita de esta especialidad")
    if 6 <= fila["edad"] <= 28:
        motivos.append("Niño, adolescente o joven (grupos con más inasistencia)")
    if fila["programa_rcv"] == "Sin programa" and fila["edad"] >= 29:
        motivos.append("No pertenece a un programa de crónicos")
    if fila["lead_time"] > 30:
        motivos.append(f"Cita asignada hace {fila['lead_time']} días")
    if fila["familia_servicio"] in ("Salud mental", "Nutrición", "Enfermería / PyP"):
        motivos.append(f"Servicio con alta inasistencia ({fila['familia_servicio']})")
    if fila["sin_historial"] == 1:
        motivos.append("Sin historial previo (riesgo estimado por perfil)")
    return "; ".join(motivos[:3]) if motivos else "Combinación de factores del perfil"


def nivel_riesgo(rango_en_el_dia):
    if rango_en_el_dia <= 0.10:
        return "Alto"
    elif rango_en_el_dia <= 0.30:
        return "Medio"
    return "Bajo"


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
def predecir_citas(fecha, carpeta=".", paquete=None, fuentes=None):
    """Devuelve (evaluadas, no_evaluadas) para las citas de la fecha indicada."""
    fecha = pd.Timestamp(fecha).normalize()
    dia_anterior = fecha - pd.Timedelta(days=1)          # D-1: día en que se corre la herramienta
    paquete = paquete or joblib.load(Path(carpeta) / RUTA_MODELO)
    fuentes = fuentes or fi.cargar_fuentes(carpeta)
    params = paquete["parametros"]

    # 1. Solo lo que se conocía en D-1
    agenda_cruda = fuentes["agendamiento"]
    agenda_cruda = agenda_cruda[pd.to_datetime(agenda_cruda["fecha_asig"]).dt.normalize() <= dia_anterior]
    agenda = fi.limpiar_agendamiento(agenda_cruda)

    rips = fi.preparar_rips(fuentes["rips"])
    rips = rips[rips["fecha_atencion"] < dia_anterior]
    fuentes_conocidas = dict(fuentes)
    meds = fuentes["medicamentos"]
    fuentes_conocidas["medicamentos"] = meds[pd.to_datetime(meds["fec_orden"]).dt.normalize() < dia_anterior]

    # 2. Historial etiquetado hasta D-2, con las reglas aprendidas en el entrenamiento
    mapa_servicio = pd.Series(paquete["mapa_servicio"], name="servicio")
    historia, todas, _, _ = fi.construir_citas_etiquetadas(
        agenda, rips, params, mapa_servicio=mapa_servicio,
        excluidas=paquete["especialidades_excluidas"], fecha_fin=fecha - pd.Timedelta(days=2),
        descontinuados=paquete["codigos_descontinuados"],
    )

    # 3. Citas de mañana
    manana = agenda[agenda["fecha_cita"] == fecha].copy()
    manana["cita_id"] = np.arange(len(manana)) + 10_000_000
    motivo_exclusion = np.select(
        [manana["tipo_agenda"] == "Procedimiento",
         manana["especialidad"].isin(paquete["especialidades_excluidas"]),
         manana["especialidad"].isin(paquete["codigos_descontinuados"])],
        ["Procedimiento (no se reporta en RIPS de consultas)",
         "Servicio no evaluado por el modelo (terapias / sin registro)",
         "Código de agenda descontinuado: verificar si la cita sigue vigente"],
        default="",
    )
    no_evaluadas = manana[motivo_exclusion != ""].assign(motivo=motivo_exclusion[motivo_exclusion != ""])
    objetivo = manana[motivo_exclusion == ""].copy()
    if objetivo.empty:
        return objetivo, no_evaluadas

    # 4. Variables (mismas funciones que en el entrenamiento) y predicción
    objetivo = fi.calcular_variables(objetivo, historia, todas, agenda, fuentes_conocidas, rips, params)
    probabilidad = paquete["modelo"].predict_proba(objetivo[paquete["variables"]])[:, 1]
    objetivo["probabilidad_inasistencia"] = probabilidad
    objetivo["rango_en_el_dia"] = objetivo["probabilidad_inasistencia"].rank(ascending=False, pct=True, method="first")
    objetivo["nivel_riesgo"] = objetivo["rango_en_el_dia"].apply(nivel_riesgo)
    objetivo["motivos"] = objetivo.apply(motivos_riesgo, axis=1)
    objetivo = objetivo.sort_values("probabilidad_inasistencia", ascending=False)
    objetivo["prioridad"] = np.arange(1, len(objetivo) + 1)
    return objetivo, no_evaluadas


def evaluar_contra_realidad(evaluadas, fuentes, paquete):
    """(Solo para demostración con fechas pasadas) compara la predicción con la asistencia real."""
    rips = fi.preparar_rips(fuentes["rips"])
    mapa_servicio = pd.Series(paquete["mapa_servicio"], name="servicio")
    real = fi.marcar_asistencia(evaluadas.drop(columns=["servicio"], errors="ignore"), rips, mapa_servicio)
    real["no_asiste"] = 1 - real["asiste"]
    resumen = real.groupby("nivel_riesgo")["no_asiste"].agg(citas="size", inasistencia_real="mean")
    return real, resumen.reindex(["Alto", "Medio", "Bajo"])


def guardar_excel(evaluadas, no_evaluadas, fecha, carpeta_salida="salidas"):
    Path(carpeta_salida).mkdir(exist_ok=True)
    ruta = Path(carpeta_salida) / f"prediccion_{pd.Timestamp(fecha).date()}.xlsx"
    columnas = {
        "prioridad": "Prioridad", "nivel_riesgo": "Nivel de riesgo", "probabilidad_inasistencia": "Probabilidad de inasistencia",
        "id_paciente": "ID paciente", "hora_cita": "Hora", "especialidad": "Especialidad", "sede": "Sede",
        "edad": "Edad", "sexo": "Sexo", "lead_time": "Días desde la asignación", "motivos": "Motivos principales",
    }
    tabla = evaluadas[list(columnas)].rename(columns=columnas)
    tabla["Probabilidad de inasistencia"] = tabla["Probabilidad de inasistencia"].round(3)
    resumen = pd.DataFrame({
        "Indicador": ["Fecha de las citas", "Citas evaluadas", "Riesgo alto (top 10 %)", "Riesgo medio (10-30 %)",
                      "Inasistencias esperadas", "Citas no evaluadas"],
        "Valor": [str(pd.Timestamp(fecha).date()), len(evaluadas), int((evaluadas["nivel_riesgo"] == "Alto").sum()),
                  int((evaluadas["nivel_riesgo"] == "Medio").sum()),
                  round(float(evaluadas["probabilidad_inasistencia"].sum())), len(no_evaluadas)],
    })
    with pd.ExcelWriter(ruta) as escritor:
        resumen.to_excel(escritor, sheet_name="Resumen", index=False)
        tabla.to_excel(escritor, sheet_name="Prioridad de llamadas", index=False)
        no_evaluadas[["id_paciente", "hora_cita", "especialidad", "sede", "motivo"]].to_excel(
            escritor, sheet_name="No evaluadas", index=False)
    return ruta


def main():
    parser = argparse.ArgumentParser(description="Lista priorizada de citas con riesgo de inasistencia")
    parser.add_argument("--fecha", required=True, help="Fecha de las citas a predecir (AAAA-MM-DD), normalmente mañana")
    parser.add_argument("--carpeta", default=".", help="Carpeta con los archivos fuente")
    parser.add_argument("--evaluar", action="store_true", help="(Demo) comparar con la asistencia real")
    args = parser.parse_args()

    paquete = joblib.load(Path(args.carpeta) / RUTA_MODELO)
    fuentes = fi.cargar_fuentes(args.carpeta)
    evaluadas, no_evaluadas = predecir_citas(args.fecha, args.carpeta, paquete, fuentes)
    ruta = guardar_excel(evaluadas, no_evaluadas, args.fecha, Path(args.carpeta) / "salidas")
    print(f"Modelo versión {paquete['version']} | citas evaluadas: {len(evaluadas)} | no evaluadas: {len(no_evaluadas)}")
    print(f"Archivo generado: {ruta}")

    if args.evaluar:
        _, resumen = evaluar_contra_realidad(evaluadas, fuentes, paquete)
        print("\nInasistencia real por nivel de riesgo:")
        print(resumen.round(3).to_string())


if __name__ == "__main__":
    main()
