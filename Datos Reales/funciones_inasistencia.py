"""
Funciones compartidas del proyecto de predicción de inasistencia a citas.

Este archivo lo usan:
- 02_Construccion_Dataset.ipynb  -> para construir el dataset de entrenamiento
- prediccion_diaria.py            -> para calcular las variables de las citas de mañana

Tener las funciones en un solo lugar garantiza que el modelo se entrene y se use
con EXACTAMENTE las mismas reglas.

Regla de oro: el modelo predice el DÍA ANTERIOR a la cita (D-1). Ese día todavía
no se sabe qué pasó en D-1, así que toda la información usada debe tener fecha
menor o igual a D-2 (parámetro REZAGO_DIAS = 1).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parámetros del proyecto
# ---------------------------------------------------------------------------
PARAMETROS = {
    # Ventana de datos con etiqueta (RIPS llega hasta el 12-sep-2026; ese último día puede estar incompleto)
    "FECHA_INICIO": pd.Timestamp("2026-01-02"),
    "FECHA_FIN_DATOS": pd.Timestamp("2026-09-11"),
    # Particiones temporales
    "FIN_TRAIN": pd.Timestamp("2026-05-31"),
    "FIN_VALIDACION": pd.Timestamp("2026-06-30"),
    "FIN_TEST": pd.Timestamp("2026-07-31"),
    # Limpieza de la etiqueta
    "TERAPIAS_EXCLUIDAS": ["FISIOTERAPIA", "FONOAUDIOLOGIA", "TERAPISTA RESPIRATORIO", "TERAPIA OCUPACIONAL"],
    # Servicios cuyo registro en RIPS cambia de forma inestable (Fase 1.3): desde abril agendan
    # actividades de PyP oral que se reportan como procedimiento, no como consulta
    "SERVICIOS_REGISTRO_INESTABLE": ["AUXILIAR EN SALUD ORAL"],
    "UMBRAL_REPORTE_RIPS": 0.25,        # especialidad con asistencia < 25 % = no se reporta en RIPS
    "MIN_CITAS_PARA_EVALUAR": 20,
    "UMBRAL_Z_DIA_SIN_REPORTE": -4.0,   # día-departamento con asistencia anormalmente baja = falla de reporte RIPS
    "MIN_CITAS_DIA_REGION": 100,
    # Cambio de codificación de la agenda (notebook 01, sección 1.6)
    "FECHA_CAMBIO_CODIFICACION": pd.Timestamp("2026-04-01"),
    "CAIDA_MINIMA_DESCONTINUADO": 0.90,   # código cuyo volumen diario cae > 90 % tras el cambio
    "MIN_CITAS_ANTES_CAMBIO": 300,
    # Momento de predicción
    "REZAGO_DIAS": 1,                   # D-1 no se conoce aún -> se usa información hasta D-2
    # Variables
    "VENTANA_MEDICAMENTOS": 90,
    "UMBRAL_POLIFARMACIA": 5,
    "VENTANA_ATENCIONES": 90,
    "DURACION_GESTACION": 280,
    "K_SUAVIZADO": 3,
}

# Palabras que identifican una cita de PROCEDIMIENTO (no es consulta y no aparece en el RIPS de consultas)
PALABRAS_PROCEDIMIENTO = [
    "ELECTROCARDIOGRAMA", "CITOLOG", "TOMA NO QUIRURGICA", "LAVADO", "CURACION", "REEMPLAZO",
    "INSERCION", "EXTRACCION", "EXTRACCIÓN", "INYECCION", "ESPIROMETRIA", "COLPOSCOPIA",
    "LARINGOSCOPIA", "IMPLANTE", "VPH", "PROCEDIMIENTOS", "SESIONES", "TERAPIA",
    "RETIRO", "BIOPSIA", "SUTURA", "DRENAJE", "AUDIOMETRIA", "ECOGRAFIA", "VACUNA",
]

MUNICIPIO_POR_SEDE = {"CENTRO": "Manizales", "LAURELES": "Manizales", "VILLAMARIA": "Villamaría", "ARMENIA": "Armenia"}
DEPARTAMENTO_POR_MUNICIPIO = {"Manizales": "Caldas", "Villamaría": "Caldas", "Armenia": "Quindío"}


# ---------------------------------------------------------------------------
# 1. Carga
# ---------------------------------------------------------------------------
def estandarizar_id(df):
    """Renombra la columna del paciente a 'id_paciente' y la deja como texto."""
    for columna in df.columns:
        if columna.lower() == "id_med_paciente":
            df = df.rename(columns={columna: "id_paciente"})
    df["id_paciente"] = df["id_paciente"].astype(str)
    return df


def cargar_fuentes(carpeta="."):
    """Lee todos los archivos del proyecto y devuelve un diccionario de DataFrames."""
    archivos = {
        "agendamiento": "agendamiento.xlsx",
        "rips": "RIPS.xlsx",
        "rcv": "RCV.xlsx",
        "gestantes": "Gestantes.xlsx",
        "medicamentos": "Medicamentos.xlsx",
    }
    fuentes = {}
    for nombre, archivo in archivos.items():
        fuentes[nombre] = estandarizar_id(pd.read_excel(f"{carpeta}/{archivo}"))
    fuentes["calendario"] = pd.read_excel(
        f"{carpeta}/DimCalendario_Colombia_2026_enriquecido.xlsx", sheet_name="Calendario"
    )
    return fuentes


def registrar(bitacora, paso, df):
    """Guarda cuántas filas quedan después de un paso (si se pasa una lista bitácora)."""
    if bitacora is not None:
        bitacora.append({"paso": paso, "filas": len(df)})
        print(f"{paso}: {len(df):,} filas")


# ---------------------------------------------------------------------------
# 2. Limpieza de agendamiento
# ---------------------------------------------------------------------------
def convertir_turno(serie):
    """Convierte la hora de la cita soportando los dos formatos del archivo (08:15:00 y 9:40 AM)."""
    texto = serie.astype(str).str.strip()
    formato_24h = pd.to_datetime(texto, format="%H:%M:%S", errors="coerce")
    formato_12h = pd.to_datetime(texto, format="%I:%M %p", errors="coerce")
    return formato_24h.fillna(formato_12h)


def clasificar_tipo_agenda(procedimiento):
    """'Procedimiento' si el texto agendado es un procedimiento; si no, 'Consulta'."""
    if pd.isna(procedimiento):
        return "Consulta"
    texto = str(procedimiento).upper()
    for palabra in PALABRAS_PROCEDIMIENTO:
        if palabra in texto:
            return "Procedimiento"
    return "Consulta"


def homologar_servicio(especialidad, procedimiento):
    """Servicio homologado: une la codificación vieja y la nueva de la agenda.

    En abril de 2026 la agenda cambió su forma de codificar: por ejemplo, "HTA/CRONICO - MEDICO"
    pasó a agendarse como "MEDICINA GENERAL" con procedimiento "HTA - MEDICO". Combinando la
    especialidad con el procedimiento agendado (que se conoce al agendar) recuperamos el servicio real.
    """
    esp = str(especialidad).upper()
    proc = "" if pd.isna(procedimiento) else str(procedimiento).upper()
    texto = f"{esp} | {proc}"
    if any(p in texto for p in ["HTA - MEDICO", "HTA/CRONICO - MEDICO", "DIABETES - MEDICO",
                                "NEFROPROTECCION - MEDICO", "CRONICO - MEDICO INTERNISTA"]):
        return "CRONICOS - MEDICO"
    if any(p in texto for p in ["HTA - ENFERMERIA", "HTA/CRONICO - ENFERMERIA"]):
        return "CRONICOS - ENFERMERIA"
    if any(p in texto for p in ["NUTRICIONISTA - CRONICOS", "CRONICO - NUTRICIONISTA"]):
        return "CRONICOS - NUTRICION"
    if any(p in texto for p in ["PSICOLOGIA - CRONICOS", "CRONICO - PSICOLOGIA"]):
        return "CRONICOS - PSICOLOGIA"
    if "SALUD MENTAL" in texto:
        return "SALUD MENTAL - MD GRAL"
    if "SALUD PUBLICA" in texto:
        return "SALUD PUBLICA - ENFERMERIA" if "ENFERMERIA" in proc or "ENFERMERIA" in esp else "SALUD PUBLICA - MEDICO"
    if "CONTROL PRENATAL" in texto:
        return "CONTROL PRENATAL - ENFERMERIA" if "ENFERMERIA" in texto else "CONTROL PRENATAL - MEDICO"
    if "PLANIFICACION FAMILIAR" in texto:
        return "PLANIFICACION FAMILIAR - MEDICO" if "MEDICO" in texto else "PLANIFICACION FAMILIAR - ENFERMERIA"
    if "GINECO" in esp or "OBSTETRICIA" in esp:
        return "GINECOLOGIA Y OBSTETRICIA"
    curso_de_vida = ["PRIMERA INFANCIA", "INFANCIA", "ADOLESCENCIA", "JUVENTUD", "ADULTEZ", "VEJEZ",
                     "ATENCION A LA FAMILIA", "CRECIMIENTO Y DESARROLLO", "ATENCIÓN INTEGRAL"]
    if any(p in texto for p in curso_de_vida):
        if "ODONT" in texto:
            return "PYP CURSO DE VIDA - ODONTOLOGIA"
        if any(p in texto for p in ["MD GRAL", "MEDICO", "PEDIATRIA"]):
            return "PYP CURSO DE VIDA - MEDICO"
        return "PYP CURSO DE VIDA - ENFERMERIA"
    return esp


def normalizar_sede(nombre):
    """Unifica los 11 nombres de contrato en 4 sedes (el sistema cambió los nombres a mitad de año)."""
    nombre = str(nombre).upper()
    if "ARMENIA" in nombre:
        return "ARMENIA"
    elif "VILLAMARIA" in nombre:
        return "VILLAMARIA"
    elif "LAURELES" in nombre:
        return "LAURELES"
    elif "CENTRO" in nombre:
        return "CENTRO"
    return "OTRA"


def limpiar_agendamiento(agendamiento, bitacora=None):
    """Limpieza de agendamiento: tipos, hora, fechas, duplicados, sede y tipo de agenda."""
    columnas = ["id_paciente", "sexo", "nivel_usr", "tipo_usuario", "edad", "nombre_cnt", "turno",
                "fecha_cita", "fecha_asig", "control", "especialidad_cita", "procedimiento_especifico"]
    citas = agendamiento[columnas].copy()
    registrar(bitacora, "Agendamiento original", citas)

    citas = citas.drop_duplicates()
    registrar(bitacora, "Sin duplicados exactos", citas)

    hora = convertir_turno(citas["turno"])
    citas["hora_cita"] = hora.dt.hour
    citas["minuto_del_dia"] = hora.dt.hour * 60 + hora.dt.minute

    citas["fecha_cita"] = pd.to_datetime(citas["fecha_cita"]).dt.normalize()
    citas["fecha_asig"] = pd.to_datetime(citas["fecha_asig"]).dt.normalize()
    citas["especialidad"] = citas["especialidad_cita"].str.strip().str.upper()

    citas = citas.dropna(subset=["fecha_asig", "especialidad", "minuto_del_dia"])
    registrar(bitacora, "Con fecha de asignación, especialidad y hora", citas)

    citas["lead_time"] = (citas["fecha_cita"] - citas["fecha_asig"]).dt.days
    citas = citas[citas["lead_time"] >= 0]

    citas["tipo_agenda"] = citas["procedimiento_especifico"].apply(clasificar_tipo_agenda)
    citas["servicio_homologado"] = [homologar_servicio(e, p) for e, p in
                                    zip(citas["especialidad"], citas["procedimiento_especifico"])]

    # Mismo servicio, mismo paciente, mismo día -> dejamos la última asignada (la vigente).
    # Se usa el servicio homologado porque en abril convivieron la codificación vieja y la nueva.
    citas = citas.sort_values(["id_paciente", "fecha_cita", "servicio_homologado", "fecha_asig", "minuto_del_dia"])
    citas = citas.drop_duplicates(subset=["id_paciente", "fecha_cita", "servicio_homologado"], keep="last")
    registrar(bitacora, "Una cita por paciente-día-servicio", citas)
    citas["sede"] = citas["nombre_cnt"].apply(normalizar_sede)
    citas["municipio"] = citas["sede"].map(MUNICIPIO_POR_SEDE)
    citas["departamento"] = citas["municipio"].map(DEPARTAMENTO_POR_MUNICIPIO)
    citas["regimen"] = np.where(citas["nombre_cnt"].str.contains("SUBSIDIADO"), "Subsidiado", "Contributivo")
    return citas.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Variable objetivo
# ---------------------------------------------------------------------------
def preparar_rips(rips):
    """Fecha de atención (solo día) y servicio = 2 últimos dígitos del CUPS (890201 -> 01)."""
    rips = rips.dropna(subset=["codconsulta"]).copy()
    rips["fecha_atencion"] = pd.to_datetime(rips["fechainicioatencion"]).dt.normalize()
    rips["servicio"] = (rips["codconsulta"] % 100).astype(int)
    return rips


def aprender_mapa_servicio(citas, rips):
    """Aprende qué servicio de RIPS corresponde a cada especialidad, usando los días sin ambigüedad
    (paciente con exactamente 1 cita y 1 atención ese día)."""
    n_citas = citas.groupby(["id_paciente", "fecha_cita"]).size().rename("n_citas")
    n_rips = rips.groupby(["id_paciente", "fecha_atencion"]).size().rename("n_rips")
    n_rips.index.names = ["id_paciente", "fecha_cita"]

    dias = pd.concat([n_citas, n_rips], axis=1).dropna()
    dias = dias[(dias["n_citas"] == 1) & (dias["n_rips"] == 1)].reset_index()

    pares = dias.merge(citas[["id_paciente", "fecha_cita", "especialidad"]], on=["id_paciente", "fecha_cita"])
    pares = pares.merge(
        rips[["id_paciente", "fecha_atencion", "servicio"]].rename(columns={"fecha_atencion": "fecha_cita"}),
        on=["id_paciente", "fecha_cita"],
    )
    mapa = {}
    for especialidad, grupo in pares.groupby("especialidad"):
        mapa[especialidad] = grupo["servicio"].value_counts().index[0]
    return pd.Series(mapa, name="servicio")


def marcar_asistencia(citas, rips, mapa_servicio):
    """asiste = 1 si hay atención en RIPS del mismo paciente, mismo día y mismo servicio.
    Si la especialidad no tiene servicio conocido, basta con mismo paciente y mismo día."""
    citas = citas.copy()
    citas["servicio"] = citas["especialidad"].map(mapa_servicio).astype("Int64")

    por_servicio = (rips[["id_paciente", "fecha_atencion", "servicio"]].drop_duplicates()
                    .rename(columns={"fecha_atencion": "fecha_cita"}).assign(asiste_servicio=1))
    por_servicio["servicio"] = por_servicio["servicio"].astype("Int64")
    por_dia = (rips[["id_paciente", "fecha_atencion"]].drop_duplicates()
               .rename(columns={"fecha_atencion": "fecha_cita"}).assign(asiste_dia=1))

    filas = len(citas)
    citas = citas.merge(por_servicio, on=["id_paciente", "fecha_cita", "servicio"], how="left")
    citas = citas.merge(por_dia, on=["id_paciente", "fecha_cita"], how="left")
    assert len(citas) == filas, "El merge duplicó filas"

    citas["asiste"] = np.where(citas["servicio"].notna(),
                               citas["asiste_servicio"].fillna(0),
                               citas["asiste_dia"].fillna(0)).astype(int)
    return citas.drop(columns=["asiste_servicio", "asiste_dia"])


def marcar_posible_reprogramacion(citas):
    """1 si, después de pedir esta cita y antes de su fecha, el paciente pidió otra del mismo servicio
    (servicio homologado, para no perder reprogramaciones que cruzan el cambio de codificación de abril)."""
    citas = citas.sort_values(["id_paciente", "servicio_homologado", "fecha_asig", "fecha_cita"])
    siguiente_asig = citas.groupby(["id_paciente", "servicio_homologado"])["fecha_asig"].shift(-1)
    reprogramada = (siguiente_asig > citas["fecha_asig"]) & (siguiente_asig < citas["fecha_cita"])
    return reprogramada.astype(int).sort_index()


def detectar_dias_sin_reporte(citas, params=PARAMETROS):
    """Días-departamento en que RIPS no reportó (total o parcialmente).

    Usamos un puntaje z ROBUSTO: en vez de promedio y desviación estándar (que se dejan arrastrar
    por los propios días malos) usamos la mediana y la MAD (mediana de las desviaciones absolutas).
    z = (tasa del día - mediana) / (1.4826 * MAD). Si z < -4, el día es anómalo.
    """
    resumen = (citas.groupby(["departamento", "fecha_cita"])["asiste"]
               .agg(n_citas="size", tasa_asistencia="mean").reset_index())
    resumen = resumen[resumen["n_citas"] >= params["MIN_CITAS_DIA_REGION"]]

    anomalos = []
    for departamento, grupo in resumen.groupby("departamento"):
        mediana = grupo["tasa_asistencia"].median()
        mad = (grupo["tasa_asistencia"] - mediana).abs().median() * 1.4826
        grupo = grupo.assign(z_robusto=(grupo["tasa_asistencia"] - mediana) / mad)
        anomalos.append(grupo[grupo["z_robusto"] < params["UMBRAL_Z_DIA_SIN_REPORTE"]])
    return pd.concat(anomalos, ignore_index=True)


def especialidades_no_reportadas(citas, params=PARAMETROS):
    """Terapias (decisión del proyecto) + servicios con registro inestable
    + especialidades con asistencia < 25 % (no se reportan en RIPS)."""
    resumen = citas.groupby("especialidad")["asiste"].agg(n_citas="size", tasa="mean")
    bajas = resumen[(resumen["n_citas"] >= params["MIN_CITAS_PARA_EVALUAR"])
                    & (resumen["tasa"] < params["UMBRAL_REPORTE_RIPS"])].index
    return sorted(set(params["TERAPIAS_EXCLUIDAS"]) | set(params["SERVICIOS_REGISTRO_INESTABLE"]) | set(bajas))


def codigos_descontinuados(citas, params=PARAMETROS):
    """Especialidades (código de agenda) que dejaron de usarse con el cambio de codificación de abril.

    Las citas que quedan con el código viejo después del cambio son 'huérfanas' del sistema anterior:
    casi ninguna aparece en RIPS (~100 % de 'inasistencia'), así que no representan comportamiento real.
    """
    cambio = params["FECHA_CAMBIO_CODIFICACION"]
    antes = citas[citas["fecha_cita"] < cambio]
    despues = citas[citas["fecha_cita"] >= cambio]
    dias_antes = max(antes["fecha_cita"].nunique(), 1)
    dias_despues = max(despues["fecha_cita"].nunique(), 1)
    diario_antes = antes["especialidad"].value_counts() / dias_antes
    diario_despues = despues["especialidad"].value_counts().reindex(diario_antes.index, fill_value=0) / dias_despues
    candidatos = antes["especialidad"].value_counts()
    candidatos = candidatos[candidatos >= params["MIN_CITAS_ANTES_CAMBIO"]].index
    caida = 1 - diario_despues[candidatos] / diario_antes[candidatos]
    return sorted(caida[caida > params["CAIDA_MINIMA_DESCONTINUADO"]].index)


def construir_citas_etiquetadas(citas_limpias, rips, params=PARAMETROS, mapa_servicio=None,
                                excluidas=None, fecha_fin=None, bitacora=None, descontinuados=None):
    """Etiqueta las citas y aplica todas las reglas de la auditoría (Fase 1).

    Devuelve:
      etiquetadas -> citas válidas para el modelo (con no_asiste)
      todas       -> todas las citas limpias con asiste y posible_reprogramacion (para variables v2)
      mapa_servicio, excluidas -> lo aprendido, para reutilizarlo en la predicción diaria
    (la lista de códigos descontinuados queda en el atributo .attrs["descontinuados"] de 'etiquetadas')
    """
    fecha_fin = params["FECHA_FIN_DATOS"] if fecha_fin is None else fecha_fin

    # La marca de reprogramación mira citas futuras, por eso se calcula antes de recortar la ventana
    citas = citas_limpias.copy()
    citas["posible_reprogramacion"] = marcar_posible_reprogramacion(citas)

    citas = citas[(citas["fecha_cita"] >= params["FECHA_INICIO"]) & (citas["fecha_cita"] <= fecha_fin)]
    registrar(bitacora, "Dentro de la ventana con RIPS", citas)

    if mapa_servicio is None:
        mapa_servicio = aprender_mapa_servicio(citas, rips)
    todas = marcar_asistencia(citas, rips, mapa_servicio)

    # Regla 1: el modelo predice el día anterior -> las citas asignadas el mismo día no existen aún
    validas = todas[todas["lead_time"] >= 1]
    registrar(bitacora, "Asignadas al menos 1 día antes (lead time >= 1)", validas)

    # Regla 2: los procedimientos no aparecen en el RIPS de consultas
    validas = validas[validas["tipo_agenda"] == "Consulta"]
    registrar(bitacora, "Solo consultas (sin procedimientos)", validas)

    # Regla 3: terapias y especialidades que RIPS no reporta
    if excluidas is None:
        excluidas = especialidades_no_reportadas(validas, params)
    validas = validas[~validas["especialidad"].isin(excluidas)]
    registrar(bitacora, "Sin terapias ni especialidades no reportadas", validas)

    # Regla 3b: citas huérfanas con códigos de agenda descontinuados en el cambio de abril
    if descontinuados is None:
        descontinuados = codigos_descontinuados(validas, params)
    huerfana = validas["especialidad"].isin(descontinuados) & (validas["fecha_cita"] >= params["FECHA_CAMBIO_CODIFICACION"])
    validas = validas[~huerfana]
    registrar(bitacora, "Sin citas huérfanas de códigos descontinuados", validas)

    # Regla 4: días en que RIPS no reportó (falla de reporte, no inasistencia)
    dias_malos = detectar_dias_sin_reporte(validas, params)[["departamento", "fecha_cita"]].assign(dia_sin_reporte=1)
    validas = validas.merge(dias_malos, on=["departamento", "fecha_cita"], how="left")
    validas = validas[validas["dia_sin_reporte"].isna()].drop(columns="dia_sin_reporte")
    registrar(bitacora, "Sin días sin reporte en RIPS", validas)

    # Regla 5: no está en RIPS pero pidió otra cita de la misma especialidad antes -> reprogramación
    validas = validas[~((validas["asiste"] == 0) & (validas["posible_reprogramacion"] == 1))]
    registrar(bitacora, "Sin posibles reprogramaciones", validas)

    validas = validas.copy()
    validas["no_asiste"] = 1 - validas["asiste"]
    validas = validas.reset_index(drop=True)
    validas.attrs["descontinuados"] = list(descontinuados)
    return validas, todas, mapa_servicio, excluidas


# ---------------------------------------------------------------------------
# 4. Variables de agendamiento y sociodemográficas
# ---------------------------------------------------------------------------
def franja_horaria(hora):
    if hora < 9:
        return "1. Temprano (<9h)"
    elif hora < 12:
        return "2. Mañana (9-12h)"
    elif hora < 14:
        return "3. Mediodía (12-14h)"
    elif hora < 17:
        return "4. Tarde (14-17h)"
    return "5. Final tarde (>=17h)"


def familia_especialidad(nombre):
    if any(p in nombre for p in ["ODONT", "SALUD ORAL", "ENDODONCIA", "CIRUGIA ORAL", "MAXILOFACIAL"]):
        return "Odontología"
    elif any(p in nombre for p in ["HTA", "CRONICO", "DIABETES", "NEFROPROTECCION"]):
        return "Crónicos"
    elif any(p in nombre for p in ["PSICOLOGIA", "PSIQUIATRIA", "SALUD MENTAL"]):
        return "Salud mental"
    elif "NUTRICIONISTA" in nombre:
        return "Nutrición"
    elif "PEDIATRIA" in nombre:
        return "Pediatría"
    elif any(p in nombre for p in ["ENFERMERIA", "- ENF"]):
        return "Enfermería / PyP"
    elif any(p in nombre for p in ["MEDICINA GENERAL", "MD GRAL", "- MEDICO"]):
        return "Medicina general"
    return "Especialista"


def variables_agendamiento(citas):
    citas = citas.copy()
    citas["turno_sin"] = np.sin(2 * np.pi * citas["minuto_del_dia"] / 1440)
    citas["turno_cos"] = np.cos(2 * np.pi * citas["minuto_del_dia"] / 1440)
    citas["franja_horaria"] = citas["hora_cita"].apply(franja_horaria)
    citas["es_control"] = (citas["control"] == "X").astype(int)
    citas["lead_time_cat"] = pd.cut(citas["lead_time"], bins=[0, 7, 30, 90, 10_000],
                                    labels=["1-7 días", "8-30 días", "31-90 días", ">90 días"]).astype(str)
    citas["familia_servicio"] = citas["servicio_homologado"].apply(familia_especialidad)
    citas["es_pac"] = citas["especialidad"].str.startswith("PAC ").astype(int)
    citas["edad"] = citas["edad"].astype(int)
    citas["grupo_edad"] = pd.cut(
        citas["edad"], bins=[-1, 5, 11, 17, 28, 59, 200],
        labels=["1. Primera infancia (0-5)", "2. Infancia (6-11)", "3. Adolescencia (12-17)",
                "4. Juventud (18-28)", "5. Adultez (29-59)", "6. Vejez (60+)"]).astype(str)
    citas["sexo"] = citas["sexo"].str.strip()
    citas["nivel_usr"] = citas["nivel_usr"].astype(int)
    citas["tipo_usuario"] = citas["tipo_usuario"].str.strip()
    return citas


# ---------------------------------------------------------------------------
# 5. Calendario
# ---------------------------------------------------------------------------
COLUMNAS_CALENDARIO = {
    "Fecha": "fecha_cita", "Día_Semana_Num": "dia_semana_num", "Día_Semana": "dia_semana", "Mes": "mes",
    "Semana_Mes": "semana_mes", "Es_Festivo": "es_festivo", "Es_Día_Previo_Festivo": "es_dia_previo_festivo",
    "Es_Día_Posterior_Festivo": "es_dia_posterior_festivo", "Es_Puente_Festivo": "es_puente_festivo",
    "Días_Hasta_Festivo": "dias_hasta_festivo", "Es_Primer_Día_Hábil_Mes": "es_primer_dia_habil_mes",
    "Es_Último_Día_Hábil_Mes": "es_ultimo_dia_habil_mes",
}


def variables_calendario(citas, calendario):
    cal = calendario[list(COLUMNAS_CALENDARIO)].rename(columns=COLUMNAS_CALENDARIO)
    cal["fecha_cita"] = pd.to_datetime(cal["fecha_cita"]).dt.normalize()
    filas = len(citas)
    citas = citas.merge(cal, on="fecha_cita", how="left", validate="m:1")
    assert len(citas) == filas
    return citas


# ---------------------------------------------------------------------------
# 6. Historial (solo información hasta D-2)
# ---------------------------------------------------------------------------
def variables_historial(objetivo, historia, params=PARAMETROS, por=("id_paciente",), sufijo=""):
    """Historial de asistencia usando solo citas con fecha <= D-2.

    Idea: resumimos la historia por día con un acumulado (cumsum) y, para cada cita objetivo,
    buscamos el último acumulado conocido antes de su fecha límite con pd.merge_asof
    ("dame el último valor conocido antes de esta fecha").
    """
    por = list(por)
    dia = (historia.groupby(por + ["fecha_cita"])["no_asiste"]
           .agg(citas_dia="size", inasistencias_dia="sum").reset_index()
           .sort_values(por + ["fecha_cita"]))
    grupo = dia.groupby(por)
    dia["acum_citas"] = grupo["citas_dia"].cumsum()
    dia["acum_inasistencias"] = grupo["inasistencias_dia"].cumsum()
    dia = dia.rename(columns={"fecha_cita": "fecha_historia"}).sort_values("fecha_historia")

    obj = objetivo[["cita_id"] + por + ["fecha_cita"]].copy()
    obj["fecha_limite"] = obj["fecha_cita"] - pd.Timedelta(days=params["REZAGO_DIAS"] + 1)
    obj = obj.sort_values("fecha_limite")

    cruce = pd.merge_asof(
        obj, dia[por + ["fecha_historia", "acum_citas", "acum_inasistencias", "inasistencias_dia"]],
        left_on="fecha_limite", right_on="fecha_historia", by=por, direction="backward",
    )
    n_citas = cruce["acum_citas"].fillna(0)
    n_inasist = cruce["acum_inasistencias"].fillna(0)
    prior = params["PRIOR_INASISTENCIA"]
    k = params["K_SUAVIZADO"]

    resultado = pd.DataFrame({"cita_id": cruce["cita_id"]})
    resultado["n_citas_previas" + sufijo] = n_citas.astype(int)
    resultado["n_inasistidas_previas" + sufijo] = n_inasist.astype(int)
    resultado["n_asistidas_previas" + sufijo] = (n_citas - n_inasist).astype(int)
    resultado["tasa_inasistencia_hist" + sufijo] = n_inasist / n_citas.replace(0, np.nan)
    resultado["tasa_inasistencia_suavizada" + sufijo] = (n_inasist + k * prior) / (n_citas + k)
    resultado["sin_historial" + sufijo] = (n_citas == 0).astype(int)
    resultado["estado_ultima_cita" + sufijo] = np.select(
        [cruce["inasistencias_dia"].isna(), cruce["inasistencias_dia"] > 0],
        ["Sin historial", "No asistió"], default="Asistió")
    resultado["dias_desde_ultima_cita" + sufijo] = (cruce["fecha_cita"] - cruce["fecha_historia"]).dt.days

    filas = len(objetivo)
    objetivo = objetivo.merge(resultado, on="cita_id", how="left", validate="1:1")
    assert len(objetivo) == filas
    return objetivo


# ---------------------------------------------------------------------------
# 7. Variables clínicas (solo información hasta D-2)
# ---------------------------------------------------------------------------
def variables_clinicas(objetivo, rcv, gestantes, medicamentos, rips, params=PARAMETROS):
    objetivo = objetivo.copy()
    base = objetivo[["cita_id", "id_paciente", "fecha_cita"]].copy()
    # Todo lo registrado antes de D-1 (es decir, hasta D-2)
    base["fecha_limite"] = base["fecha_cita"] - pd.Timedelta(days=params["REZAGO_DIAS"])

    # RCV (foto sin fecha)
    r = rcv.copy()
    r["programa_rcv"] = r["Programa"].fillna("Programa de Hipertension")
    r["diabetes"] = (r["Diabetes"] == "X").astype(int)
    r = r[["id_paciente", "programa_rcv", "diabetes"]].drop_duplicates("id_paciente")
    objetivo = objetivo.merge(r, on="id_paciente", how="left", validate="m:1")
    objetivo["es_cronico"] = objetivo["programa_rcv"].notna().astype(int)
    objetivo["programa_rcv"] = objetivo["programa_rcv"].fillna("Sin programa")
    objetivo["diabetes"] = objetivo["diabetes"].fillna(0).astype(int)

    # Gestante activa: FUM <= fecha cita <= FUM + 280 días
    g = gestantes[["id_paciente", "fum"]].copy()
    g["fum"] = pd.to_datetime(g["fum"]).dt.normalize()
    cruce = base.merge(g, on="id_paciente")
    cruce["gestante"] = ((cruce["fecha_cita"] >= cruce["fum"])
                         & (cruce["fecha_cita"] <= cruce["fum"] + pd.Timedelta(days=params["DURACION_GESTACION"])))
    objetivo["gestante_activa"] = objetivo["cita_id"].map(cruce.groupby("cita_id")["gestante"].max()).fillna(0).astype(int)
    objetivo.loc[objetivo["sexo"] == "Masculino", "gestante_activa"] = 0  # error de registro

    # Medicamentos distintos formulados en los 90 días previos
    m = medicamentos[["id_paciente", "medicamento", "fec_orden"]].copy()
    m["fecha_orden"] = pd.to_datetime(m["fec_orden"]).dt.normalize()
    cruce = base.merge(m, on="id_paciente")
    en_ventana = ((cruce["fecha_orden"] < cruce["fecha_limite"])
                  & (cruce["fecha_orden"] >= cruce["fecha_cita"] - pd.Timedelta(days=params["VENTANA_MEDICAMENTOS"])))
    n_meds = cruce[en_ventana].groupby("cita_id")["medicamento"].nunique()
    objetivo["n_medicamentos_90d"] = objetivo["cita_id"].map(n_meds).fillna(0).astype(int)
    objetivo["polifarmacia"] = (objetivo["n_medicamentos_90d"] >= params["UMBRAL_POLIFARMACIA"]).astype(int)

    # Diagnósticos distintos antes de la cita (primera fecha en que apareció cada diagnóstico)
    columnas_dx = ["codDiagnosticoPrincipal", "codDiagnosticoRelacionado1",
                   "codDiagnosticoRelacionado2", "codDiagnosticoRelacionado3"]
    dx = rips.melt(id_vars=["id_paciente", "fecha_atencion"], value_vars=columnas_dx, value_name="diagnostico")
    dx = dx.dropna(subset=["diagnostico"])
    primera_vez = dx.groupby(["id_paciente", "diagnostico"])["fecha_atencion"].min().reset_index()
    cruce = base.merge(primera_vez, on="id_paciente")
    cruce = cruce[cruce["fecha_atencion"] < cruce["fecha_limite"]]
    objetivo["n_diagnosticos_previos"] = objetivo["cita_id"].map(cruce.groupby("cita_id")["diagnostico"].nunique()).fillna(0).astype(int)

    # Días con atención en los 90 días previos
    dias_atencion = rips[["id_paciente", "fecha_atencion"]].drop_duplicates()
    cruce = base.merge(dias_atencion, on="id_paciente")
    en_ventana = ((cruce["fecha_atencion"] < cruce["fecha_limite"])
                  & (cruce["fecha_atencion"] >= cruce["fecha_cita"] - pd.Timedelta(days=params["VENTANA_ATENCIONES"])))
    objetivo["n_atenciones_90d"] = objetivo["cita_id"].map(cruce[en_ventana].groupby("cita_id").size()).fillna(0).astype(int)
    return objetivo


# ---------------------------------------------------------------------------
# 8. Variables v2 (Fase 4)
# ---------------------------------------------------------------------------
def variables_reprogramaciones_previas(objetivo, todas, params=PARAMETROS):
    """Cuántas citas reprogramó/canceló antes el paciente (citas con fecha <= D-2)."""
    eventos = todas[(todas["posible_reprogramacion"] == 1) & (todas["asiste"] == 0)]
    dia = eventos.groupby(["id_paciente", "fecha_cita"]).size().rename("n").reset_index()
    dia = dia.sort_values(["id_paciente", "fecha_cita"])
    dia["acum"] = dia.groupby("id_paciente")["n"].cumsum()
    dia = dia.rename(columns={"fecha_cita": "fecha_evento"}).sort_values("fecha_evento")

    obj = objetivo[["cita_id", "id_paciente", "fecha_cita"]].copy()
    obj["fecha_limite"] = obj["fecha_cita"] - pd.Timedelta(days=params["REZAGO_DIAS"] + 1)
    obj = obj.sort_values("fecha_limite")
    cruce = pd.merge_asof(obj, dia[["id_paciente", "fecha_evento", "acum"]], left_on="fecha_limite",
                          right_on="fecha_evento", by="id_paciente", direction="backward")
    objetivo = objetivo.copy()
    objetivo["n_reprogramaciones_previas"] = objetivo["cita_id"].map(cruce.set_index("cita_id")["acum"]).fillna(0).astype(int)
    return objetivo


def variables_carga_agenda(objetivo, agenda):
    """Otras citas del paciente que ya estaban en la agenda el día anterior (asignadas antes de D)."""
    otras = agenda[["id_paciente", "fecha_cita", "fecha_asig", "especialidad"]].rename(
        columns={"fecha_cita": "fecha_otra", "fecha_asig": "asig_otra", "especialidad": "esp_otra"})
    cruce = objetivo[["cita_id", "id_paciente", "fecha_cita", "especialidad"]].merge(otras, on="id_paciente")
    conocida = cruce["asig_otra"] < cruce["fecha_cita"]
    es_ella_misma = (cruce["fecha_otra"] == cruce["fecha_cita"]) & (cruce["esp_otra"] == cruce["especialidad"])
    cruce = cruce[conocida & ~es_ella_misma]
    dias = (cruce["fecha_otra"] - cruce["fecha_cita"]).dt.days

    objetivo = objetivo.copy()
    objetivo["n_otras_citas_mismo_dia"] = objetivo["cita_id"].map(cruce[dias == 0].groupby("cita_id").size()).fillna(0).astype(int)
    proximas = cruce[(dias >= 1) & (dias <= 7)]
    objetivo["n_citas_proximos_7d"] = objetivo["cita_id"].map(proximas.groupby("cita_id").size()).fillna(0).astype(int)
    recientes = cruce[(dias >= -7) & (dias <= -1)]
    objetivo["n_citas_ultimos_7d"] = objetivo["cita_id"].map(recientes.groupby("cita_id").size()).fillna(0).astype(int)
    return objetivo


def variables_interacciones(objetivo):
    """Cruces de categorías con hipótesis de negocio (útiles sobre todo para la regresión logística)."""
    objetivo = objetivo.copy()
    objetivo["familia_x_lead"] = objetivo["familia_servicio"] + " | " + objetivo["lead_time_cat"]
    objetivo["edad_x_control"] = objetivo["grupo_edad"] + " | " + np.where(objetivo["es_control"] == 1, "Control", "Primera vez")
    return objetivo


# ---------------------------------------------------------------------------
# 9. Ensamble
# ---------------------------------------------------------------------------
def asignar_particion(fecha, params=PARAMETROS):
    if fecha <= params["FIN_TRAIN"]:
        return "train"
    elif fecha <= params["FIN_VALIDACION"]:
        return "validacion"
    elif fecha <= params["FIN_TEST"]:
        return "test"
    return "produccion_simulada"


def calcular_variables(objetivo, historia, todas, agenda, fuentes, rips, params=PARAMETROS):
    """Calcula TODAS las variables para un conjunto de citas objetivo.
    Se usa igual en el entrenamiento y en la predicción diaria."""
    objetivo = variables_agendamiento(objetivo)
    objetivo = variables_calendario(objetivo, fuentes["calendario"])
    objetivo = variables_historial(objetivo, historia, params)
    objetivo = variables_historial(objetivo, historia, params, por=("id_paciente", "servicio_homologado"), sufijo="_esp")
    objetivo = variables_clinicas(objetivo, fuentes["rcv"], fuentes["gestantes"], fuentes["medicamentos"], rips, params)
    objetivo = variables_reprogramaciones_previas(objetivo, todas, params)
    objetivo = variables_carga_agenda(objetivo, agenda)
    objetivo = variables_interacciones(objetivo)
    return objetivo


# Columnas que NO son variables del modelo
COLUMNAS_ID = ["cita_id", "id_paciente", "fecha_cita", "fecha_asig", "particion", "no_asiste"]

VARIABLES_V1 = [
    "lead_time", "lead_time_cat", "hora_cita", "franja_horaria", "turno_sin", "turno_cos", "es_control",
    "servicio_homologado", "especialidad", "familia_servicio", "es_pac",
    "edad", "grupo_edad", "sexo", "nivel_usr", "tipo_usuario", "regimen", "sede", "municipio",
    "dia_semana_num", "dia_semana", "mes", "semana_mes", "es_festivo", "es_dia_previo_festivo",
    "es_dia_posterior_festivo", "es_puente_festivo", "dias_hasta_festivo", "es_primer_dia_habil_mes",
    "es_ultimo_dia_habil_mes",
    "n_citas_previas", "n_asistidas_previas", "n_inasistidas_previas", "tasa_inasistencia_hist",
    "tasa_inasistencia_suavizada", "estado_ultima_cita", "dias_desde_ultima_cita", "sin_historial",
    "es_cronico", "programa_rcv", "diabetes", "gestante_activa", "n_medicamentos_90d", "polifarmacia",
    "n_diagnosticos_previos", "n_atenciones_90d",
]

BLOQUES_V2 = {
    "historial_especialidad": ["n_citas_previas_esp", "n_inasistidas_previas_esp", "tasa_inasistencia_suavizada_esp",
                               "estado_ultima_cita_esp", "dias_desde_ultima_cita_esp", "sin_historial_esp"],
    "reprogramaciones": ["n_reprogramaciones_previas"],
    "carga_agenda": ["n_otras_citas_mismo_dia", "n_citas_proximos_7d", "n_citas_ultimos_7d"],
    "interacciones": ["familia_x_lead", "edad_x_control"],
}


def construir_dataset(fuentes, params=PARAMETROS, bitacora=None):
    """Pipeline completo de entrenamiento. Devuelve (dataset, metadatos)."""
    params = dict(params)
    rips = preparar_rips(fuentes["rips"])
    agenda = limpiar_agendamiento(fuentes["agendamiento"], bitacora)

    etiquetadas, todas, mapa_servicio, excluidas = construir_citas_etiquetadas(agenda, rips, params, bitacora=bitacora)
    etiquetadas["cita_id"] = np.arange(len(etiquetadas))
    etiquetadas["particion"] = etiquetadas["fecha_cita"].apply(lambda f: asignar_particion(f, params))

    # Prior para la tasa suavizada: SOLO con el periodo de entrenamiento
    params["PRIOR_INASISTENCIA"] = float(etiquetadas.loc[etiquetadas["particion"] == "train", "no_asiste"].mean())

    dataset = calcular_variables(etiquetadas, etiquetadas, todas, agenda, fuentes, rips, params)
    metadatos = {
        "mapa_servicio": mapa_servicio.to_dict(),
        "especialidades_excluidas": list(excluidas),
        "codigos_descontinuados": etiquetadas.attrs.get("descontinuados", []),
        "PRIOR_INASISTENCIA": params["PRIOR_INASISTENCIA"],
    }
    return dataset, metadatos
