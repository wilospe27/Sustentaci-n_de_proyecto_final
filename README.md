# Sustentación de Proyecto Final — Predicción de Inasistencia

Este repositorio contiene el desarrollo completo del proyecto de análisis de datos e inteligencia artificial para la predicción de inasistencia a citas médicas, incluyendo el análisis exploratorio, la construcción del dataset, el modelado, la evaluación y el despliegue de un flujo de predicción diaria con análisis de costo-efectividad.

## Documentación principal

Para una explicación completa del proyecto (contexto, metodología, arquitectura y resultados), revisar:

- `Datos Reales/documentacion/Documentacion_Proyecto_Inasistencia.docx` (o su versión `.md`)
- `Datos Reales/documentacion/Arquitectura_Datos_y_Analisis_Bivariado_APA.docx`
- `Datos Reales/documentacion/Documento_Proyecto_IA_Inasistencia_Viva1A.docx`

Estos documentos están pensados como punto de partida para quien no ha seguido el desarrollo paso a paso.

## Estructura del repositorio

```
Datos Reales/
├── 01_Auditoria_Etiqueta.ipynb          # Auditoría inicial de la variable objetivo
├── 02_Construccion_Dataset.ipynb        # Construcción y unión de fuentes de datos
├── 03_EDA.ipynb                         # Análisis exploratorio de datos
├── 04_Metricas_y_Variables_v2.ipynb     # Definición de métricas y variables del modelo
├── 05_Modelado.ipynb                    # Entrenamiento de modelos
├── 06_Evaluacion_Final.ipynb            # Evaluación y selección del modelo final
├── 07_Costo_Efectividad_Llamadas.ipynb  # Análisis de costo-efectividad de llamadas
├── 08_Analisis_Bivariado.ipynb          # Análisis bivariado adicional
├── Proyecto.ipynb                       # Notebook consolidado del proyecto
│
├── funciones_inasistencia.py            # Funciones auxiliares del modelo de inasistencia
├── funciones_modelado.py                # Funciones auxiliares de modelado
├── prediccion_diaria.py                 # Script de predicción diaria en producción
├── app_streamlit.py                     # Aplicación interactiva (Streamlit)
│
├── dataset_modelo.parquet               # Dataset final usado en el modelo
├── dataset_modelo_v2.parquet            # Versión actualizada del dataset
├── diccionario_variables.csv            # Diccionario de variables
├── DimCalendario_Colombia_2026_enriquecido.xlsx
├── Gestantes.xlsx / Medicamentos.xlsx / RCV.xlsx / RIPS.xlsx / agendamiento.xlsx  # Fuentes de datos originales
│
├── documentacion/
│   ├── figuras/                         # Gráficas del EDA, modelado y evaluación (26 imágenes)
│   ├── muestra_dataset.csv
│   ├── tipos_dataset_modelo.csv
│   ├── esquema_fuentes.csv
│   └── (documentos .docx/.md descritos arriba)
│
├── modelos/
│   ├── modelo_inasistencia.joblib       # Modelo entrenado (serializado)
│   ├── config_modelo_final.json
│   ├── config_negocio.json
│   ├── mejores_parametros.json
│   ├── metadatos_dataset.json
│   ├── seleccion_eda.json
│   ├── variables_modelo.json
│   └── costos_por_servicio.xlsx
│
└── salidas/
    ├── prediccion_2026-08-27.xlsx               # Ejemplo de predicción diaria generada
    └── llamadas_costo_efectividad_2026-08-27.xlsx  # Ejemplo de análisis de costo-efectividad
```

## Cómo revisar el proyecto

1. Empezar por la documentación en `Datos Reales/documentacion/` para el contexto general.
2. Seguir los notebooks en orden numérico (`01_...` a `08_...`) para ver el proceso completo: auditoría de datos, construcción del dataset, EDA, definición de variables, modelado y evaluación.
3. Los scripts `.py` (`funciones_inasistencia.py`, `funciones_modelado.py`, `prediccion_diaria.py`) contienen la lógica reutilizable que soporta los notebooks y el flujo de predicción en producción.
4. Las carpetas `modelos/` y `salidas/` contienen los artefactos finales (modelo entrenado, configuración y resultados de ejemplo).

## Autores

Santiago Osorio Idárraga · Leslie Pérez · Wilson Ospina Pérez · Juliana Alarcón · Jonatan De La Ossa · Daniela Dominguez Castaño
