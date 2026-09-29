"""
Funciones de modelado y evaluación del proyecto de inasistencia a citas.

- folds_temporales: validación cruzada con ventana creciente (nunca aleatoria)
- metricas: AUC, PR-AUC, Brier y, para la capacidad de llamadas, precisión, recall,
  especificidad y F1 marcando el top-N % de riesgo de cada día
- crear_modelo: pipelines de scikit-learn (preprocesamiento + modelo en un solo objeto)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Paleta de colores del proyecto (validada para daltonismo)
AZUL, NARANJA, AQUA, AMARILLO, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
TINTA, TINTA_SUAVE = "#0b0b0b", "#52514e"


def estilo_graficos():
    """Estilo sobrio: sin bordes superiores/derechos, grilla suave, texto en gris oscuro."""
    plt.rcParams.update({
        "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRIS, "axes.labelcolor": TINTA_SUAVE, "axes.titlecolor": TINTA,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.8,
        "xtick.color": TINTA_SUAVE, "ytick.color": TINTA_SUAVE, "font.size": 10,
        "legend.frameon": False,
    })


# ---------------------------------------------------------------------------
# Validación temporal
# ---------------------------------------------------------------------------
MESES_VALIDACION = [3, 4, 5, 6]   # ene-feb -> mar, ene-mar -> abr, ene-abr -> may, ene-may -> jun


def folds_temporales(fechas, meses=MESES_VALIDACION):
    """Lista de (posiciones_train, posiciones_validación) con ventana creciente."""
    fechas = pd.Series(pd.to_datetime(fechas)).reset_index(drop=True)
    folds = []
    for mes in meses:
        inicio_mes = pd.Timestamp(year=2026, month=mes, day=1)
        train = np.where(fechas < inicio_mes)[0]
        validacion = np.where(fechas.dt.month == mes)[0]
        folds.append((train, validacion))
    return folds


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def marcar_top_por_dia(probabilidad, fechas, capacidad):
    """1 para las citas que están en el top 'capacidad' (p. ej. 10 %) de riesgo de SU día."""
    tabla = pd.DataFrame({"p": np.asarray(probabilidad), "fecha": np.asarray(fechas)})
    rango = tabla.groupby("fecha")["p"].rank(method="first", ascending=False, pct=True)
    return (rango <= capacidad).astype(int).to_numpy()


def matriz_confusion(y, marcado):
    y, marcado = np.asarray(y), np.asarray(marcado)
    vp = int(((marcado == 1) & (y == 1)).sum())
    fp = int(((marcado == 1) & (y == 0)).sum())
    fn = int(((marcado == 0) & (y == 1)).sum())
    vn = int(((marcado == 0) & (y == 0)).sum())
    return vp, fp, fn, vn


def metricas_clasificacion(y, marcado, sufijo=""):
    vp, fp, fn, vn = matriz_confusion(y, marcado)
    precision = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    especificidad = vn / (vn + fp) if vn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {f"Precision{sufijo}": precision, f"Recall{sufijo}": recall,
            f"Especificidad{sufijo}": especificidad, f"F1{sufijo}": f1}


def metricas(y, probabilidad, fechas, capacidades=(0.10, 0.30)):
    """Todas las métricas del proyecto para un conjunto de predicciones."""
    resultado = {
        "AUC": roc_auc_score(y, probabilidad),
        "PR_AUC": average_precision_score(y, probabilidad),
        "Brier": brier_score_loss(y, probabilidad),
    }
    for capacidad in capacidades:
        marcado = marcar_top_por_dia(probabilidad, fechas, capacidad)
        resultado.update(metricas_clasificacion(y, marcado, sufijo=f"@{int(capacidad * 100)}%"))
    return resultado


def resumir_folds(tabla_folds):
    """Convierte la tabla por fold en 'media ± desviación'."""
    resumen = {}
    for columna in tabla_folds.columns:
        if columna == "fold":
            continue
        resumen[columna] = f"{tabla_folds[columna].mean():.3f} ± {tabla_folds[columna].std():.3f}"
    return resumen


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def separar_tipos(X):
    categoricas = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "category"]
    numericas = [c for c in X.columns if c not in categoricas]
    return categoricas, numericas


def crear_modelo(nombre, categoricas, numericas, **parametros):
    """Crea un Pipeline = preprocesamiento + modelo. Así el preprocesamiento se 'aprende'
    solo con los datos de entrenamiento de cada fold (evita fuga de información)."""
    if nombre == "logistica":
        # La regresión logística NECESITA: escalar numéricas, imputar vacíos, one-hot en categorías
        preprocesamiento = ColumnTransformer([
            ("num", Pipeline([("imputar", SimpleImputer(strategy="median", add_indicator=True)),
                              ("escalar", StandardScaler())]), numericas),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100), categoricas),
        ])
        modelo = LogisticRegression(max_iter=3000, **parametros)

    elif nombre == "random_forest":
        # Los árboles NO necesitan escalar; las categorías se convierten a números (códigos)
        preprocesamiento = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), numericas),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                   encoded_missing_value=-1), categoricas),
        ])
        modelo = RandomForestClassifier(n_jobs=-1, random_state=42, **parametros)

    elif nombre == "hist_gradient_boosting":
        # Maneja vacíos y categorías de forma nativa (las categorías van primero)
        preprocesamiento = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                   encoded_missing_value=np.nan), categoricas),
            ("num", "passthrough", numericas),
        ])
        es_categorica = [True] * len(categoricas) + [False] * len(numericas)
        modelo = HistGradientBoostingClassifier(categorical_features=es_categorica, random_state=42,
                                                early_stopping=False, **parametros)

    elif nombre == "lightgbm":
        from lightgbm import LGBMClassifier
        preprocesamiento = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                   encoded_missing_value=np.nan), categoricas),
            ("num", "passthrough", numericas),
        ])
        modelo = LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1, **parametros)
    else:
        raise ValueError(f"Modelo desconocido: {nombre}")

    return Pipeline([("prep", preprocesamiento), ("clf", modelo)])


def parametros_fit(nombre, categoricas):
    """LightGBM necesita saber qué columnas son categóricas al momento de entrenar."""
    if nombre == "lightgbm":
        return {"clf__categorical_feature": list(range(len(categoricas)))}
    return {}


def evaluar_en_folds(nombre, X, y, fechas, folds, parametros=None, capacidades=(0.10, 0.30)):
    """Entrena y evalúa un modelo en cada fold temporal. Devuelve una fila de métricas por fold."""
    parametros = parametros or {}
    categoricas, numericas = separar_tipos(X)
    filas = []
    for i, (train, validacion) in enumerate(folds, start=1):
        modelo = crear_modelo(nombre, categoricas, numericas, **parametros)
        modelo.fit(X.iloc[train], y.iloc[train], **parametros_fit(nombre, categoricas))
        p = modelo.predict_proba(X.iloc[validacion])[:, 1]
        fila = metricas(y.iloc[validacion], p, fechas.iloc[validacion], capacidades)
        fila["fold"] = i
        filas.append(fila)
    return pd.DataFrame(filas)


def evaluar_puntaje_en_folds(puntaje, y, fechas, folds, capacidades=(0.10, 0.30)):
    """Para la línea base: un puntaje fijo (no se entrena nada)."""
    filas = []
    for i, (_, validacion) in enumerate(folds, start=1):
        fila = metricas(y.iloc[validacion], puntaje.iloc[validacion], fechas.iloc[validacion], capacidades)
        fila["fold"] = i
        filas.append(fila)
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Gráficos de apoyo
# ---------------------------------------------------------------------------
def intervalo_confianza(tasa, n, z=1.96):
    """Intervalo de confianza del 95 % para una proporción (aproximación normal)."""
    error = z * np.sqrt(tasa * (1 - tasa) / n)
    return error


def tasa_por_categoria(df, variable, objetivo="no_asiste", min_n=100):
    tabla = df.groupby(variable, observed=True)[objetivo].agg(citas="size", tasa="mean").reset_index()
    tabla = tabla[tabla["citas"] >= min_n]
    tabla["ic95"] = intervalo_confianza(tabla["tasa"], tabla["citas"])
    return tabla


def grafico_tasa(ax, tabla, variable, tasa_global, titulo=None, horizontal=False, leyenda=False):
    """Barras de tasa de inasistencia con intervalo de confianza y línea de la tasa global."""
    etiquetas = tabla[variable].astype(str)
    if horizontal:
        ax.barh(etiquetas, tabla["tasa"], xerr=tabla["ic95"], color=AZUL, height=0.6,
                error_kw={"ecolor": TINTA_SUAVE, "capsize": 2, "linewidth": 1})
        ax.axvline(tasa_global, color=NARANJA, linestyle="--", linewidth=1.5, label=f"Global {tasa_global:.1%}")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    else:
        ax.bar(etiquetas, tabla["tasa"], yerr=tabla["ic95"], color=AZUL, width=0.6,
               error_kw={"ecolor": TINTA_SUAVE, "capsize": 2, "linewidth": 1})
        ax.axhline(tasa_global, color=NARANJA, linestyle="--", linewidth=1.5, label=f"Global {tasa_global:.1%}")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="x", visible=False)
    ax.set_title(titulo or variable)
    if leyenda:
        ax.legend(loc="best", fontsize=8)
