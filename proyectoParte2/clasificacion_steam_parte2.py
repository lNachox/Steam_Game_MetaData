"""
==========================================================================
 CLASIFICACIÓN SUPERVISADA — Steam Games (predicción de éxito)
 Parte 2: Diseño experimental, implementación y análisis de resultados
 Modelos: Random Forest vs Logistic Regression
==========================================================================

FLUJO COMPLETO:
  Dataset → Features (tags) + Target (Exitoso)
  → Train/Test Split → Entrenamiento → Evaluación → Comparación → Ejemplos
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, classification_report
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")


# =========================================================================
# PASO 1 — CARGA Y PREPARACIÓN DE DATOS
# =========================================================================
# Features: las 134 columnas tag_* (binarias, 0/1)
#   → Información disponible ANTES de lanzar el juego
#   → No requieren normalización ni encoding
#
# Target: columna Exitoso (0 = no exitoso, 1 = exitoso)
#   → Definida en la Parte 1 del proyecto
#
# IMPORTANTE — Sin data leakage:
#   Solo usamos tags como features. Columnas como ReviewScore,
#   PositiveRatio o TotalReviews NO se incluyen porque son
#   consecuencia del éxito, no predictores de él.
# =========================================================================

df = pd.read_csv("SteamGames_prepared.csv")

tag_cols = [c for c in df.columns if c.startswith("tag_")]
X = df[tag_cols].copy()
y = df["Exitoso"].copy()

n_exitosos   = y.sum()
n_no_exit    = (y == 0).sum()
pct_exitosos = y.mean() * 100

print("=" * 65)
print("PASO 1 — CARGA Y PREPARACIÓN DE DATOS")
print("=" * 65)
print(f"  Total de juegos        : {len(df):,}")
print(f"  Features (tags)        : {len(tag_cols)}")
print(f"  Juegos exitosos        : {n_exitosos:,}  ({pct_exitosos:.1f}%)")
print(f"  Juegos no exitosos     : {n_no_exit:,}  ({100-pct_exitosos:.1f}%)")
print(f"  Valores nulos en X     : {X.isnull().sum().sum()}")
print()
print("  Nota: sin data leakage — solo se usan tags como features.")
print("  ReviewScore, PositiveRatio y TotalReviews fueron excluidas")
print("  porque son consecuencia del éxito, no predictores.")
print()


# =========================================================================
# PASO 2 — DIVISIÓN TRAIN / TEST
# =========================================================================
# Estrategia: Holdout 80% / 20%
#
# Justificación:
#   - Con 22,531 juegos, el 20% de test (≈4,506 juegos) es suficiente
#     para obtener estimaciones estables de las métricas.
#   - stratify=y garantiza que la proporción de exitosos/no exitosos
#     sea la misma en train y test (evita sesgo por desbalance).
#   - random_state fijo para reproducibilidad.
#
# Alternativa descartada — validación cruzada k-fold:
#   Sería más robusta estadísticamente, pero más costosa en cómputo
#   para Random Forest con 134 features. El holdout es suficiente
#   dado el tamaño del dataset.
# =========================================================================

TEST_SIZE    = 0.20
RANDOM_STATE = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("=" * 65)
print("PASO 2 — DIVISIÓN TRAIN / TEST")
print("=" * 65)
print(f"  Estrategia     : Holdout {int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test")
print(f"  Estratificado  : sí (mantiene proporción de clases)")
print(f"  Train          : {len(X_train):,} juegos")
print(f"  Test           : {len(X_test):,} juegos")
print(f"  Exitosos train : {y_train.mean()*100:.1f}%")
print(f"  Exitosos test  : {y_test.mean()*100:.1f}%")
print()


# =========================================================================
# PASO 3 — DEFINICIÓN DE MODELOS
# =========================================================================
# Se comparan dos clasificadores con enfoques complementarios:
#
# MODELO 1 — Random Forest
#   Ensemble de árboles de decisión entrenados con bootstrap.
#   Pros: robusto, maneja bien features binarias, no requiere
#         normalización, captura relaciones no lineales entre tags.
#   Contras: menos interpretable que regresión, más lento de entrenar.
#   class_weight='balanced': ajusta pesos inversamente proporcional
#   a la frecuencia de cada clase (mitiga el desbalance 69/31).
#
# MODELO 2 — Logistic Regression
#   Modelo lineal que estima la probabilidad P(Exitoso=1 | tags).
#   Pros: rápido, coeficientes interpretables (qué tags suman/restan
#         probabilidad de éxito), buena baseline para comparar.
#   Contras: asume relaciones lineales entre features y log-odds.
#   max_iter=1000: suficiente para convergencia con 134 features.
# =========================================================================

modelos = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs"
    )
}

print("=" * 65)
print("PASO 3 — DEFINICIÓN DE MODELOS")
print("=" * 65)
print()
print("  MODELO 1: Random Forest")
print("    n_estimators  : 100 árboles")
print("    max_depth     : sin límite (crece hasta hojas puras)")
print("    class_weight  : balanced (mitiga desbalance 69/31)")
print()
print("  MODELO 2: Logistic Regression")
print("    max_iter      : 1000 (garantiza convergencia)")
print("    class_weight  : balanced")
print("    solver        : lbfgs (eficiente para muchas features)")
print()


# =========================================================================
# PASO 4 — ENTRENAMIENTO Y EVALUACIÓN
# =========================================================================
# Para cada modelo:
#   1. Entrenar con X_train, y_train
#   2. Predecir sobre X_test (datos nunca vistos)
#   3. Calcular métricas sobre y_test vs predicciones
#
# Métricas utilizadas:
#   - Accuracy   : % de predicciones correctas (todas las clases)
#   - Precision  : de los que predijo como exitosos, ¿cuántos lo son?
#   - Recall     : de los exitosos reales, ¿cuántos detectó?
#   - F1-score   : media armónica entre precision y recall
#   - ROC-AUC    : capacidad de discriminar entre clases (0.5=azar, 1=perfecto)
#   - Matriz de confusión: desglose de aciertos y errores por clase
# =========================================================================

print("=" * 65)
print("PASO 4 — ENTRENAMIENTO Y EVALUACIÓN")
print("=" * 65)
print()

resultados = {}

for nombre, modelo in modelos.items():
    print(f"  Entrenando {nombre}... ", end="", flush=True)
    modelo.fit(X_train, y_train)
    print("listo.")

    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    resultados[nombre] = {
        "modelo"    : modelo,
        "y_pred"    : y_pred,
        "y_proba"   : y_proba,
        "accuracy"  : accuracy_score(y_test, y_pred),
        "precision" : precision_score(y_test, y_pred),
        "recall"    : recall_score(y_test, y_pred),
        "f1"        : f1_score(y_test, y_pred),
        "roc_auc"   : roc_auc_score(y_test, y_proba),
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

print()


# =========================================================================
# PASO 5 — COMPARACIÓN DE RESULTADOS
# =========================================================================

print("=" * 65)
print("PASO 5 — COMPARACIÓN DE RESULTADOS")
print("=" * 65)
print()

# Tabla comparativa
metricas = ["accuracy", "precision", "recall", "f1", "roc_auc"]
nombres_metricas = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

print(f"  {'Métrica':<15} {'Random Forest':>16} {'Logistic Reg.':>16} {'Diferencia':>12}")
print("  " + "─" * 62)

rf_res  = resultados["Random Forest"]
lr_res  = resultados["Logistic Regression"]

for metrica, nombre in zip(metricas, nombres_metricas):
    val_rf = rf_res[metrica]
    val_lr = lr_res[metrica]
    diff   = val_rf - val_lr
    signo  = "+" if diff >= 0 else ""
    print(f"  {nombre:<15} {val_rf:>15.4f}  {val_lr:>15.4f}  {signo}{diff:>10.4f}")

print()

# Matrices de confusión
for nombre, res in resultados.items():
    cm = res["conf_matrix"]
    tn, fp, fn, tp = cm.ravel()
    print(f"  Matriz de confusión — {nombre}:")
    print(f"    {'':>20} Pred: No exitoso   Pred: Exitoso")
    print(f"    Real: No exitoso       {tn:>8}          {fp:>8}")
    print(f"    Real: Exitoso          {fn:>8}          {tp:>8}")
    print(f"    → Falsos positivos (FP): {fp}  |  Falsos negativos (FN): {fn}")
    print()

# Reporte completo por clase
for nombre, res in resultados.items():
    print(f"  Reporte detallado — {nombre}:")
    reporte = classification_report(
        y_test, res["y_pred"],
        target_names=["No exitoso", "Exitoso"]
    )
    for linea in reporte.split("\n"):
        print(f"    {linea}")
    print()


# =========================================================================
# PASO 6 — ANÁLISIS DE RESULTADOS
# =========================================================================
# Determinamos el mejor modelo y analizamos trade-offs.
# =========================================================================

print("=" * 65)
print("PASO 6 — ANÁLISIS DE RESULTADOS")
print("=" * 65)
print()

# Determinar ganador por F1 (más equilibrado para clases desbalanceadas)
f1_rf = rf_res["f1"]
f1_lr = lr_res["f1"]
ganador = "Random Forest" if f1_rf >= f1_lr else "Logistic Regression"

print(f"  ★ Mejor modelo (por F1-Score): {ganador}")
print()
print("  CRITERIO DE SELECCIÓN:")
print("  Se usa F1-Score como métrica principal porque el dataset")
print("  está desbalanceado (69% exitosos). El F1 pondera precision")
print("  y recall, siendo más informativo que accuracy sola.")
print()
print("  TRADE-OFFS:")
print("  • Random Forest captura relaciones no lineales entre tags")
print("    (ej: la combinación '2d + indie' puede ser más poderosa")
print("    que cada tag por separado). Más lento de entrenar.")
print("  • Logistic Regression asume independencia entre tags.")
print("    Más rápido e interpretable: cada coeficiente indica")
print("    cuánto suma o resta un tag a la probabilidad de éxito.")
print()
print("  ERRORES MÁS IMPORTANTES EN ESTE CONTEXTO:")
print("  Los Falsos Negativos (FN) son más costosos: predecir que")
print("  un juego NO será exitoso cuando sí lo sería implica perder")
print("  una oportunidad de inversión o desarrollo.")
print("  → Un recall alto en la clase 'Exitoso' es deseable.")
print()
print("  LIMITACIONES DEL EXPERIMENTO:")
print("  1. Los tags son información subjetiva asignada por usuarios.")
print("  2. El éxito depende de factores no capturados: marketing,")
print("     precio, competencia, fecha de lanzamiento.")
print("  3. El dataset puede tener sesgo temporal (juegos recientes")
print("     tienen más reviews que juegos antiguos).")
print()


# =========================================================================
# PASO 7 — FEATURES MÁS IMPORTANTES
# =========================================================================
# ¿Qué tags son los mejores predictores de éxito?
# =========================================================================

print("=" * 65)
print("PASO 7 — TAGS MÁS IMPORTANTES PARA PREDECIR ÉXITO")
print("=" * 65)
print()

# Random Forest: feature importances nativas
rf_model = resultados["Random Forest"]["modelo"]
importancias = pd.Series(
    rf_model.feature_importances_,
    index=tag_cols
).sort_values(ascending=False)

print("  [Random Forest — Top 10 tags que más predicen éxito]")
print(f"  {'Tag':<30} {'Importancia':>12}")
print("  " + "─" * 44)
for tag, imp in importancias.head(10).items():
    nombre = tag.replace("tag_", "").replace("_", " ")
    barra  = "█" * int(imp * 500)
    print(f"  {nombre:<30} {imp:>10.4f}  {barra}")
print()

# Logistic Regression: coeficientes
lr_model = resultados["Logistic Regression"]["modelo"]
coefs = pd.Series(
    lr_model.coef_[0],
    index=tag_cols
)

print("  [Logistic Regression — Tags que MÁS suman probabilidad de éxito]")
print(f"  {'Tag':<30} {'Coeficiente':>12}")
print("  " + "─" * 44)
for tag, coef in coefs.sort_values(ascending=False).head(10).items():
    nombre = tag.replace("tag_", "").replace("_", " ")
    print(f"  {nombre:<30} {coef:>+12.4f}")
print()

print("  [Logistic Regression — Tags que MÁS RESTAN probabilidad de éxito]")
print(f"  {'Tag':<30} {'Coeficiente':>12}")
print("  " + "─" * 44)
for tag, coef in coefs.sort_values(ascending=True).head(10).items():
    nombre = tag.replace("tag_", "").replace("_", " ")
    print(f"  {nombre:<30} {coef:>+12.4f}")
print()


# =========================================================================
# PASO 8 — EJEMPLOS DE USO (mínimo 3 requeridos)
# =========================================================================
# Aplicamos el mejor modelo a casos concretos.
# Mostramos cómo se interpreta la salida del modelo.
# =========================================================================

print("=" * 65)
print("PASO 8 — EJEMPLOS DE USO DEL MODELO")
print("=" * 65)
print()
print(f"  Modelo usado: {ganador}")
print()

mejor_modelo = resultados[ganador]["modelo"]

def predecir_juego(nombre_ejemplo, tags_activos, modelo, todas_las_tags):
    """
    Predice si un juego será exitoso dados sus tags.

    Parámetros:
        nombre_ejemplo : nombre descriptivo del juego ejemplo
        tags_activos   : lista de tags que tiene el juego (sin prefijo 'tag_')
        modelo         : modelo entrenado
        todas_las_tags : lista completa de columnas tag_*
    """
    # Construir vector de features
    vector = pd.DataFrame(
        [[0] * len(todas_las_tags)],
        columns=todas_las_tags
    )
    for tag in tags_activos:
        col = f"tag_{tag.replace(' ', '_')}"
        if col in vector.columns:
            vector[col] = 1

    prediccion = modelo.predict(vector)[0]
    probabilidad = modelo.predict_proba(vector)[0][1]
    etiqueta = "✅ EXITOSO" if prediccion == 1 else "❌ NO EXITOSO"

    tags_encontrados = [t for t in tags_activos if f"tag_{t.replace(' ', '_')}" in todas_las_tags]
    tags_no_encontrados = [t for t in tags_activos if f"tag_{t.replace(' ', '_')}" not in todas_las_tags]

    print(f"  ─── Ejemplo: {nombre_ejemplo} ───")
    print(f"  Tags ingresados : {', '.join(tags_encontrados)}")
    if tags_no_encontrados:
        print(f"  Tags ignorados  : {', '.join(tags_no_encontrados)} (no están en el dataset)")
    print(f"  Predicción      : {etiqueta}")
    print(f"  Probabilidad    : {probabilidad*100:.1f}% de ser exitoso")
    if probabilidad >= 0.7:
        print("  Interpretación  : Alta probabilidad de éxito. El perfil de tags")
        print("                    es consistente con juegos exitosos en Steam.")
    elif probabilidad >= 0.4:
        print("  Interpretación  : Probabilidad moderada. El juego podría ser exitoso")
        print("                    pero su combinación de tags no es determinante.")
    else:
        print("  Interpretación  : Baja probabilidad de éxito. El perfil de tags")
        print("                    se asocia más a juegos no exitosos en Steam.")
    print()


# Ejemplo 1: Juego indie 2D tipo puzzle
predecir_juego(
    nombre_ejemplo="Juego indie 2D tipo puzzle",
    tags_activos=["indie", "2d", "puzzle", "pixel_graphics", "cute"],
    modelo=mejor_modelo,
    todas_las_tags=tag_cols
)

# Ejemplo 2: Shooter multijugador realista
predecir_juego(
    nombre_ejemplo="Shooter multijugador realista",
    tags_activos=["multiplayer", "first_person", "realistic", "pvp", "free_to_play"],
    modelo=mejor_modelo,
    todas_las_tags=tag_cols
)

# Ejemplo 3: RPG de mundo abierto en tercera persona
predecir_juego(
    nombre_ejemplo="RPG mundo abierto en tercera persona",
    tags_activos=["rpg", "open_world", "third_person", "story_rich", "action"],
    modelo=mejor_modelo,
    todas_las_tags=tag_cols
)

# Ejemplo 4: Juego de estrategia retro con historia
predecir_juego(
    nombre_ejemplo="Estrategia retro con historia",
    tags_activos=["strategy", "retro", "story_rich", "indie", "difficult"],
    modelo=mejor_modelo,
    todas_las_tags=tag_cols
)


# =========================================================================
# RESUMEN FINAL
# =========================================================================

print("=" * 65)
print("RESUMEN FINAL")
print("=" * 65)
print()
print(f"  Problema        : Predecir si un juego Steam será exitoso")
print(f"  Features        : {len(tag_cols)} tags binarios")
print(f"  Dataset         : {len(df):,} juegos  ({pct_exitosos:.1f}% exitosos)")
print(f"  División        : 80% train / 20% test (estratificado)")
print()
print(f"  {'Modelo':<22} {'Accuracy':>9} {'F1':>9} {'ROC-AUC':>9}")
print("  " + "─" * 52)
for nombre, res in resultados.items():
    print(f"  {nombre:<22} {res['accuracy']:>9.4f} {res['f1']:>9.4f} {res['roc_auc']:>9.4f}")
print()
print(f"  ★ Modelo recomendado : {ganador}")
print(f"    Justificación      : Mayor F1-Score, mejor balance entre")
print(f"    precision y recall en un dataset con clases desbalanceadas.")
print()
print("=" * 65)
