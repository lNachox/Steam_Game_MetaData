"""
SCRIPT 1 – PREPARACIÓN Y LIMPIEZA DEL DATASET
Steam Games: ¿Será un juego exitoso en base a sus tags?

Requisitos:
    pip install pandas numpy

Uso:
    python 01_preparacion.py

Input:  SteamGames_cleaned.csv   (en la misma carpeta)
Output: SteamGames_prepared.csv  (en la misma carpeta)
"""

import pandas as pd
import numpy as np
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE       = 'SteamGames_cleaned.csv'
OUTPUT_FILE      = 'SteamGames_prepared.csv'
TAG_FREQ_MIN     = 50      # Tags con menos de esta frecuencia se descartan
REVIEW_THRESHOLD = 7       # ReviewScore >= este valor = juego "exitoso"

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_FILE)
print(f"[CARGA] {df.shape[0]} filas × {df.shape[1]} columnas")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FILTRO: solo juegos (descartar DLC, music, mod, video)
# ─────────────────────────────────────────────────────────────────────────────

df = df[df['Type'] == 'game'].copy()
print(f"[FILTRO tipo=game] {df.shape[0]} filas")

# ─────────────────────────────────────────────────────────────────────────────
# 3. DUPLICADOS
#    Los duplicados por Appid son filas prácticamente idénticas (mismo juego
#    indexado dos veces). Se conserva la primera ocurrencia.
# ─────────────────────────────────────────────────────────────────────────────

antes = df.shape[0]
df = df.drop_duplicates(subset='Appid', keep='first').reset_index(drop=True)
print(f"[DUPLICADOS] eliminados {antes - df.shape[0]} filas → {df.shape[0]} filas")

# ─────────────────────────────────────────────────────────────────────────────
# 4. COLUMNAS DESCARTADAS
#    - Thumbnail    : URL, no aporta al modelo
#    - Description  : texto libre, requiere NLP por separado
#    - Tags         : versión cruda (mayúscula), usamos 'tags' (minúscula, limpia)
#    - Genres       : redundante con tags para nuestro objetivo
#    - OsRequirement: texto no estructurado, alta varianza
# ─────────────────────────────────────────────────────────────────────────────

df.drop(columns=['Thumbnail', 'Description', 'Tags', 'OsRequirement', 'Genres'],
        inplace=True)
print(f"[COLUMNAS] descartadas 5 columnas → {df.shape[1]} columnas restantes")

# ─────────────────────────────────────────────────────────────────────────────
# 5. VALORES FALTANTES
# ─────────────────────────────────────────────────────────────────────────────

# Name: sin nombre no podemos identificar el juego → eliminar fila
antes = df.shape[0]
df = df[df['Name'].notna()].copy()
print(f"[NULOS Name] eliminadas {antes - df.shape[0]} filas")

# Developers / Publishers: reemplazar con 'Unknown'
df['Developers'] = df['Developers'].fillna('Unknown')
df['Publishers']  = df['Publishers'].fillna('Unknown')

# MemoryRequirement: imputar con la mediana
mediana_mem = df['MemoryRequirement'].median()
df['MemoryRequirement'] = df['MemoryRequirement'].fillna(mediana_mem)
print(f"[NULOS MemoryRequirement] imputados con mediana ({mediana_mem:.0f} MB)")

# tags: sin tags no podemos hacer la predicción → eliminar fila
antes = df.shape[0]
df = df[df['tags'].notna()].copy()
print(f"[NULOS tags] eliminadas {antes - df.shape[0]} filas")

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRANSFORMACIONES
# ─────────────────────────────────────────────────────────────────────────────

# Fecha de lanzamiento como datetime + extraer año
df['ReleaseDate'] = pd.to_datetime(df['ReleaseDate'], errors='coerce')
df['ReleaseYear'] = df['ReleaseDate'].dt.year

# Variable objetivo binaria
df['Exitoso'] = (df['ReviewScore'] >= REVIEW_THRESHOLD).astype(int)

# Total de reseñas y ratio positivo
df['TotalReviews'] = df['PositiveReview'] + df['NegativeReview']
df['PositiveRatio'] = np.where(
    df['TotalReviews'] > 0,
    df['PositiveReview'] / df['TotalReviews'],
    np.nan
)

print(f"[TRANSFORMACIONES] ReleaseYear, Exitoso, TotalReviews, PositiveRatio creados")

# ─────────────────────────────────────────────────────────────────────────────
# 7. ONE-HOT ENCODING DE TAGS
#    Cada tag con frecuencia >= TAG_FREQ_MIN se convierte en columna binaria.
#    Nombre de columna: tag_<nombre_del_tag> (espacios y guiones → guión bajo)
# ─────────────────────────────────────────────────────────────────────────────

# Contar frecuencia de todos los tags
all_tags = []
for row in df['tags']:
    all_tags.extend([t.strip() for t in row.split(',')])
tag_counts = Counter(all_tags)

# Seleccionar tags con frecuencia suficiente
selected_tags = [t for t, c in tag_counts.items() if c >= TAG_FREQ_MIN]
print(f"[TAGS] {len(selected_tags)} tags seleccionados (freq >= {TAG_FREQ_MIN})")

# Crear columnas binarias
tag_data = {}
for tag in selected_tags:
    col = 'tag_' + tag.replace(' ', '_').replace('-', '_')
    tag_data[col] = df['tags'].apply(
        lambda x: 1 if tag in [t.strip() for t in x.split(',')] else 0
    )

df = pd.concat([df, pd.DataFrame(tag_data, index=df.index)], axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 8. RESUMEN FINAL Y GUARDADO
# ─────────────────────────────────────────────────────────────────────────────

tag_cols = [c for c in df.columns if c.startswith('tag_')]
print(f"\n{'='*50}")
print(f"DATASET FINAL: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"  - Columnas de tags: {len(tag_cols)}")
print(f"  - Distribución Exitoso:")
print(f"    Exitoso (1): {df['Exitoso'].sum():,}  ({df['Exitoso'].mean()*100:.1f}%)")
print(f"    No exitoso (0): {(df['Exitoso']==0).sum():,}  ({(1-df['Exitoso'].mean())*100:.1f}%)")
print(f"  - Nulos remanentes: {df.isnull().sum().sum()}")
print(f"{'='*50}")

df.to_csv(OUTPUT_FILE, index=False)
print(f"\n[OK] Dataset guardado en: {OUTPUT_FILE}")
