"""
SCRIPT 2 – ANÁLISIS EXPLORATORIO DE DATOS (EDA)
Steam Games: ¿Será un juego exitoso en base a sus tags?

Requisitos:
    pip install pandas numpy matplotlib seaborn

Uso:
    python 02_eda.py

Input:  SteamGames_prepared.csv  (generado por 01_preparacion.py)
Output: 4 imágenes PNG con los gráficos del EDA
    - eda_fig1_overview.png
    - eda_fig2_dist_outliers.png
    - eda_fig3_relaciones.png
    - eda_fig4_calidad.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # sin pantalla; cambiar a 'TkAgg' si quieres ventana emergente
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE = 'SteamGames_prepared.csv'

BLUE   = '#4A90D9'
ORANGE = '#E07B39'
GREEN  = '#5BAD72'
RED    = '#D94A4A'
PURPLE = '#7B5BAD'
TEAL   = '#3AADA8'
BG     = '#F7F9FC'
DARK   = '#1E2A3A'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor':   DARK, 'axes.labelcolor': DARK,
    'xtick.color':      DARK, 'ytick.color':     DARK,
    'text.color':       DARK, 'font.family':     'DejaVu Sans',
    'axes.spines.top':  False, 'axes.spines.right': False,
})

# ─────────────────────────────────────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_FILE)
df['ReleaseDate'] = pd.to_datetime(df['ReleaseDate'], errors='coerce')
df['ReleaseYear'] = df['ReleaseDate'].dt.year
tag_cols = [c for c in df.columns if c.startswith('tag_')]
df['n_tags'] = df[tag_cols].sum(axis=1)
print(f"[CARGA] {df.shape[0]} filas × {df.shape[1]} columnas | {len(tag_cols)} tag cols")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURA 1 – VISIÓN GENERAL
# ═════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 11), facecolor=BG)
fig.suptitle('Steam Games – EDA  |  Visión General',
             fontsize=16, fontweight='bold', color=DARK, y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 1a. Valores faltantes ANTES de limpiar
ax = fig.add_subplot(gs[0, 0])
missing_before = {
    'Name': 25, 'Tags': 229, 'tags': 287, 'Genres': 406,
    'Developers': 234, 'Publishers': 165, 'MemoryRequirement': 45
}
bars = ax.barh(list(missing_before.keys()), list(missing_before.values()),
               color=ORANGE, edgecolor='none', height=0.6)
ax.set_title('Valores Faltantes (antes)', fontsize=10, fontweight='bold')
ax.set_xlabel('N° nulos')
for bar, v in zip(bars, missing_before.values()):
    ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
            str(v), va='center', fontsize=8)
ax.tick_params(axis='y', labelsize=8)

# 1b. Distribución ReviewScore
ax = fig.add_subplot(gs[0, 1])
vc = df['ReviewScore'].value_counts().sort_index()
colors_rs = [RED if i < 7 else GREEN for i in vc.index]
bars2 = ax.bar(vc.index, vc.values, color=colors_rs, edgecolor='none', width=0.7)
ax.set_title('Distribución ReviewScore', fontsize=10, fontweight='bold')
ax.set_xlabel('ReviewScore')
ax.set_ylabel('N° juegos')
ax.axvline(6.5, color=DARK, linestyle='--', linewidth=1.2, label='Corte éxito (≥7)')
ax.legend(fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
            f'{bar.get_height():,.0f}', ha='center', fontsize=7)

# 1c. Variable objetivo (pie)
ax = fig.add_subplot(gs[0, 2])
counts = df['Exitoso'].value_counts()
ax.pie(counts, labels=['Exitoso\n(≥7)', 'No exitoso\n(<7)'],
       autopct='%1.1f%%', colors=[GREEN, RED], startangle=90,
       wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax.set_title('Variable Objetivo: Exitoso', fontsize=10, fontweight='bold')

# 1d. Distribución precio
ax = fig.add_subplot(gs[1, 0])
ax.hist(df['price'].clip(upper=60), bins=40, color=BLUE, edgecolor='none', alpha=0.85)
ax.set_title('Distribución Precio (cap. $60)', fontsize=10, fontweight='bold')
ax.set_xlabel('Precio (USD)')
ax.set_ylabel('N° juegos')
ax.axvline(df['price'].median(), color=ORANGE, linestyle='--', linewidth=1.5,
           label=f'Mediana ${df["price"].median()}')
ax.legend(fontsize=8)

# 1e. Juegos lanzados por año
ax = fig.add_subplot(gs[1, 1])
year_counts = df[df['ReleaseYear'].between(2005, 2024)]['ReleaseYear'].value_counts().sort_index()
ax.bar(year_counts.index, year_counts.values, color=PURPLE, edgecolor='none', width=0.7)
ax.set_title('Juegos Lanzados por Año', fontsize=10, fontweight='bold')
ax.set_xlabel('Año')
ax.set_ylabel('N° juegos')
ax.tick_params(axis='x', rotation=45, labelsize=7)

# 1f. Top 15 tags más frecuentes
ax = fig.add_subplot(gs[1, 2])
tag_freq = df[tag_cols].sum().sort_values(ascending=True).tail(15)
tag_labels = [c.replace('tag_', '').replace('_', ' ') for c in tag_freq.index]
ax.barh(tag_labels, tag_freq.values, color=BLUE, edgecolor='none', height=0.7)
ax.set_title('Top 15 Tags Más Frecuentes', fontsize=10, fontweight='bold')
ax.set_xlabel('N° juegos')
ax.tick_params(axis='y', labelsize=8)

plt.savefig('eda_fig1_overview.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('[OK] eda_fig1_overview.png')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURA 2 – DISTRIBUCIONES Y OUTLIERS
# ═════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 11), facecolor=BG)
fig.suptitle('Steam Games – Distribuciones y Detección de Outliers',
             fontsize=16, fontweight='bold', color=DARK, y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 2a. Boxplot precio
ax = fig.add_subplot(gs[0, 0])
ax.boxplot(df['price'], vert=False, patch_artist=True,
           boxprops=dict(facecolor=BLUE, alpha=0.7),
           medianprops=dict(color=ORANGE, linewidth=2),
           flierprops=dict(marker='o', color=RED, alpha=0.3, markersize=3))
ax.set_title('Outliers – Precio', fontsize=10, fontweight='bold')
ax.set_xlabel('Precio (USD)')
q1, q3 = df['price'].quantile([.25, .75])
upper_fence = q3 + 1.5 * (q3 - q1)
outliers_price = (df['price'] > upper_fence).sum()
ax.text(0.98, 0.85, f'Outliers: {outliers_price}\n(>{upper_fence:.1f} USD)',
        transform=ax.transAxes, ha='right', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=RED, alpha=0.15))

# 2b. Boxplot TotalReviews (log)
ax = fig.add_subplot(gs[0, 1])
total_rev_log = np.log1p(df['TotalReviews'])
ax.boxplot(total_rev_log, vert=False, patch_artist=True,
           boxprops=dict(facecolor=PURPLE, alpha=0.7),
           medianprops=dict(color=ORANGE, linewidth=2),
           flierprops=dict(marker='o', color=RED, alpha=0.3, markersize=3))
ax.set_title('Outliers – Total Reseñas (log1p)', fontsize=10, fontweight='bold')
ax.set_xlabel('log(1 + TotalReviews)')
q1r, q3r = total_rev_log.quantile([.25, .75])
out_rev = (total_rev_log > q3r + 1.5 * (q3r - q1r)).sum()
ax.text(0.98, 0.85, f'Outliers: {out_rev}',
        transform=ax.transAxes, ha='right', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=RED, alpha=0.15))

# 2c. MemoryRequirement
ax = fig.add_subplot(gs[0, 2])
ax.hist(df['MemoryRequirement'].clip(upper=16384), bins=30,
        color=TEAL, edgecolor='none', alpha=0.85)
ax.set_title('Requisito de Memoria (MB, cap. 16GB)', fontsize=10, fontweight='bold')
ax.set_xlabel('MB')
ax.set_ylabel('N° juegos')
ax.axvline(df['MemoryRequirement'].median(), color=ORANGE, linestyle='--', linewidth=1.5,
           label=f'Mediana: {df["MemoryRequirement"].median():.0f} MB')
ax.legend(fontsize=8)

# 2d. Precio por ReviewScore (violín)
ax = fig.add_subplot(gs[1, 0])
scores = sorted(df['ReviewScore'].unique())
data_violin = [df[df['ReviewScore'] == s]['price'].clip(upper=40).values for s in scores]
parts = ax.violinplot(data_violin, positions=scores, showmedians=True, showextrema=False)
for pc in parts['bodies']:
    pc.set_facecolor(BLUE); pc.set_alpha(0.6)
parts['cmedians'].set_color(ORANGE); parts['cmedians'].set_linewidth(2)
ax.set_title('Precio vs ReviewScore', fontsize=10, fontweight='bold')
ax.set_xlabel('ReviewScore')
ax.set_ylabel('Precio USD (cap. $40)')

# 2e. PositiveRatio por clase
ax = fig.add_subplot(gs[1, 1])
df_r = df[df['PositiveRatio'].notna()]
ax.hist(df_r[df_r['Exitoso'] == 1]['PositiveRatio'], bins=40,
        alpha=0.6, color=GREEN, label='Exitoso', density=True)
ax.hist(df_r[df_r['Exitoso'] == 0]['PositiveRatio'], bins=40,
        alpha=0.6, color=RED, label='No exitoso', density=True)
ax.set_title('PositiveRatio por Clase', fontsize=10, fontweight='bold')
ax.set_xlabel('Ratio de reseñas positivas')
ax.set_ylabel('Densidad')
ax.legend(fontsize=8)

# 2f. CPU_tier vs Exitoso
ax = fig.add_subplot(gs[1, 2])
ct_pct = (df.groupby(['CPU_tier', 'Exitoso']).size()
           .unstack(fill_value=0)
           .pipe(lambda x: x.div(x.sum(axis=1), axis=0) * 100))
ct_pct.plot(kind='bar', ax=ax, color=[RED, GREEN], edgecolor='none', width=0.6)
ax.set_title('CPU Tier vs Exitoso (%)', fontsize=10, fontweight='bold')
ax.set_xlabel('CPU Tier')
ax.set_ylabel('%')
ax.legend(['No exitoso', 'Exitoso'], fontsize=8)
ax.tick_params(axis='x', rotation=0)

plt.savefig('eda_fig2_dist_outliers.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('[OK] eda_fig2_dist_outliers.png')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURA 3 – RELACIÓN ENTRE VARIABLES
# ═════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 13), facecolor=BG)
fig.suptitle('Steam Games – Relación entre Variables y Tags vs Éxito',
             fontsize=16, fontweight='bold', color=DARK, y=0.99)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

# 3a. Heatmap correlación
ax = fig.add_subplot(gs[0, :2])
num_cols = ['ReviewScore', 'PositiveRatio', 'price', 'MemoryRequirement',
            'TotalReviews', 'CPU_GHz', 'CPU_tier', 'ReleaseYear', 'Exitoso']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            linewidths=0.5, annot_kws={'size': 8},
            cbar_kws={'shrink': 0.8})
ax.set_title('Correlación Variables Numéricas', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', rotation=0, labelsize=8)

# 3b. Tags: diferencia de tasa de éxito (top/bottom 10)
ax = fig.add_subplot(gs[0, 2])
tag_success = {
    col.replace('tag_', '').replace('_', ' '): {
        'diff': df[df[col] == 1]['Exitoso'].mean() - df[df[col] == 0]['Exitoso'].mean()
    }
    for col in tag_cols
}
ts_df = pd.DataFrame(tag_success).T.sort_values('diff')
combined = pd.concat([ts_df.head(10), ts_df.tail(10)])
colors_bar = [RED if v < 0 else GREEN for v in combined['diff']]
ax.barh(combined.index, combined['diff'] * 100, color=colors_bar, edgecolor='none', height=0.7)
ax.axvline(0, color=DARK, linewidth=1)
ax.set_title('Tags: Δ tasa éxito\n(con tag vs sin tag)', fontsize=10, fontweight='bold')
ax.set_xlabel('Δ% Exitoso')
ax.tick_params(axis='y', labelsize=7)

# 3c. N° de tags por juego vs Exitoso
ax = fig.add_subplot(gs[1, 0])
ax.hist(df[df['Exitoso'] == 1]['n_tags'], bins=20,
        alpha=0.6, color=GREEN, density=True, label='Exitoso')
ax.hist(df[df['Exitoso'] == 0]['n_tags'], bins=20,
        alpha=0.6, color=RED, density=True, label='No exitoso')
ax.set_title('N° de Tags por Juego vs Éxito', fontsize=10, fontweight='bold')
ax.set_xlabel('N° tags asignados')
ax.set_ylabel('Densidad')
ax.legend(fontsize=8)

# 3d. Tasa de éxito: gratis vs pagado
ax = fig.add_subplot(gs[1, 1])
df['es_gratis'] = (df['price'] == 0).astype(int)
g = df.groupby('es_gratis')['Exitoso'].mean() * 100
ax.bar(['Pagado', 'Gratis'], g.values, color=[BLUE, ORANGE], edgecolor='none', width=0.5)
ax.set_title('Tasa de Éxito: Gratis vs Pagado', fontsize=10, fontweight='bold')
ax.set_ylabel('% Exitoso')
ax.set_ylim(0, 100)
for i, v in enumerate(g.values):
    ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

# 3e. TotalReviews (log) vs Exitoso
ax = fig.add_subplot(gs[1, 2])
ax.hist(np.log1p(df[df['Exitoso'] == 1]['TotalReviews']),
        bins=35, alpha=0.6, color=GREEN, density=True, label='Exitoso')
ax.hist(np.log1p(df[df['Exitoso'] == 0]['TotalReviews']),
        bins=35, alpha=0.6, color=RED, density=True, label='No exitoso')
ax.set_title('Total Reseñas (log1p) vs Exitoso', fontsize=10, fontweight='bold')
ax.set_xlabel('log(1 + TotalReviews)')
ax.set_ylabel('Densidad')
ax.legend(fontsize=8)

plt.savefig('eda_fig3_relaciones.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('[OK] eda_fig3_relaciones.png')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURA 4 – CALIDAD DEL DATASET
# ═════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 10), facecolor=BG)
fig.suptitle('Steam Games – Observaciones de Calidad del Dataset',
             fontsize=16, fontweight='bold', color=DARK, y=0.99)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

# 4a. Flujo de limpieza
ax = fig.add_subplot(gs[0, 0])
steps  = ['Raw\n(29,905)', 'Solo games\n(22,699)', 'Sin dup.\nAppid (22,588)',
          'Sin Name\nnulo (22,563)', 'Sin tags\nnulo (22,531)']
counts_flow = [29905, 22699, 22588, 22563, 22531]
colors_flow = [BLUE, PURPLE, PURPLE, ORANGE, GREEN]
ax.barh(steps[::-1], counts_flow[::-1], color=colors_flow[::-1], edgecolor='none', height=0.6)
ax.set_title('Flujo de Limpieza', fontsize=10, fontweight='bold')
ax.set_xlabel('N° filas')
for i, v in enumerate(counts_flow[::-1]):
    ax.text(v + 100, i, f'{v:,}', va='center', fontsize=8)

# 4b. Balance de clases
ax = fig.add_subplot(gs[0, 1])
exit_counts = df['Exitoso'].value_counts()
ax.bar(['No exitoso (0)', 'Exitoso (1)'], exit_counts.values,
       color=[RED, GREEN], edgecolor='none', width=0.5)
ax.set_title('Balance de Clases', fontsize=10, fontweight='bold')
ax.set_ylabel('N° juegos')
ratio = exit_counts[1] / exit_counts[0]
for i, v in enumerate(exit_counts.values):
    ax.text(i, v + 100, f'{v:,}\n({v/exit_counts.sum()*100:.1f}%)',
            ha='center', fontsize=9, fontweight='bold')
ax.text(0.5, 0.82, f'Ratio: {ratio:.2f}:1', transform=ax.transAxes,
        ha='center', fontsize=10, color=ORANGE,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=ORANGE, alpha=0.15))

# 4c. N° tags por juego
ax = fig.add_subplot(gs[0, 2])
vc = df['n_tags'].clip(upper=10).value_counts().sort_index()
ax.bar(vc.index, vc.values, color=TEAL, edgecolor='none', width=0.7)
ax.set_title('N° de Tags (freq≥50) por Juego', fontsize=10, fontweight='bold')
ax.set_xlabel('N° tags asignados')
ax.set_ylabel('N° juegos')
zero_tags = (df['n_tags'] == 0).sum()
ax.text(0.98, 0.95, f'Sin ningún tag: {zero_tags}',
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=ORANGE, alpha=0.2))

# 4d. Nulos remanentes
ax = fig.add_subplot(gs[1, 0])
final_nulls = df.isnull().sum()
final_nulls = final_nulls[final_nulls > 0].sort_values(ascending=True)
if len(final_nulls) > 0:
    ax.barh(final_nulls.index, final_nulls.values, color=ORANGE, edgecolor='none', height=0.6)
    for i, v in enumerate(final_nulls.values):
        ax.text(v + 1, i, str(v), va='center', fontsize=8)
else:
    ax.text(0.5, 0.5, '✓ Sin valores\nfaltantes', transform=ax.transAxes,
            ha='center', va='center', fontsize=14, color=GREEN, fontweight='bold')
ax.set_title('Nulos Remanentes (Dataset Final)', fontsize=10, fontweight='bold')

# 4e. Juegos sin reseñas (ReviewScore=0)
ax = fig.add_subplot(gs[1, 1])
score0 = (df['ReviewScore'] == 0).sum()
score_rest = len(df) - score0
ax.pie([score_rest, score0],
       labels=[f'Con reseñas\n({score_rest:,})', f'Sin reseñas\n({score0:,})'],
       colors=[BLUE, ORANGE], autopct='%1.1f%%', startangle=90,
       wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax.set_title('Juegos sin Reseñas (ReviewScore=0)', fontsize=10, fontweight='bold')

# 4f. Distribución temporal
ax = fig.add_subplot(gs[1, 2])
year_range = df['ReleaseYear'].dropna()
ax.hist(year_range, bins=range(int(year_range.min()), 2026),
        color=PURPLE, edgecolor='none', alpha=0.85)
ax.set_title('Distribución Temporal', fontsize=10, fontweight='bold')
ax.set_xlabel('Año de lanzamiento')
ax.set_ylabel('N° juegos')
ax.tick_params(axis='x', rotation=45, labelsize=7)

plt.savefig('eda_fig4_calidad.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('[OK] eda_fig4_calidad.png')

print('\n[LISTO] Todos los gráficos generados.')
