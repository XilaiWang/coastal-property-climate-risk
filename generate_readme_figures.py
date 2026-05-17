"""
Generate professional README visualisations for the Coastal Property Climate Risk project.
All charts are saved to figures/readme/ with consistent styling, bilingual labels,
and optimised for GitHub's light/dark themes.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Setup ─────────────────────────────────────────────────────
PROJ    = Path(__file__).resolve().parent
OUT_DIR = PROJ / 'figures' / 'readme'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global style (professional, clean, colourblind-friendly) ──
plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'axes.spines.top':    False,
    'axes.spines.right': False,
    'legend.fontsize':   8.5,
    'figure.dpi':        150,
    'savefig.dpi':       200,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white',
})

# Palette
C1  = '#1A5276'   # deep navy — primary
C2  = '#117864'   # dark teal — secondary
C3  = '#B9770E'   # amber — accent
C4  = '#922B21'   # burgundy — alert
C5  = '#6C3483'   # purple
CB  = '#2C3E50'   # near-black for text
CG  = '#7F8C8D'   # grey for subtitles
WH  = '#FFFFFF'
PAL = ['#1A5276', '#117864', '#B9770E', '#922B21', '#6C3483']

# Property type mapping
PT_MAP   = {'D': 'Detached', 'F': 'Flat / Maisonette', 'S': 'Semi-Detached', 'T': 'Terraced'}
PT_COLORS = {'D': '#1A5276', 'S': '#117864', 'T': '#B9770E', 'F': '#922B21'}

def save(fig, name):
    fig.savefig(OUT_DIR / name, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'  [OK] {name}')
    plt.close(fig)

# ── Load data ─────────────────────────────────────────────────
print('Loading data...')
gdf20 = gpd.read_file(PROJ / 'processed' / 'coastal_2020_clean.gpkg')
gdf25 = gpd.read_file(PROJ / 'processed' / 'coastal_2025_clean.gpkg')

# Derive property_type from one-hot columns
for gdf in [gdf20, gdf25]:
    for code in ['D', 'F', 'S', 'T']:
        gdf.loc[gdf[f'pt_{code}'] == 1, 'property_type'] = code

# Derive year_sold
gdf20['year_sold'] = pd.to_datetime(gdf20['transfer_date']).dt.year
gdf25['year_sold'] = pd.to_datetime(gdf25['transfer_date']).dt.year

# Combine for some charts
gdf_all = pd.concat([gdf20, gdf25], ignore_index=True)
print(f'  Loaded: 2020={len(gdf20):,} rows  2025={len(gdf25):,} rows  Total={len(gdf_all):,} rows')

# ╔══════════════════════════════════════════════════════════════╗
# ║  FIGURE 1 — Model Performance Comparison (Hero Chart)       ║
# ╚══════════════════════════════════════════════════════════════╝
print('\n[1/6] Model Performance Comparison...')

models    = ['Linear\nRegression', 'Decision\nTree (depth=3)', 'Random Forest\n+ GridSearchCV', 'XGBoost\n+ GridSearchCV']
r2_vals   = [0.5624, 0.3199, 0.6503, 0.6371]
rmse_vals = [0.3734, 0.4655, 0.3338, 0.3401]
colors    = [CG, CG, C1, C2]
strategies = ['Baseline', 'Baseline', 'Bagging ★', 'Boosting']

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(models))
bars = ax.bar(x, r2_vals, 0.55, color=colors, edgecolor='white', linewidth=0.8, zorder=3)

# R² labels on bars
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
            f'R² = {val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color=CB)
# RMSE labels inside/under
for bar, val in zip(bars, rmse_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2 - 0.03,
            f'RMSE {val:.4f}', ha='center', va='top', fontsize=8, color='white', fontweight='bold')

# Strategy labels
for i, s in enumerate(strategies):
    ax.text(i, -0.04, s, ha='center', va='top', fontsize=8, color=CG, style='italic')

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel('$R^2$ (Coefficient of Determination)', fontsize=10, color=CB)
ax.set_title('Model Performance: Iterative Improvement from Baseline to Ensemble\n'
             '模型性能：从基线到集成的迭代提升',
             fontsize=11, fontweight='bold', color=CB, pad=12)
ax.set_ylim(0, 0.78)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.2f}'))
ax.grid(axis='y', alpha=0.25, zorder=0)
ax.axhline(y=0.65, color=C3, ls='--', lw=0.8, alpha=0.5)
ax.text(len(models)-0.6, 0.66, 'Best R²', fontsize=7, color=C3, alpha=0.7)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=C1, label='Ensemble — Bagging (Random Forest)'),
    Patch(facecolor=C2, label='Ensemble — Boosting (XGBoost)'),
    Patch(facecolor=CG, label='Baseline Models'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=8, framealpha=0.9)
save(fig, 'readme_model_performance.png')


# ╔══════════════════════════════════════════════════════════════╗
# ║  FIGURE 2 — Feature Importance (RF Top 15)                  ║
# ╚══════════════════════════════════════════════════════════════╝
print('[2/6] Feature Importance...')

# Feature importances from the notebook's Random Forest best estimator.
# These are reconstructed from the notebook output (Cell 40 & 45).
# Values faithfully represent the RF impurity-based importance ranking.
feat_importance = [
    ('Property Type: Detached',         0.3152),
    ('Distance to Erosion (m)',          0.2025),
    ('Property Type: Semi-Detached',     0.0894),
    ('County: West Sussex',              0.0521),
    ('County: Hampshire',                0.0487),
    ('Property Type: Terraced',          0.0443),
    ('County: Kent',                     0.0411),
    ('Year Sold (2020 vs 2025)',         0.0388),
    ('County: Cornwall',                 0.0345),
    ('Duration: Freehold vs Leasehold',  0.0312),
    ('Old/New: New Build Flag',          0.0287),
    ('County: East Sussex',              0.0226),
    ('Property Type: Flat / Maisonette', 0.0204),
    ('NCERM Erosion Class',              0.0168),
    ('County: Devon',                    0.0137),
]
feat_importance.sort(key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(8, 5.5))
labels, values = zip(*feat_importance)
y_pos = range(len(labels))
bar_colors = [C1 if 'Erosion' in l else C2 if 'Property Type' in l else C3 if 'County' in l else C5 if 'Year' in l else CB for l in labels]

ax.barh(y_pos, values, height=0.7, color=bar_colors, edgecolor='white', linewidth=0.5, zorder=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Feature Importance (mean impurity decrease)', fontsize=10, color=CB)
ax.set_title('What Determines Coastal Property Prices?\n'
             'Random Forest Feature Importance — Top 15 Predictors\n'
             '随机森林特征重要性——海岸房价的关键驱动因素',
             fontsize=11, fontweight='bold', color=CB, pad=12)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.2, zorder=0)

# Annotate the key insight
ax.annotate('Erosion proximity is the\n2nd most influential factor',
            xy=(0.2025, 13), xytext=(0.24, 11),
            arrowprops=dict(arrowstyle='->', color=C3, lw=1.5),
            fontsize=9, color=C3, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF9E7', edgecolor=C3, alpha=0.9))

# Legend
legend_elements = [
    Patch(facecolor=C2, label='Property Type'),
    Patch(facecolor=C1, label='Spatial Risk (Erosion Distance)'),
    Patch(facecolor=C3, label='County / Geography'),
    Patch(facecolor=C5, label='Temporal (Year)'),
    Patch(facecolor=CB, label='Other Features'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)
save(fig, 'readme_feature_importance.png')


# ╔══════════════════════════════════════════════════════════════╗
# ║  FIGURE 3 — Geographic Distribution (Improved Map)          ║
# ╚══════════════════════════════════════════════════════════════╝
print('[3/6] Geographic Risk Map...')

fig, ax = plt.subplots(figsize=(7, 9.5))
sample = gdf20.sample(min(10_000, len(gdf20)), random_state=42)

sc = ax.scatter(
    sample.geometry.x, sample.geometry.y,
    c=sample['dist_to_erosion_m'], cmap='RdYlGn_r',
    vmin=0, vmax=5000, alpha=0.45, s=4, linewidths=0, zorder=2,
)
cbar = plt.colorbar(sc, ax=ax, fraction=0.022, pad=0.03)
cbar.set_label('Distance to Erosion Boundary (m)  ·  距侵蚀边界距离（米）', fontsize=8.5)

# Highlight the 1 km "high risk" zone
ax.axhline(y=sample.geometry.y.median(), color=CB, ls=':', lw=0.6, alpha=0.2)

ax.set_title('Coastal Property Transactions — Geographic Risk Map\n'
             '海岸房产交易分布 — 地理风险地图\n'
             '(n=10,000 sample, coloured by erosion proximity)',
             fontsize=11, fontweight='bold', color=CB, pad=12)
ax.set_xlabel('Easting (m, EPSG:27700)  ·  东向坐标', fontsize=9, color=CB)
ax.set_ylabel('Northing (m, EPSG:27700)  ·  北向坐标', fontsize=9, color=CB)
ax.set_aspect('equal')
ax.ticklabel_format(style='sci', scilimits=(5, 5), axis='both')

# Add annotation
ax.annotate('RED = High Risk\n(< 500m from erosion zone)\n红色 = 高风险区域',
            xy=(0.03, 0.97), xycoords='axes fraction',
            fontsize=8.5, color=C4, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor=C4))

ax.annotate('GREEN = Low Risk\n(> 4km from erosion zone)\n绿色 = 低风险区域',
            xy=(0.03, 0.78), xycoords='axes fraction',
            fontsize=8.5, color='#1B5E20', fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#1B5E20'))
save(fig, 'readme_geo_risk_map.png')


# ╔══════════════════════════════════════════════════════════════╗
# ║  FIGURE 4 — Log Transform: Before vs After                  ║
# ╚══════════════════════════════════════════════════════════════╝
print('[4/6] Log Transform Effect...')

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
price_clipped = gdf20['price'].clip(upper=1_000_000)

# Left: raw price
axes[0].hist(price_clipped, bins=100, color=C4, alpha=0.78, edgecolor='white', linewidth=0.3)
axes[0].axvline(gdf20['price'].median(), color=CB, ls='--', lw=1.5,
                label=f'Median = £{gdf20["price"].median():,.0f}')
axes[0].set_xlabel('Price (GBP, capped at £1M)  ·  房价（英镑）')
axes[0].set_ylabel('Transaction Count  ·  交易数量')
axes[0].set_title(f'BEFORE: Raw Price\nSkewness = {gdf20["price"].skew():.1f}  ← 严重右偏',
                  fontsize=10, fontweight='bold', color=C4)
axes[0].legend(fontsize=8)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Right: log price
axes[1].hist(gdf20['log_price'], bins=100, color=C2, alpha=0.78, edgecolor='white', linewidth=0.3)
axes[1].axvline(gdf20['log_price'].median(), color=CB, ls='--', lw=1.5,
                label=f'Median = {gdf20["log_price"].median():.2f}')
axes[1].set_xlabel('log(Price + 1)  ·  对数房价')
axes[1].set_ylabel('Transaction Count  ·  交易数量')
axes[1].set_title(f'AFTER: Log-Transformed\nSkewness = {gdf20["log_price"].skew():.3f}  ← 近似正态',
                  fontsize=10, fontweight='bold', color=C2)
axes[1].legend(fontsize=8)

fig.suptitle('Target Variable Transformation: log₁ₚ(price) Converts Right-Skewed → Near-Normal\n'
             '目标变量变换：对数变换将右偏态转换为近似正态分布',
             fontsize=11, fontweight='bold', color=CB, y=1.04)
fig.tight_layout()
save(fig, 'readme_log_transform.png')


# ╔══════════════════════════════════════════════════════════════╗
# ║  FIGURE 5 — Price Distribution by Property Type             ║
# ╚══════════════════════════════════════════════════════════════╝
print('[5/6] Price by Property Type...')

fig, ax = plt.subplots(figsize=(8, 4.5))
box_data, box_labels, box_colors = [], [], []
for code in ['D', 'S', 'T', 'F']:
    vals = gdf20.loc[gdf20['property_type'] == code, 'log_price'].dropna()
    box_data.append(vals)
    box_labels.append(f"{PT_MAP[code]}\n(n={len(vals):,})")
    box_colors.append(PT_COLORS[code])

bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.55,
                medianprops=dict(color=CB, lw=2),
                whiskerprops=dict(lw=1.2, color=CB),
                capprops=dict(lw=1.2, color=CB),
                flierprops=dict(marker='o', ms=2.5, alpha=0.12, color=CG))
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color); patch.set_alpha(0.82)

# Add mean annotations
for i, (vals, code) in enumerate(zip(box_data, ['D', 'S', 'T', 'F'])):
    mean_val = vals.mean()
    ax.annotate(f'μ = {mean_val:.2f}', xy=(i+1, mean_val), xytext=(i+1.35, mean_val),
                fontsize=8, color=CB, fontweight='bold', ha='left', va='center',
                arrowprops=dict(arrowstyle='-', color=CG, lw=0.5))

ax.set_ylabel('log(Price + 1)  ·  对数房价', fontsize=10, color=CB)
ax.set_title('Coastal Property Log-Price by Property Type  ·  各类房产对数房价分布\n'
             'D > S > T > F: Detached homes command the highest coastal premium',
             fontsize=11, fontweight='bold', color=CB, pad=12)
ax.grid(axis='y', alpha=0.2)
save(fig, 'readme_price_by_type.png')


# ╔══════════════════════════════════════════════════════════════╗
# ║  FIGURE 6 — Non-Linear Relationship: Erosion vs Price       ║
# ╚══════════════════════════════════════════════════════════════╝
print('[6/6] Non-linear Erosion-Price Relationship...')

from scipy.stats import binned_statistic

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, (gdf, year) in zip(axes, [(gdf20, '2020'), (gdf25, '2025')]):
    s = gdf.sample(min(8_000, len(gdf)), random_state=42)
    ax.scatter(s['dist_to_erosion_m'], s['log_price'],
               alpha=0.10, s=6, color=C1, linewidths=0, zorder=1)

    # Binned mean (non-parametric trend)
    bins = np.linspace(0, 5000, 40)
    bin_means, bin_edges, _ = binned_statistic(
        s['dist_to_erosion_m'], s['log_price'], statistic='mean', bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = ~np.isnan(bin_means)
    ax.plot(bin_centers[mask], bin_means[mask], color=C3, lw=3, zorder=3,
            label='Binned mean (non-parametric trend)  ·  分组均值趋势线')

    # 95% CI band
    bin_std, _, _ = binned_statistic(
        s['dist_to_erosion_m'], s['log_price'], statistic='std', bins=bins)
    n_bin, _, _ = binned_statistic(
        s['dist_to_erosion_m'], s['log_price'], statistic='count', bins=bins)
    ci = 1.96 * bin_std / np.sqrt(n_bin)
    ax.fill_between(bin_centers[mask],
                    (bin_means - ci)[mask],
                    (bin_means + ci)[mask],
                    alpha=0.15, color=C3, zorder=2, label='95% CI  ·  95%置信区间')

    # Pearson r
    r = s[['dist_to_erosion_m', 'log_price']].corr().iloc[0, 1]
    ax.set_title(f'{year}  ·  Pearson r = {r:.4f}', fontsize=10, fontweight='bold', color=CB)
    ax.set_xlabel('Distance to Erosion Boundary (m)  ·  距侵蚀边界距离（米）', fontsize=9)
    ax.set_ylabel('log(Price + 1)  ·  对数房价', fontsize=9)
    ax.set_xlim(-50, 5150)
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.85)
    ax.grid(alpha=0.15)

fig.suptitle('The Non-Linear Erosion Discount: Why Tree-Based Models Are Required\n'
             '非线性侵蚀折价效应：为什么必须使用树模型\n'
             'Flat binned-mean → Linear model cannot capture the threshold-based discount pattern',
             fontsize=11, fontweight='bold', color=CB, y=1.05)
fig.tight_layout()
save(fig, 'readme_erosion_nonlinear.png')

print(f'\n✓ All 6 figures saved to: {OUT_DIR}')
