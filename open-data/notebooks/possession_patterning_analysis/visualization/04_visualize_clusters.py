"""
Phase 4: cluster grid and centroid overview visualizations.
Run: python 04_visualize_clusters.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mplsoccer import Pitch
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PITCH_LENGTH = 120
PITCH_WIDTH  = 80
N_SAMPLES    = 30
RANDOM_STATE = 42

STRATA_ORDER = ['Defensive', 'Midfield', 'Attacking']

# ==============================================================================
# PATHS
# ==============================================================================

script_dir    = os.path.dirname(os.path.abspath(__file__))
project_root  = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
stats_dir     = os.path.join(project_root, 'results')
input_dir     = os.path.join(stats_dir, 'possession_patterns')
viz_dir       = os.path.join(project_root, 'visualizations', 'possession_patterning_analysis')
os.makedirs(viz_dir, exist_ok=True)

# ==============================================================================
# LOAD DATA
# ==============================================================================

print("Loading data...")
clustered_df = pd.read_csv(os.path.join(input_dir, 'possessions_clustered.csv'))
profiles_df  = pd.read_csv(os.path.join(input_dir, 'cluster_profiles.csv'))

N_RESAMPLE = 0
while f'p{N_RESAMPLE}_x' in clustered_df.columns:
    N_RESAMPLE += 1

clustered_df['cluster_id'] = clustered_df['cluster_id'].astype(str)
profiles_df['cluster_id'] = profiles_df['cluster_id'].astype(str)
clustered_df = clustered_df[clustered_df['cluster_id'] != '-1'].copy()
n_clusters = clustered_df['cluster_id'].nunique()
print(f"  {len(clustered_df):,} assigned possessions, {n_clusters} clusters")

pva_values = profiles_df['mean_total_pva'].values
pva_norm = Normalize(vmin=pva_values.min(), vmax=pva_values.max())
pva_cmap = plt.cm.RdYlGn

# ==============================================================================
# HELPERS
# ==============================================================================

def to_pitch_coords(row_or_array):
    if hasattr(row_or_array, '__getitem__') and not isinstance(row_or_array, np.ndarray):
        xs = [row_or_array[f'p{i}_x'] * PITCH_LENGTH for i in range(N_RESAMPLE)]
        ys = [row_or_array[f'p{i}_y'] * PITCH_WIDTH  for i in range(N_RESAMPLE)]
    else:
        xs = [row_or_array[2*i] * PITCH_LENGTH for i in range(N_RESAMPLE)]
        ys = [row_or_array[2*i+1] * PITCH_WIDTH for i in range(N_RESAMPLE)]
    return np.array(xs), np.array(ys)

def get_centroid_coords(cid):
    row = profiles_df[profiles_df['cluster_id'] == cid].iloc[0]
    xs = [row[f'centroid_p{i}_x'] * PITCH_LENGTH for i in range(N_RESAMPLE)]
    ys = [row[f'centroid_p{i}_y'] * PITCH_WIDTH  for i in range(N_RESAMPLE)]
    return np.array(xs), np.array(ys)

def draw_path_arrows(ax, xs, ys, color, n_arrows=2, size=12, zorder=5):
    total_pts = len(xs)
    if total_pts < 3:
        return
    positions = np.linspace(0.25, 0.75, n_arrows)
    for frac in positions:
        idx = int(frac * (total_pts - 1))
        idx = max(1, min(idx, total_pts - 2))
        dx = xs[idx + 1] - xs[idx - 1]
        dy = ys[idx + 1] - ys[idx - 1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        ax.annotate('', xy=(xs[idx] + dx * 0.01, ys[idx] + dy * 0.01),
                    xytext=(xs[idx], ys[idx]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                    mutation_scale=size),
                    zorder=zorder)

# ==============================================================================
# PER-STRATUM CLUSTER GRIDS
# ==============================================================================

rng = np.random.RandomState(RANDOM_STATE)

for stratum_name in STRATA_ORDER:
    stratum_profiles = profiles_df[profiles_df['stratum'] == stratum_name].copy()
    stratum_profiles = stratum_profiles.sort_values('mean_total_pva', ascending=False)
    cluster_order = stratum_profiles['cluster_id'].tolist()
    n_cls = len(cluster_order)

    if n_cls == 0:
        continue

    print(f"Creating grid for {stratum_name} stratum ({n_cls} clusters)...")

    n_cols = min(n_cls, 5)
    n_rows = int(np.ceil(n_cls / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    fig.suptitle(f'{stratum_name} Third — Possession Pattern Clusters (Sorted by Mean PVA)',
                 fontsize=13, fontweight='bold', y=1.04)
    axes = np.atleast_2d(axes)

    for idx, cid in enumerate(cluster_order):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx, col_idx]

        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b',
                      line_color='white', linewidth=1, goal_type='box')
        pitch.draw(ax=ax)

        profile = stratum_profiles[stratum_profiles['cluster_id'] == cid].iloc[0]
        pva = profile['mean_total_pva']
        color = pva_cmap(pva_norm(pva))

        cluster_data = clustered_df[clustered_df['cluster_id'] == cid]
        n_to_sample = min(N_SAMPLES, len(cluster_data))
        samples = cluster_data.sample(n=n_to_sample, random_state=rng)

        for _, srow in samples.iterrows():
            sx, sy = to_pitch_coords(srow)
            ax.plot(sx, sy, color=color, linewidth=0.6, alpha=0.25, zorder=2)
            ax.scatter(sx[0], sy[0], color='white', s=8, zorder=2,
                       edgecolors='none', alpha=0.3, marker='o')
            ax.scatter(sx[-1], sy[-1], color=color, s=10, zorder=2,
                       edgecolors='black', linewidths=0.2, alpha=0.3, marker='s')

        cx, cy = get_centroid_coords(cid)
        ax.plot(cx, cy, color=color, linewidth=3, zorder=4, alpha=0.9)
        draw_path_arrows(ax, cx, cy, color='white', n_arrows=2, size=10, zorder=5)
        ax.scatter(cx[0], cy[0], color='white', s=50, zorder=6, edgecolors='black',
                   linewidths=0.5, marker='o')
        ax.scatter(cx[-1], cy[-1], color=color, s=60, zorder=6, edgecolors='black',
                   linewidths=0.5, marker='s')

        n_poss = profile['n_possessions']
        ax.set_title(f"{cid}  (PVA={pva:.3f}, N={n_poss:,})",
                     fontsize=18, fontweight='bold', pad=3)

    for idx in range(n_cls, n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx, col_idx].set_visible(False)

    sm = ScalarMappable(cmap=pva_cmap, norm=pva_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Mean Total PVA', fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    fig.tight_layout(rect=[0, 0, 0.91, 1.0])
    slug = stratum_name.lower()
    out_path = os.path.join(viz_dir, f'cluster_grid_{slug}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")

# ==============================================================================
# CENTROIDS BY STRATUM
# ==============================================================================

print("Creating Figure: Centroids faceted by stratum...")

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, hspace=0.25, wspace=0.15)

ax_positions = {
    'Defensive': gs[0, 0],
    'Midfield':  gs[0, 1],
    'Attacking': gs[1, :],
}

for stratum_name in STRATA_ORDER:
    stratum_profiles = profiles_df[profiles_df['stratum'] == stratum_name]
    if len(stratum_profiles) == 0:
        continue

    if stratum_name == 'Attacking':
        ax = fig.add_subplot(gs[1, :])
        pos = ax.get_position()
        new_width = pos.width / 2
        new_x = pos.x0 + (pos.width - new_width) / 2
        ax.set_position([new_x, pos.y0, new_width, pos.height])
    else:
        ax = fig.add_subplot(ax_positions[stratum_name])

    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b',
                  line_color='white', linewidth=1.5, goal_type='box')
    pitch.draw(ax=ax)

    ax.axvline(x=40, color='yellow', linestyle='--', linewidth=1, alpha=0.4)
    ax.axvline(x=80, color='yellow', linestyle='--', linewidth=1, alpha=0.4)

    for _, profile in stratum_profiles.iterrows():
        cid = profile['cluster_id']
        pva = profile['mean_total_pva']
        color = pva_cmap(pva_norm(pva))
        cx, cy = get_centroid_coords(cid)

        ax.plot(cx, cy, color='black', linewidth=4.5, alpha=0.3, zorder=3,
                solid_capstyle='round')
        ax.plot(cx, cy, color=color, linewidth=3, alpha=0.85, zorder=4,
                solid_capstyle='round')
        draw_path_arrows(ax, cx, cy, color='white', n_arrows=2, size=10, zorder=5)
        ax.scatter(cx[0], cy[0], color='white', s=50, zorder=6, edgecolors='black',
                   linewidths=0.5, marker='o')
        ax.scatter(cx[-1], cy[-1], color=color, s=60, zorder=6, edgecolors='black',
                   linewidths=0.5, marker='s')
        mid = N_RESAMPLE // 2
        ax.text(cx[mid], cy[mid] - 3, cid,
                fontsize=6, color='white', fontweight='bold',
                ha='center', va='top', zorder=7,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='black',
                          alpha=0.5, edgecolor='none'))

    n_cls = len(stratum_profiles)
    ax.set_title(f'{stratum_name} Third ({n_cls} patterns)',
                 fontsize=22, fontweight='bold', color='white', pad=8)

fig.suptitle('Possession Pattern Centroids by Starting Third',
             fontsize=26, fontweight='bold', color='white', y=0.98)

sm = ScalarMappable(cmap=pva_cmap, norm=pva_norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Mean Total PVA', fontsize=20, color='white')
cbar.ax.yaxis.set_tick_params(color='white', labelsize=12)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=16)

out_path = os.path.join(viz_dir, 'cluster_centroids_all.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#22312b')
plt.close()
print(f"  Saved: {out_path}")

# ==============================================================================

print(f"\nAll visualizations saved to: {viz_dir}")
for s in STRATA_ORDER:
    n = profiles_df[profiles_df['stratum'] == s].shape[0]
    print(f"  cluster_grid_{s.lower()}.png     — {n} {s} clusters")
print(f"  cluster_centroids_all.png   — All {n_clusters} centroids on single pitch")
