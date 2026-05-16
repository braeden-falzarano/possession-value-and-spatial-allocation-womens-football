"""
Phase 5: per-team pattern usage grids and all-paths summaries.
Run: python 05_team_visualizations.py
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from mplsoccer import Pitch
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PITCH_LENGTH = 120
PITCH_WIDTH  = 80
N_SAMPLES    = 20
RANDOM_STATE = 42
DPI          = 150

COMPLETE_SEASONS = {'FA WSL 2018-19', 'FA WSL 2020-21'}

# ==============================================================================
# PATHS
# ==============================================================================

script_dir    = os.path.dirname(os.path.abspath(__file__))
project_root  = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
stats_dir     = os.path.join(project_root, 'results')
input_dir     = os.path.join(stats_dir, 'possession_patterns')
viz_base_dir  = os.path.join(project_root, 'visualizations', 'possession_patterning_analysis', 'teams')
os.makedirs(viz_base_dir, exist_ok=True)

# ==============================================================================
# LOAD DATA
# ==============================================================================

print("Loading data...")
clustered_df = pd.read_csv(os.path.join(input_dir, 'possessions_clustered.csv'))
profiles_df  = pd.read_csv(os.path.join(input_dir, 'cluster_profiles.csv'))
metrics_df   = pd.read_csv(os.path.join(input_dir, 'team_pattern_metrics.csv'))

N_RESAMPLE = 0
while f'p{N_RESAMPLE}_x' in clustered_df.columns:
    N_RESAMPLE += 1

profiles_df = profiles_df.sort_values('mean_total_pva', ascending=False).reset_index(drop=True)
cluster_order = profiles_df['cluster_id'].tolist()
n_clusters = len(cluster_order)

cluster_pva_global = profiles_df.set_index('cluster_id')['mean_total_pva'].to_dict()

centroid_data = {}
for _, row in profiles_df.iterrows():
    cid = row['cluster_id']
    xs = [row[f'centroid_p{i}_x'] * PITCH_LENGTH for i in range(N_RESAMPLE)]
    ys = [row[f'centroid_p{i}_y'] * PITCH_WIDTH  for i in range(N_RESAMPLE)]
    centroid_data[cid] = (np.array(xs), np.array(ys))

metrics_df = metrics_df[metrics_df['league'].isin(COMPLETE_SEASONS)].reset_index(drop=True)

print(f"  {len(clustered_df):,} total possessions, {n_clusters} clusters")
print(f"  {len(metrics_df)} team-seasons to visualize (complete seasons only: "
      f"{', '.join(sorted(COMPLETE_SEASONS))})")

# ==============================================================================
# HELPERS
# ==============================================================================

def to_pitch_coords(row):
    xs = [row[f'p{i}_x'] * PITCH_LENGTH for i in range(N_RESAMPLE)]
    ys = [row[f'p{i}_y'] * PITCH_WIDTH  for i in range(N_RESAMPLE)]
    return np.array(xs), np.array(ys)

def sanitize_name(league, team):
    short_league = (league.lower()
                    .replace('fa wsl ', 'wsl_')
                    .replace('nwsl ', 'nwsl_')
                    .replace(' ', '_'))
    safe_team = re.sub(r'[^a-z0-9]+', '_', team.lower()).strip('_')
    return f"{short_league}_{safe_team}"

def draw_third_lines(ax):
    ax.axvline(x=40, color='yellow', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.axvline(x=80, color='yellow', linestyle='--', alpha=0.4, linewidth=0.5)

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

from matplotlib.colors import LinearSegmentedColormap
diff_cmap = LinearSegmentedColormap.from_list('tight_diverging', [
    (0.0,  '#08306b'),   # deep navy
    (0.15, '#2171b5'),   # strong blue
    (0.35, '#6baed6'),   # medium blue
    (0.48, '#f0f0f0'),   # very narrow neutral
    (0.52, '#f0f0f0'),
    (0.65, '#e34a33'),   # medium red
    (0.85, '#cb181d'),   # strong red
    (1.0,  '#67000d'),   # deep crimson
], N=256)

# ==============================================================================
# GENERATE PER-TEAM FIGURES
# ==============================================================================

rng = np.random.RandomState(RANDOM_STATE)

for idx, team_row in metrics_df.iterrows():
    league = team_row['league']
    team   = team_row['team']
    rva_all = team_row['rva_all']
    mean_pva = team_row['mean_pva']
    pct_unasgn = team_row['pct_unassigned']
    n_poss = int(team_row['n_possessions'])

    folder_name = sanitize_name(league, team)
    team_dir = os.path.join(viz_base_dir, folder_name)
    os.makedirs(team_dir, exist_ok=True)

    team_all = clustered_df[
        (clustered_df['league'] == league) &
        (clustered_df['possession_team'] == team)
    ]
    team_assigned = team_all[team_all['cluster_id'].astype(str) != '-1']

    team_cluster_counts = team_assigned.groupby('cluster_id').size().to_dict()

    league_df = clustered_df[clustered_df['league'] == league]
    league_other = league_df[league_df['possession_team'] != team]
    league_other_assigned = league_other[league_other['cluster_id'].astype(str) != '-1']
    league_other_total = len(league_other)
    league_other_counts = league_other_assigned.groupby('cluster_id').size().to_dict()

    cluster_pva_diff = {}
    team_cluster_pva = {}
    league_cluster_pva = {}
    for cid in cluster_order:
        team_cdf = team_assigned[team_assigned['cluster_id'] == cid]
        other_cdf = league_other_assigned[league_other_assigned['cluster_id'] == cid]
        t_pva = team_cdf['total_pva'].mean() if len(team_cdf) >= 3 else np.nan
        l_pva = other_cdf['total_pva'].mean() if len(other_cdf) >= 3 else np.nan
        team_cluster_pva[cid] = t_pva
        league_cluster_pva[cid] = l_pva
        if not np.isnan(t_pva) and not np.isnan(l_pva):
            cluster_pva_diff[cid] = t_pva - l_pva
        else:
            cluster_pva_diff[cid] = 0.0

    diff_vals = [v for v in cluster_pva_diff.values() if v != 0]
    if diff_vals:
        max_abs = max(abs(min(diff_vals)), abs(max(diff_vals)), 0.01)
    else:
        max_abs = 0.01
    diff_norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    rva_str = f"{rva_all:.6f}" if not np.isnan(rva_all) else "N/A"
    print(f"  {league} — {team} (rva={rva_str}, mean_pva={mean_pva:.4f})")

    # ==================================================================
    # PATTERN USAGE GRID
    # ==================================================================

    n_cols = 5
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = np.atleast_2d(axes)

    for i, cid in enumerate(cluster_order):
        row_idx = i // n_cols
        col_idx = i % n_cols
        ax = axes[row_idx, col_idx]

        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b',
                      line_color='white', linewidth=1, goal_type='box')
        pitch.draw(ax=ax)
        draw_third_lines(ax)

        cx, cy = centroid_data[cid]
        team_n = team_cluster_counts.get(cid, 0)
        team_pct = team_n / n_poss * 100 if n_poss > 0 else 0
        pva_d = cluster_pva_diff[cid]

        color = diff_cmap(diff_norm(pva_d))

        cluster_team_df = team_assigned[team_assigned['cluster_id'] == cid]
        if len(cluster_team_df) > 0:
            sample = cluster_team_df.sample(
                n=min(N_SAMPLES, len(cluster_team_df)),
                random_state=RANDOM_STATE
            )
            for _, srow in sample.iterrows():
                sx, sy = to_pitch_coords(srow)
                ax.plot(sx, sy, color=color, alpha=0.3, linewidth=1.0, zorder=3)
                ax.scatter(sx[0], sy[0], color='white', s=12, zorder=3,
                           edgecolors='none', alpha=0.4, marker='o')
                ax.scatter(sx[-1], sy[-1], color=color, s=15, zorder=3,
                           edgecolors='black', linewidths=0.2, alpha=0.4, marker='s')

        centroid_alpha = 0.9 if team_n > 0 else 0.3
        ax.plot(cx, cy, color=color, linewidth=3, alpha=centroid_alpha,
                solid_capstyle='round', zorder=5)
        draw_path_arrows(ax, cx, cy, color='white', n_arrows=2, size=10, zorder=6)
        ax.scatter(cx[0], cy[0], color='white', s=50, zorder=7, edgecolors='black',
                   linewidths=0.5, marker='o')
        ax.scatter(cx[-1], cy[-1], color=color, s=60, zorder=7, edgecolors='black',
                   linewidths=0.5, marker='s')

        diff_str = f"+{pva_d:.3f}" if pva_d > 0 else f"{pva_d:.3f}"
        title_color = '#b2182b' if pva_d > 0 else ('#2166ac' if pva_d < 0 else 'black')
        ax.set_title(f"C{cid} — {team_n} ({team_pct:.1f}%)  [{diff_str}]",
                     fontsize=16, color=title_color, fontweight='bold', pad=2)

    for i in range(n_clusters, n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = i % n_cols
        axes[row_idx, col_idx].set_visible(False)

    sm = ScalarMappable(cmap=diff_cmap, norm=diff_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Team PVA − League PVA', fontsize=30)
    cbar.ax.tick_params(labelsize=24)

    fig.suptitle(f"{team} — {league}\n"
                 f"RVA = {rva_str}  |  Mean PVA = {mean_pva:.4f}  |  "
                 f"{n_poss} possessions ({pct_unasgn:.0f}% unassigned)",
                 fontsize=32, fontweight='bold', y=1.04)
    fig.tight_layout(rect=[0, 0, 0.91, 1.0])
    fig.savefig(os.path.join(team_dir, 'pattern_usage_grid.png'),
                dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ==================================================================
    # ALL-PATHS SUMMARY
    # ==================================================================

    fig, ax = plt.subplots(figsize=(14, 9))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b',
                  line_color='white', linewidth=1.5, goal_type='box')
    pitch.draw(ax=ax)
    draw_third_lines(ax)

    for _, prow in team_assigned.iterrows():
        cid = prow['cluster_id']
        color = diff_cmap(diff_norm(cluster_pva_diff[cid]))
        px, py = to_pitch_coords(prow)
        ax.plot(px, py, color=color, alpha=0.08, linewidth=0.4, zorder=2)
        ax.scatter(px[0], py[0], color='white', s=6, zorder=2, edgecolors='none',
                   alpha=0.08, marker='o')
        ax.scatter(px[-1], py[-1], color=color, s=8, zorder=2, edgecolors='none',
                   alpha=0.08, marker='s')

    for cid in cluster_order:
        team_n = team_cluster_counts.get(cid, 0)
        if team_n == 0:
            continue
        cx, cy = centroid_data[cid]
        color = diff_cmap(diff_norm(cluster_pva_diff[cid]))
        team_pct = team_n / n_poss * 100
        pva_d = cluster_pva_diff[cid]
        diff_str = f"+{pva_d:.3f}" if pva_d > 0 else f"{pva_d:.3f}"

        ax.plot(cx, cy, color='black', linewidth=5, alpha=0.4, zorder=3,
                solid_capstyle='round')
        ax.plot(cx, cy, color=color, linewidth=3.5, alpha=0.9, zorder=4,
                solid_capstyle='round')
        draw_path_arrows(ax, cx, cy, color='white', n_arrows=2, size=12, zorder=5)
        ax.scatter(cx[0], cy[0], color='white', s=60, zorder=6, edgecolors='black',
                   linewidths=0.8, marker='o')
        ax.scatter(cx[-1], cy[-1], color=color, s=80, zorder=6, edgecolors='black',
                   linewidths=0.8, marker='s')
        mid = N_RESAMPLE // 2
        ax.text(cx[mid], cy[mid] - 3, f"C{cid} {team_pct:.0f}%\n{diff_str}",
                fontsize=5.5, color='white', fontweight='bold',
                ha='center', va='top', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                          alpha=0.6, edgecolor='none'))

    sm = ScalarMappable(cmap=diff_cmap, norm=diff_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('Team PVA − League PVA', fontsize=20, color='white')
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=16)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=16)

    n_assigned = len(team_assigned)
    ax.set_title(f"{team} — {league}\n"
                 f"RVA = {rva_str}  |  Mean PVA = {mean_pva:.4f}  |  "
                 f"{n_assigned} assigned paths ({pct_unasgn:.0f}% unassigned)",
                 fontsize=20, fontweight='bold', color='white', pad=10)

    fig.savefig(os.path.join(team_dir, 'all_paths_summary.png'),
                dpi=DPI, bbox_inches='tight', facecolor='#22312b')
    plt.close(fig)

print(f"\nAll team visualizations saved to: {viz_base_dir}")
print(f"  {len(metrics_df)} team-seasons, 2 figures each")
