"""
6x3 grid zone visualizations: league-wide, per-team, facet comparisons, and ZUE quadrant maps.
"""

LEAGUE_CONFIG = {
    'name': 'FA WSL 2018-19',
    'slug': 'fa_wsl_2018-19',
    'actions_csv': 'fa_wsl_2018-19/fa_wsl_2018-19_actions_with_pva.csv',
}

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from mplsoccer import Pitch
from scipy.stats import zscore
from pypalettes import load_cmap
import warnings
warnings.filterwarnings('ignore')

buildup_vs_league_cmap = LinearSegmentedColormap.from_list(
    'BlueOrange_vivid',
    [(0.00, '#001AFF'),   # deep blue
     (0.15, '#2266FF'),   # vivid blue
     (0.30, '#5599FF'),   # medium blue
     (0.42, '#99CCFF'),   # light blue (still tinted)
     (0.50, '#FFF8E0'),   # narrow warm white center
     (0.58, '#FFCC77'),   # light orange (still tinted)
     (0.70, '#FF8833'),   # medium orange
     (0.85, '#FF4411'),   # vivid orange-red
     (1.00, '#CC1100')],  # deep red
    N=256
)

from matplotlib.colors import LinearSegmentedColormap
zva_cmap = LinearSegmentedColormap.from_list('tight_diverging', [
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
# PATHS
# ==============================================================================

script_dir    = os.path.dirname(os.path.abspath(__file__))
project_root  = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
processed_dir = os.path.join(project_root, 'data', 'processed')
viz_dir       = os.path.join(project_root, 'visualizations', 'zone_analysis', LEAGUE_CONFIG['slug'])

grid_dir        = os.path.join(viz_dir, 'GRID')
league_dir      = os.path.join(grid_dir, 'league_wide')
team_dir        = os.path.join(grid_dir, 'by_team')
comparison_dir  = os.path.join(grid_dir, 'team_comparison_by_category')
ue_dir          = os.path.join(grid_dir, 'zone_utilization_efficiency')

for d in [league_dir, team_dir, comparison_dir, ue_dir]:
    os.makedirs(d, exist_ok=True)

print("="*80)
print(f"{LEAGUE_CONFIG['name']} GRID VISUALIZATIONS - 6x3 GRID (18 CELLS)")
print("="*80)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PITCH_LENGTH = 120
PITCH_WIDTH  = 80
X_BINS = 6
Y_BINS = 3

DEFENSIVE_THIRD_END = 40
MIDFIELD_THIRD_END  = 80

DANGER_ZONE_X_MIN = 100
DANGER_ZONE_X_MAX = 120
DANGER_ZONE_Y_MIN = PITCH_WIDTH / Y_BINS
DANGER_ZONE_Y_MAX = 2 * PITCH_WIDTH / Y_BINS

cell_width  = PITCH_LENGTH / X_BINS
cell_height = PITCH_WIDTH  / Y_BINS

defensive_x_end = int(DEFENSIVE_THIRD_END / cell_width)
midfield_x_end  = int(MIDFIELD_THIRD_END  / cell_width)

defensive_cmap = LinearSegmentedColormap.from_list('defensive',
    ['#08519c', '#f7f7f7', '#6baed6'])
midfield_cmap  = LinearSegmentedColormap.from_list('midfield',
    ['#006d2c', '#f7f7f7', '#74c476'])
attacking_cmap = LinearSegmentedColormap.from_list('attacking',
    ['#d94801', '#f7f7f7', '#fdae6b'])

# ==============================================================================
# LOAD DATA
# ==============================================================================

print("\n[Loading processed data...]")
actions_csv_path = os.path.join(processed_dir, LEAGUE_CONFIG['slug'] + '_actions_with_pva.csv')
actions_df = pd.read_csv(actions_csv_path)
print(f"  Loaded {len(actions_df):,} actions with clean PVA values")

def compute_cell_idx(x, y):
    cx = min(int(x / cell_width),  X_BINS - 1)
    cy = min(int(y / cell_height), Y_BINS - 1)
    return cx * Y_BINS + cy

actions_df['cell_6x3'] = actions_df.apply(
    lambda r: compute_cell_idx(r['x'], r['y']), axis=1
)

def is_in_six_yard_box(x, y):
    return (x >= 114) and (x <= 120) and (y >= 30) and (y <= 50)

before = len(actions_df)
actions_df_viz = actions_df[
    ~actions_df.apply(lambda r: is_in_six_yard_box(r['x'], r['y']), axis=1)
].copy()
print(f"  Filtered {before - len(actions_df_viz):,} actions from 6-yard box area")

def is_in_danger_zone(x, y):
    return (x >= DANGER_ZONE_X_MIN) and (x <= DANGER_ZONE_X_MAX) and \
           (y >= DANGER_ZONE_Y_MIN) and (y <= DANGER_ZONE_Y_MAX)

actions_df_buildup = actions_df_viz[
    ~actions_df_viz.apply(lambda r: is_in_danger_zone(r['x'], r['y']), axis=1)
].copy()
print(f"  Build-up dataset: {len(actions_df_buildup):,} actions (danger zone excluded)")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def make_grid_arrays(df):
    """Return (pva_arr, usage_arr) for the 6x3 grid."""
    pva_by_cell   = df.groupby('cell_6x3')['pva'].mean().to_dict()
    count_by_cell = df.groupby('cell_6x3').size().to_dict()
    pva_arr   = np.zeros((X_BINS, Y_BINS))
    usage_arr = np.zeros((X_BINS, Y_BINS))
    for cell_idx, val in pva_by_cell.items():
        cx = int(cell_idx) // Y_BINS
        cy = int(cell_idx) % Y_BINS
        if 0 <= cx < X_BINS and 0 <= cy < Y_BINS:
            pva_arr[cx, cy] = val
    for cell_idx, count in count_by_cell.items():
        cx = int(cell_idx) // Y_BINS
        cy = int(cell_idx) % Y_BINS
        if 0 <= cx < X_BINS and 0 <= cy < Y_BINS:
            usage_arr[cx, cy] = count
    return pva_arr, usage_arr

def add_third_markers(ax):
    ax.axvline(x=DEFENSIVE_THIRD_END, color='yellow', linestyle='--', linewidth=2, alpha=0.5)
    ax.axvline(x=MIDFIELD_THIRD_END,  color='yellow', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(DEFENSIVE_THIRD_END/2, PITCH_WIDTH + 3, 'Defensive',
            ha='center', fontsize=18, fontweight='bold', color='yellow')
    ax.text((DEFENSIVE_THIRD_END + MIDFIELD_THIRD_END)/2, PITCH_WIDTH + 3, 'Midfield',
            ha='center', fontsize=18, fontweight='bold', color='yellow')
    ax.text((MIDFIELD_THIRD_END + PITCH_LENGTH)/2, PITCH_WIDTH + 3, 'Attacking',
            ha='center', fontsize=18, fontweight='bold', color='yellow')

def add_grid_lines(ax):
    for i in range(1, X_BINS):
        ax.axvline(x=i * cell_width,  color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    for j in range(1, Y_BINS):
        ax.axhline(y=j * cell_height, color='white', linestyle='-', linewidth=0.5, alpha=0.3)

def draw_cells(ax, value_arr, usage_arr, cmap, norm, alpha=0.75,
               annotate=True, fmt='.3f', fontsize=10):
    for i in range(X_BINS):
        for j in range(Y_BINS):
            if usage_arr[i, j] == 0:
                continue
            x = i * cell_width
            y = j * cell_height
            color = cmap(norm(value_arr[i, j]))
            rect = patches.Rectangle(
                (x, y), cell_width, cell_height,
                linewidth=0.5, edgecolor='white', facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)
            if annotate:
                cx = x + cell_width / 2
                cy = y + cell_height / 2
                brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
                tc = 'black' if brightness > 0.5 else 'white'
                ax.text(cx, cy, f'{value_arr[i, j]:{fmt}}',
                        ha='center', va='center', fontsize=fontsize,
                        color=tc, fontweight='bold')

def draw_thirds_cells(ax, value_norm_arr, usage_arr, z_norm, alpha=0.8, annotate=True):
    for i in range(X_BINS):
        for j in range(Y_BINS):
            if usage_arr[i, j] == 0:
                continue
            x = i * cell_width
            y = j * cell_height
            z_val = np.clip(value_norm_arr[i, j], -3, 3)
            if i < defensive_x_end:
                color = defensive_cmap(z_norm(z_val))
            elif i < midfield_x_end:
                color = midfield_cmap(z_norm(z_val))
            else:
                color = attacking_cmap(z_norm(z_val))
            rect = patches.Rectangle(
                (x, y), cell_width, cell_height,
                linewidth=0.5, edgecolor='white', facecolor=color, alpha=alpha
            )
            ax.add_patch(rect)
            if annotate:
                cx = x + cell_width / 2
                cy = y + cell_height / 2
                brightness = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
                tc = 'black' if brightness > 0.5 else 'white'
                ax.text(cx, cy, f'{z_val:.2f}',
                        ha='center', va='center', fontsize=10,
                        color=tc, fontweight='bold')

def draw_danger_zone(ax):
    rect = patches.Rectangle(
        (DANGER_ZONE_X_MIN, DANGER_ZONE_Y_MIN),
        DANGER_ZONE_X_MAX - DANGER_ZONE_X_MIN,
        DANGER_ZONE_Y_MAX - DANGER_ZONE_Y_MIN,
        linewidth=2, edgecolor='white', facecolor='gray',
        alpha=0.5, hatch='///', linestyle='--'
    )
    ax.add_patch(rect)

def new_pitch():
    fig, ax = plt.subplots(figsize=(16, 10))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b',
                  line_color='white', linewidth=2, goal_type='box')
    pitch.draw(ax=ax)
    return fig, ax

def add_colorbar(fig, ax, cmap, norm, label):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, rotation=270, labelpad=20, fontsize=20, fontweight='bold')
    cbar.ax.tick_params(labelsize=16)
    return cbar

def normalize_by_thirds(arr, reference_usage=None):
    if reference_usage is None:
        reference_usage = arr
    arr_norm = np.zeros_like(arr, dtype=float)
    for x_start, x_end in [
        (0, defensive_x_end),
        (defensive_x_end, midfield_x_end),
        (midfield_x_end, X_BINS)
    ]:
        region = arr[x_start:x_end, :]
        if region.size > 0 and region.std() > 0:
            arr_norm[x_start:x_end, :] = zscore(region.flatten()).reshape(region.shape)
    return arr_norm

# ==============================================================================
# COMPUTE LEAGUE BASELINES
# ==============================================================================

print("\n[Computing league baselines (team-weighted)...]")

_all_teams_viz = sorted(actions_df_viz['team'].unique())
_n_teams = len(_all_teams_viz)

_sum_pva       = np.zeros((X_BINS, Y_BINS))
_cnt_pva       = np.zeros((X_BINS, Y_BINS))
_sum_prop      = np.zeros((X_BINS, Y_BINS))
_sum_bu_pva    = np.zeros((X_BINS, Y_BINS))
_cnt_bu_pva    = np.zeros((X_BINS, Y_BINS))
_sum_bu_prop   = np.zeros((X_BINS, Y_BINS))

_team_pva_arrs     = {}
_team_prop_arrs    = {}
_team_bu_pva_arrs  = {}
_team_bu_prop_arrs = {}

for _t in _all_teams_viz:
    _ta  = actions_df_viz[actions_df_viz['team'] == _t]
    _bu  = actions_df_buildup[actions_df_buildup['team'] == _t]
    _pva_t, _cnt_t   = make_grid_arrays(_ta)
    _pva_bu, _cnt_bu = make_grid_arrays(_bu)

    _has_t  = _cnt_t  > 0
    _has_bu = _cnt_bu > 0
    _sum_pva    += np.where(_has_t,  _pva_t,  0.0)
    _cnt_pva    += _has_t.astype(float)
    _sum_bu_pva += np.where(_has_bu, _pva_bu, 0.0)
    _cnt_bu_pva += _has_bu.astype(float)

    _t_total  = _cnt_t.sum()
    _bu_total = _cnt_bu.sum()
    _sum_prop    += (_cnt_t  / _t_total  if _t_total  > 0 else _cnt_t)
    _sum_bu_prop += (_cnt_bu / _bu_total if _bu_total > 0 else _cnt_bu)

    _team_pva_arrs[_t]     = np.where(_has_t,  _pva_t,  np.nan)
    _team_prop_arrs[_t]    = _cnt_t  / _t_total  if _t_total  > 0 else np.zeros((X_BINS, Y_BINS))
    _team_bu_pva_arrs[_t]  = np.where(_has_bu, _pva_bu, np.nan)
    _team_bu_prop_arrs[_t] = _cnt_bu / _bu_total if _bu_total > 0 else np.zeros((X_BINS, Y_BINS))

league_pva_arr       = np.divide(_sum_pva,    _cnt_pva,    out=np.zeros_like(_sum_pva),    where=_cnt_pva    > 0)
league_buildup_pva_arr = np.divide(_sum_bu_pva, _cnt_bu_pva, out=np.zeros_like(_sum_bu_pva), where=_cnt_bu_pva > 0)
league_usage_prop    = _sum_prop    / _n_teams
league_buildup_prop  = _sum_bu_prop / _n_teams

# LOO baselines: each team's reference = mean of the other teams
loo_pva_arrs        = {}
loo_bu_pva_arrs     = {}
loo_usage_prop_arrs = {}
loo_bu_prop_arrs    = {}
for _t in _all_teams_viz:
    _other = [tt for tt in _all_teams_viz if tt != _t]
    loo_pva_arrs[_t]        = np.nanmean(np.stack([_team_pva_arrs[tt]     for tt in _other], axis=0), axis=0)
    loo_bu_pva_arrs[_t]     = np.nanmean(np.stack([_team_bu_pva_arrs[tt]  for tt in _other], axis=0), axis=0)
    loo_usage_prop_arrs[_t] = np.mean(   np.stack([_team_prop_arrs[tt]    for tt in _other], axis=0), axis=0)
    loo_bu_prop_arrs[_t]    = np.mean(   np.stack([_team_bu_prop_arrs[tt] for tt in _other], axis=0), axis=0)

_, league_usage_arr         = make_grid_arrays(actions_df_viz)
_, league_buildup_usage_arr = make_grid_arrays(actions_df_buildup)

league_pva_norm   = normalize_by_thirds(league_pva_arr)
league_usage_norm = normalize_by_thirds(league_usage_arr)

z_norm = TwoSlopeNorm(vcenter=0, vmin=-3, vmax=3)

print(f"  League PVA range: [{league_pva_arr.min():.4f}, {league_pva_arr.max():.4f}]")
print(f"  League usage range: [{league_usage_arr.min():.0f}, {league_usage_arr.max():.0f}]")

# ==============================================================================
# VIZ 1: LEAGUE-WIDE ZONE POSSESSION VALUES
# ==============================================================================

print("\n[Creating Viz 1: League-Wide Zone Possession Values...]")

fig, ax = new_pitch()
norm = Normalize(vmin=league_pva_arr.min(), vmax=league_pva_arr.max())
draw_cells(ax, league_pva_arr, league_usage_arr, plt.cm.RdYlGn, norm)
add_colorbar(fig, ax, plt.cm.RdYlGn, norm, 'Average PVA per Action')
add_grid_lines(ax)
add_third_markers(ax)
ax.set_title(f'{LEAGUE_CONFIG["name"]} — Mean PVA by Zone',
             fontsize=15, fontweight='bold', pad=25)
plt.tight_layout()
plt.savefig(os.path.join(league_dir, '01_league_possession_values_clean.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  Saved: 01_league_possession_values_clean.png")
plt.close()

# ==============================================================================
# VIZ 1B: BUILD-UP PLAY (excluding danger zone)
# ==============================================================================

print("\n[Creating Viz 1B: Build-Up Play Values...]")

fig, ax = new_pitch()
norm_bu = Normalize(vmin=league_buildup_pva_arr.min(), vmax=league_buildup_pva_arr.max())
draw_cells(ax, league_buildup_pva_arr, league_buildup_usage_arr, plt.cm.RdYlGn, norm_bu)
draw_danger_zone(ax)
add_colorbar(fig, ax, plt.cm.RdYlGn, norm_bu, 'Average PVA per Action')
add_grid_lines(ax)
add_third_markers(ax)
ax.set_title(f'{LEAGUE_CONFIG["name"]} — Build-Up Play PVA by Zone\n'
             'Excluding danger zone',
             fontsize=15, fontweight='bold', pad=25)
plt.tight_layout()
plt.savefig(os.path.join(league_dir, '01b_league_buildup_play_clean.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  Saved: 01b_league_buildup_play_clean.png")
plt.close()

# ==============================================================================
# VIZ 2: LEAGUE-WIDE ZONE USAGE
# ==============================================================================

print("\n[Creating Viz 2: League-Wide Zone Usage...]")

fig, ax = new_pitch()
norm_u = Normalize(vmin=league_usage_arr.min(), vmax=league_usage_arr.max())
draw_cells(ax, league_usage_arr, league_usage_arr, plt.cm.YlOrRd, norm_u, fmt='.0f')
add_colorbar(fig, ax, plt.cm.YlOrRd, norm_u, 'Action Frequency')
add_grid_lines(ax)
add_third_markers(ax)
ax.set_title(f'{LEAGUE_CONFIG["name"]} — Action Frequency by Zone',
             fontsize=15, fontweight='bold', pad=25)
plt.tight_layout()
plt.savefig(os.path.join(league_dir, '02_league_usage_clean.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  Saved: 02_league_usage_clean.png")
plt.close()

# ==============================================================================
# VIZ 3: THIRDS-NORMALIZED POSSESSION VALUES
# ==============================================================================

print("\n[Creating Viz 3: Thirds-Normalized Possession Values...]")

fig, ax = new_pitch()
draw_thirds_cells(ax, league_pva_norm, league_usage_arr, z_norm)
add_grid_lines(ax)
add_third_markers(ax)
ax.set_title(f'{LEAGUE_CONFIG["name"]} — PVA by Zone (Normalized within Thirds)',
             fontsize=15, fontweight='bold', pad=25)
plt.tight_layout()
plt.savefig(os.path.join(league_dir, '03_league_possession_thirds_normalized_clean.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  Saved: 03_league_possession_thirds_normalized_clean.png")
plt.close()

# ==============================================================================
# VIZ 4: THIRDS-NORMALIZED USAGE
# ==============================================================================

print("\n[Creating Viz 4: Thirds-Normalized Usage...]")

fig, ax = new_pitch()
draw_thirds_cells(ax, league_usage_norm, league_usage_arr, z_norm)
add_grid_lines(ax)
add_third_markers(ax)
ax.set_title(f'{LEAGUE_CONFIG["name"]} — Usage by Zone (Normalized within Thirds)',
             fontsize=15, fontweight='bold', pad=25)
plt.tight_layout()
plt.savefig(os.path.join(league_dir, '04_league_usage_thirds_normalized_clean.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  Saved: 04_league_usage_thirds_normalized_clean.png")
plt.close()

# ==============================================================================
# TEAM-SPECIFIC VISUALIZATIONS
# ==============================================================================

print("\n" + "="*80)
print("CREATING TEAM-SPECIFIC VISUALIZATIONS")
print("="*80)

teams = sorted(actions_df_viz['team'].unique())

team_grid_data = {
    'possession':        {},
    'buildup':           {},
    'usage':             {},
    'vs_league':         {},
    'buildup_vs_league': {},
    'zva_contribution': {},
    'zva_buildup': {},
}

for team_idx, team_name in enumerate(teams, 1):
    print(f"\n  [{team_idx}/{len(teams)}] {team_name}...")

    team_actions  = actions_df_viz[actions_df_viz['team'] == team_name]
    team_buildup  = actions_df_buildup[actions_df_buildup['team'] == team_name]
    safe_name     = team_name.replace(' ', '_').replace('/', '_')
    n_total       = len(team_actions)
    n_buildup     = len(team_buildup)

    if n_total < 10:
        print(f"    Skipped (too few actions: {n_total})")
        continue

    team_pva_arr,    team_usage_arr    = make_grid_arrays(team_actions)
    team_bu_pva_arr, team_bu_usage_arr = make_grid_arrays(team_buildup)

    team_total    = team_usage_arr.sum()
    team_usage_prop = team_usage_arr / team_total if team_total > 0 else team_usage_arr

    pva_diff    = team_pva_arr    - loo_pva_arrs[team_name]
    bu_pva_diff = team_bu_pva_arr - loo_bu_pva_arrs[team_name]

    team_grid_data['possession'][team_name]        = (team_pva_arr,    team_usage_arr)
    team_grid_data['buildup'][team_name]           = (team_bu_pva_arr, team_bu_usage_arr)
    team_grid_data['usage'][team_name]             = (team_usage_prop, team_usage_arr)
    team_grid_data['vs_league'][team_name]         = (pva_diff,        team_usage_arr)
    team_grid_data['buildup_vs_league'][team_name] = (bu_pva_diff,     team_bu_usage_arr)

    # VIZ A: Possession Values
    fig, ax = new_pitch()
    norm_p = Normalize(vmin=team_pva_arr.min(), vmax=team_pva_arr.max())
    draw_cells(ax, team_pva_arr, team_usage_arr, plt.cm.RdYlGn, norm_p)
    add_colorbar(fig, ax, plt.cm.RdYlGn, norm_p, 'Average PVA per Action')
    add_grid_lines(ax)
    add_third_markers(ax)
    ax.set_title(f'{team_name} — Mean PVA by Zone  (n={n_total:,})',
                 fontsize=15, fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(os.path.join(team_dir, f'{safe_name}_possession_clean.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # VIZ B: Build-Up Play
    if n_buildup >= 10:
        fig, ax = new_pitch()
        norm_tb = Normalize(vmin=team_bu_pva_arr.min(), vmax=team_bu_pva_arr.max())
        draw_cells(ax, team_bu_pva_arr, team_bu_usage_arr, plt.cm.RdYlGn, norm_tb)
        draw_danger_zone(ax)
        add_colorbar(fig, ax, plt.cm.RdYlGn, norm_tb, 'Average PVA per Action')
        add_grid_lines(ax)
        add_third_markers(ax)
        ax.set_title(f'{team_name} — Build-Up Play PVA  (n={n_buildup:,})',
                     fontsize=15, fontweight='bold', pad=25)
        plt.tight_layout()
        plt.savefig(os.path.join(team_dir, f'{safe_name}_buildup_clean.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # VIZ C: Usage Density
    fig, ax = new_pitch()
    norm_tu = Normalize(vmin=team_usage_prop.min(), vmax=team_usage_prop.max())
    draw_cells(ax, team_usage_prop, team_usage_arr, plt.cm.YlOrRd, norm_tu, fmt='.3f')
    add_colorbar(fig, ax, plt.cm.YlOrRd, norm_tu, 'Proportion of Team Actions')
    add_grid_lines(ax)
    add_third_markers(ax)
    ax.set_title(f'{team_name} — Usage by Zone  (n={n_total:,})',
                 fontsize=15, fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(os.path.join(team_dir, f'{safe_name}_usage_clean.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # VIZ C2: ZVA Contribution
    usage_diff_arr = team_usage_prop - loo_usage_prop_arrs[team_name]
    zva_arr = usage_diff_arr * loo_pva_arrs[team_name]

    team_grid_data['zva_contribution'][team_name] = (zva_arr, team_usage_arr)

    fig, ax = new_pitch()
    max_abs_zva = max(abs(np.nanmin(zva_arr)), abs(np.nanmax(zva_arr))) or 0.001
    norm_zva = TwoSlopeNorm(vmin=-max_abs_zva, vcenter=0, vmax=max_abs_zva)
    draw_cells(ax, zva_arr, team_usage_arr, zva_cmap, norm_zva)
    add_colorbar(fig, ax, zva_cmap, norm_zva, 'ZVA Contribution')
    add_grid_lines(ax)
    add_third_markers(ax)
    ax.set_title(f'{team_name} — ZVA Contribution  (n={n_total:,})',
                 fontsize=15, fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(os.path.join(team_dir, f'{safe_name}_zva_contribution_clean.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # VIZ C3: Buildup ZVA Contribution
    bu_usage_diff_arr = (team_bu_usage_arr / team_bu_usage_arr.sum()
                         if team_bu_usage_arr.sum() > 0 else team_bu_usage_arr) \
                        - loo_bu_prop_arrs[team_name]
    bu_zva_arr = bu_usage_diff_arr * loo_bu_pva_arrs[team_name]

    team_grid_data['zva_buildup'][team_name] = (bu_zva_arr, team_bu_usage_arr)

    fig, ax = new_pitch()
    max_abs_bu = max(abs(np.nanmin(bu_zva_arr)), abs(np.nanmax(bu_zva_arr))) or 0.001
    norm_bu_zva = TwoSlopeNorm(vmin=-max_abs_bu, vcenter=0, vmax=max_abs_bu)
    draw_cells(ax, bu_zva_arr, team_bu_usage_arr, zva_cmap, norm_bu_zva)
    add_colorbar(fig, ax, zva_cmap, norm_bu_zva, 'Build-Up ZVA Contribution')
    add_grid_lines(ax)
    add_third_markers(ax)
    danger_rect = patches.Rectangle((DANGER_ZONE_X_MIN, DANGER_ZONE_Y_MIN),
                                    DANGER_ZONE_X_MAX - DANGER_ZONE_X_MIN,
                                    DANGER_ZONE_Y_MAX - DANGER_ZONE_Y_MIN,
                                    linewidth=2, edgecolor='white', facecolor='gray',
                                    alpha=0.5, hatch='///', linestyle='--')
    ax.add_patch(danger_rect)
    ax.set_title(f'{team_name} — Build-Up ZVA Contribution  (n={n_buildup:,})\nExcluding danger zone',
                 fontsize=15, fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(os.path.join(team_dir, f'{safe_name}_zva_buildup_clean.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # VIZ D: PVA vs League Average
    fig, ax = new_pitch()
    max_abs = max(abs(pva_diff.min()), abs(pva_diff.max())) or 0.001
    norm_vl = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    draw_cells(ax, pva_diff, team_usage_arr, buildup_vs_league_cmap, norm_vl)
    add_colorbar(fig, ax, buildup_vs_league_cmap, norm_vl, 'PVA vs League Average')
    add_grid_lines(ax)
    add_third_markers(ax)
    ax.set_title(f'{team_name} — PVA vs League Average  (n={n_total:,})',
                 fontsize=15, fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(os.path.join(team_dir, f'{safe_name}_pva_vs_league_clean.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # VIZ E: Build-Up vs League Average
    if n_buildup >= 10:
        fig, ax = new_pitch()
        max_abs_bu_diff = max(abs(np.nanmin(bu_pva_diff)), abs(np.nanmax(bu_pva_diff))) or 0.001
        norm_bvl = TwoSlopeNorm(vmin=-max_abs_bu_diff, vcenter=0, vmax=max_abs_bu_diff)
        draw_cells(ax, bu_pva_diff, team_bu_usage_arr, buildup_vs_league_cmap, norm_bvl)
        draw_danger_zone(ax)
        add_colorbar(fig, ax, buildup_vs_league_cmap, norm_bvl, 'Build-Up PVA vs League Avg')
        add_grid_lines(ax)
        add_third_markers(ax)
        ax.set_title(f'{team_name} — Build-Up PVA vs League Average  (n={n_buildup:,})',
                     fontsize=15, fontweight='bold', pad=25)
        plt.tight_layout()
        plt.savefig(os.path.join(team_dir, f'{safe_name}_buildup_vs_league_clean.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    print(f"    Created 7 grid visualizations (incl. ZVA + buildup ZVA)")

# ==============================================================================
# FACET GRID COMPARISONS - ALL TEAMS BY CATEGORY
# ==============================================================================

print("\n" + "="*80)
print("CREATING FACET GRID COMPARISONS")
print("="*80)

n_teams = len(teams)
n_cols  = 4
n_rows  = (n_teams + n_cols - 1) // n_cols

def create_facet_grid(data_dict, title, filename, cmap=plt.cm.RdYlGn,
                      use_diverging=False, show_danger_zone=False):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 6 * n_rows))
    axes = axes.flatten()

    all_vals = []
    for tn in teams:
        if tn in data_dict:
            val_arr, _ = data_dict[tn]
            all_vals.extend(val_arr.flatten())

    if len(all_vals) == 0:
        plt.close()
        return

    g_min, g_max = np.nanmin(all_vals), np.nanmax(all_vals)
    if use_diverging:
        vmax = max(abs(g_min), abs(g_max)) or 0.001
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        norm = Normalize(vmin=g_min, vmax=g_max)

    for idx, team_name in enumerate(teams):
        ax = axes[idx]
        if team_name not in data_dict:
            ax.axis('off')
            continue

        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b',
                      line_color='white', linewidth=1.5, goal_type='box')
        pitch.draw(ax=ax)

        val_arr, usage_arr = data_dict[team_name]

        for i in range(X_BINS):
            for j in range(Y_BINS):
                if usage_arr[i, j] == 0:
                    continue
                x = i * cell_width
                y = j * cell_height
                color = cmap(norm(val_arr[i, j]))
                rect = patches.Rectangle(
                    (x, y), cell_width, cell_height,
                    linewidth=0.3, edgecolor='white', facecolor=color, alpha=0.75
                )
                ax.add_patch(rect)

        if show_danger_zone:
            draw_danger_zone(ax)

        for i in range(1, X_BINS):
            ax.axvline(x=i * cell_width,  color='white', linestyle='-', linewidth=0.3, alpha=0.2)
        for j in range(1, Y_BINS):
            ax.axhline(y=j * cell_height, color='white', linestyle='-', linewidth=0.3, alpha=0.2)

        ax.axvline(x=DEFENSIVE_THIRD_END, color='yellow', linestyle='--', linewidth=1, alpha=0.4)
        ax.axvline(x=MIDFIELD_THIRD_END,  color='yellow', linestyle='--', linewidth=1, alpha=0.4)

        ax.set_title(team_name, fontsize=22, fontweight='bold', pad=8)

    for idx in range(n_teams, len(axes)):
        axes[idx].axis('off')

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=16)

    fig.suptitle(title, fontsize=42, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(os.path.join(comparison_dir, filename),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")

print("\nGenerating 7 team comparison visualizations...")

create_facet_grid(
    team_grid_data['possession'],
    f'{LEAGUE_CONFIG["name"]} — Mean PVA by Zone (All Teams)',
    '01_all_teams_possession_values.png',
    cmap=plt.cm.RdYlGn, use_diverging=False
)

create_facet_grid(
    team_grid_data['buildup'],
    f'{LEAGUE_CONFIG["name"]} — Build-Up Play PVA (All Teams)',
    '02_all_teams_buildup_play.png',
    cmap=plt.cm.RdYlGn, use_diverging=False, show_danger_zone=True
)

create_facet_grid(
    team_grid_data['usage'],
    f'{LEAGUE_CONFIG["name"]} — Usage by Zone (All Teams)',
    '03_all_teams_usage_density.png',
    cmap=plt.cm.YlOrRd, use_diverging=False
)

create_facet_grid(
    team_grid_data['vs_league'],
    f'{LEAGUE_CONFIG["name"]} — PVA vs League Average (All Teams)',
    '04_all_teams_vs_league_avg.png',
    cmap=buildup_vs_league_cmap, use_diverging=True
)

create_facet_grid(
    team_grid_data['buildup_vs_league'],
    f'{LEAGUE_CONFIG["name"]} — Build-Up PVA vs League Average (All Teams)',
    '05_all_teams_buildup_vs_league_avg.png',
    cmap=buildup_vs_league_cmap, use_diverging=True, show_danger_zone=True
)

create_facet_grid(
    team_grid_data['zva_contribution'],
    f'{LEAGUE_CONFIG["name"]} — Zone Value Added Contribution (All Teams)',
    '06_all_teams_zva_contribution.png',
    cmap=zva_cmap, use_diverging=True
)

create_facet_grid(
    team_grid_data['zva_buildup'],
    f'{LEAGUE_CONFIG["name"]} — Build-Up ZVA Contribution (All Teams)',
    '07_all_teams_zva_buildup.png',
    cmap=zva_cmap, use_diverging=True, show_danger_zone=True
)

print(f"\n  Created 7 team comparison facet grids")

# ==============================================================================
# ZONE UTILIZATION EFFICIENCY - QUADRANT MAP (1 per team)
# ==============================================================================

print("\n" + "="*80)
print("CREATING ZONE UTILIZATION EFFICIENCY VISUALIZATIONS")
print("="*80)
QUADRANT_COLORS = {
    'golden':   '#FFD700',
    'problem':  '#CC2222',
    'untapped': '#2266CC',
    'avoided':  '#666666',
}

legend_labels = {
    'golden':   'Golden Zone\n(High use + Above avg)',
    'problem':  'Problem Zone\n(High use + Below avg)',
    'untapped': 'Untapped Zone\n(Low use + Above avg)',
    'avoided':  'Correctly Avoided\n(Low use + Below avg)',
}

for team_idx, team_name in enumerate(teams, 1):
    if (team_name not in team_grid_data['vs_league'] or
            team_name not in team_grid_data['usage']):
        continue

    print(f"\n  [{team_idx}/{len(teams)}] {team_name}...")
    safe_name = team_name.replace(' ', '_').replace('/', '_')

    eff_arr,  usage_ref = team_grid_data['vs_league'][team_name]
    util_arr, _         = team_grid_data['usage'][team_name]

    usage_diff_arr = util_arr - loo_usage_prop_arrs[team_name]

    fig, ax = new_pitch()

    for i in range(X_BINS):
        for j in range(Y_BINS):
            if usage_ref[i, j] == 0:
                continue
            x = i * cell_width
            y = j * cell_height
            above_avg = eff_arr[i, j]       > 0   # outperforms league PVA
            high_use  = usage_diff_arr[i, j] > 0   # over-indexes vs league usage
            if above_avg and high_use:
                q = 'golden'
            elif not above_avg and high_use:
                q = 'problem'
            elif above_avg and not high_use:
                q = 'untapped'
            else:
                q = 'avoided'
            rect = patches.Rectangle(
                (x, y), cell_width, cell_height,
                linewidth=0.5, edgecolor='white', facecolor=QUADRANT_COLORS[q], alpha=0.8
            )
            ax.add_patch(rect)
            cx = x + cell_width / 2
            cy = y + cell_height / 2
            ax.text(cx, cy, q[0].upper(),
                    ha='center', va='center', fontsize=20,
                    color='white', fontweight='bold')

    add_grid_lines(ax)
    add_third_markers(ax)

    legend_patches = [
        patches.Patch(color=QUADRANT_COLORS[k], label=v)
        for k, v in legend_labels.items()
    ]
    ax.legend(handles=legend_patches, loc='lower left',
              fontsize=18, framealpha=0.8, fancybox=True)

    ax.set_title(f'{team_name} — Zone Utilization Efficiency',
                 fontsize=14, fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig(os.path.join(ue_dir, f'{safe_name}_zone_util_efficiency_quadrant.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Created quadrant map")

# ==============================================================================
# SUMMARY
# ==============================================================================

n_league    = 5
n_team_viz  = len([t for t in teams if t in team_grid_data['possession']]) * 7
n_compare   = 7
n_ue        = len([t for t in teams if t in team_grid_data['vs_league']])

print("\n" + "="*80)
print("GRID VISUALIZATION GENERATION COMPLETE")
print("="*80)
print(f"\nCreated:")
print(f"  - {n_league} league-wide grid visualizations (league_wide/)")
print(f"  - {n_team_viz} team-specific grid visualizations (by_team/)")
print(f"  - {n_compare} facet grid comparison visualizations (team_comparison_by_category/)")
print(f"  - {n_ue} zone utilization efficiency visualizations (zone_utilization_efficiency/)")
print(f"  Total: {n_league + n_team_viz + n_compare + n_ue} visualizations")
print(f"\nGrid: 6 columns x 3 rows = 18 zones")
print(f"  Defensive: cols 0-1 (x: 0-40 yd)")
print(f"  Midfield:  cols 2-3 (x: 40-80 yd)")
print(f"  Attacking: cols 4-5 (x: 80-120 yd)")
print(f"\nSaved to: {grid_dir}")
print("="*80)
