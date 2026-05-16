"""
KDE zone visualizations: continuous density-based heatmaps.
"""

LEAGUE_CONFIG = {
    'name': 'FA WSL 2018-19',
    'slug': 'fa_wsl_2018-19',
    'actions_csv': 'fa_wsl_2018-19/fa_wsl_2018-19_actions_with_pva.csv',
}

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from mplsoccer import Pitch
from scipy.stats import gaussian_kde
from pypalettes import load_cmap
import os

_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
_PROCESSED_DIR = os.path.join(_PROJECT_ROOT, 'data', 'processed')
_VIZ_ROOT      = os.path.join(_PROJECT_ROOT, 'visualizations', 'zone_analysis')

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

viz_dir = os.path.join(_VIZ_ROOT, LEAGUE_CONFIG['slug'])
kde_dir = os.path.join(viz_dir, 'KDE/league_wide')
os.makedirs(kde_dir, exist_ok=True)
kde_team_dir = os.path.join(viz_dir, 'KDE/by_team')
os.makedirs(kde_team_dir, exist_ok=True)

print("="*80)
print(f"{LEAGUE_CONFIG['name']} KDE VISUALIZATIONS - CONTINUOUS DENSITY")
print("="*80)

print("\n[Loading processed data...]")
actions_csv_path = os.path.join(_PROCESSED_DIR, LEAGUE_CONFIG['slug'] + '_actions_with_pva.csv')
actions_df = pd.read_csv(actions_csv_path)
print(f"[OK] Loaded {len(actions_df):,} actions with clean PVA values")

# Configuration
PITCH_LENGTH = 120
PITCH_WIDTH = 80
DEFENSIVE_THIRD_END = 40
MIDFIELD_THIRD_END = 80

def is_in_six_yard_box_area(x, y):
    if pd.isna(x) or pd.isna(y):
        return False
    return (x >= 114) and (x <= 120) and (y >= 30) and (y <= 50)

before_viz_filter = len(actions_df)
actions_df['in_six_yard'] = actions_df.apply(
    lambda row: is_in_six_yard_box_area(row['x'], row['y']), axis=1
)
actions_df_viz = actions_df[~actions_df['in_six_yard']].copy()
filtered_count = before_viz_filter - len(actions_df_viz)
print(f"[OK] Filtered {filtered_count:,} actions from 6-yard box area from visualizations")

def add_third_markers(ax):
    ax.axvline(x=DEFENSIVE_THIRD_END, color='yellow', linestyle='--',
               linewidth=2, alpha=0.5)
    ax.axvline(x=MIDFIELD_THIRD_END, color='yellow', linestyle='--',
               linewidth=2, alpha=0.5)
    ax.text(DEFENSIVE_THIRD_END/2, PITCH_WIDTH + 3, 'Defensive',
            ha='center', fontsize=18, fontweight='bold', color='yellow')
    ax.text((DEFENSIVE_THIRD_END + MIDFIELD_THIRD_END)/2, PITCH_WIDTH + 3, 'Midfield',
            ha='center', fontsize=18, fontweight='bold', color='yellow')
    ax.text((MIDFIELD_THIRD_END + PITCH_LENGTH)/2, PITCH_WIDTH + 3, 'Attacking',
            ha='center', fontsize=18, fontweight='bold', color='yellow')

def create_weighted_kde(x, y, weights, gridsize=100):
    """Weighted 2D KDE. Returns (Xi, Yi, Zi)."""
    xi = np.linspace(0, PITCH_LENGTH, gridsize)
    yi = np.linspace(0, PITCH_WIDTH, gridsize)
    Xi, Yi = np.meshgrid(xi, yi)

    positions = np.vstack([x, y])
    kernel = gaussian_kde(positions, weights=weights)

    grid_coords = np.vstack([Xi.ravel(), Yi.ravel()])
    Zi = kernel(grid_coords).reshape(Xi.shape)

    return Xi, Yi, Zi

def create_value_kde(x, y, values, gridsize=100, grid_bins=24):
    """Sign-preserving KDE via grid averaging + Gaussian smoothing."""
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import RegularGridInterpolator

    grid_width = PITCH_LENGTH / grid_bins
    grid_height = PITCH_WIDTH / grid_bins

    value_grid = np.zeros((grid_bins, grid_bins))
    count_grid = np.zeros((grid_bins, grid_bins))

    for i in range(len(x)):
        gx = int(np.clip(x[i] / grid_width, 0, grid_bins - 1))
        gy = int(np.clip(y[i] / grid_height, 0, grid_bins - 1))
        value_grid[gx, gy] += values[i]
        count_grid[gx, gy] += 1

    mask = count_grid > 0
    avg_value_grid = np.zeros_like(value_grid)
    avg_value_grid[mask] = value_grid[mask] / count_grid[mask]

    value_smooth = gaussian_filter(avg_value_grid, sigma=1.5)

    xi = np.linspace(0, PITCH_LENGTH, gridsize)
    yi = np.linspace(0, PITCH_WIDTH, gridsize)
    Xi, Yi = np.meshgrid(xi, yi)

    x_centers = np.linspace(grid_width/2, PITCH_LENGTH - grid_width/2, grid_bins)
    y_centers = np.linspace(grid_height/2, PITCH_WIDTH - grid_height/2, grid_bins)

    interpolator = RegularGridInterpolator((x_centers, y_centers), value_smooth,
                                          bounds_error=False, fill_value=0)

    points = np.array([Xi.ravel(), Yi.ravel()]).T
    Zi = interpolator(points).reshape(Xi.shape)

    return Xi, Yi, Zi

# ==============================================================================
# DANGER ZONE DEFINITION + LOO BASELINES
# ==============================================================================

DANGER_ZONE_X_MIN = 100
DANGER_ZONE_X_MAX = 120
DANGER_ZONE_Y_MIN = 80 / 3
DANGER_ZONE_Y_MAX = 2 * 80 / 3

def is_in_danger_zone(x, y):
    return (x >= DANGER_ZONE_X_MIN) and (x <= DANGER_ZONE_X_MAX) and \
           (y >= DANGER_ZONE_Y_MIN) and (y <= DANGER_ZONE_Y_MAX)

buildup_mask = ~actions_df_viz.apply(lambda row: is_in_danger_zone(row['x'], row['y']), axis=1)
actions_buildup = actions_df_viz[buildup_mask].copy()

print("\n[Pre-computing per-team KDE surfaces for LOO baselines...]")
_teams_kde     = sorted(actions_df_viz['team'].unique())
_team_pva_kdes = {}
_team_bu_kdes  = {}

_team_usage_kdes = {}
_team_bu_usage_kdes = {}

for _t in _teams_kde:
    _ta = actions_df_viz[actions_df_viz['team'] == _t]
    if len(_ta) < 50:
        continue
    _, _, _Zi = create_value_kde(_ta['x'].values, _ta['y'].values, _ta['pva'].values,
                                  gridsize=120, grid_bins=24)
    _team_pva_kdes[_t] = _Zi

    _w = np.ones(len(_ta)) / len(_ta)
    _, _, _Zi_u = create_weighted_kde(_ta['x'].values, _ta['y'].values, _w, gridsize=120)
    _team_usage_kdes[_t] = _Zi_u

    _bu = _ta[~_ta.apply(lambda row: is_in_danger_zone(row['x'], row['y']), axis=1)]
    if len(_bu) >= 50:
        _, _, _Zi_bu = create_value_kde(_bu['x'].values, _bu['y'].values, _bu['pva'].values,
                                         gridsize=120, grid_bins=24)
        _team_bu_kdes[_t] = _Zi_bu
        _w_bu = np.ones(len(_bu)) / len(_bu)
        _, _, _Zi_bu_u = create_weighted_kde(_bu['x'].values, _bu['y'].values, _w_bu, gridsize=120)
        _team_bu_usage_kdes[_t] = _Zi_bu_u

league_avg_kde_pva = np.mean(np.stack(list(_team_pva_kdes.values()), axis=0), axis=0)
league_avg_kde_bu  = np.mean(np.stack(list(_team_bu_kdes.values()),  axis=0), axis=0)

loo_pva_kdes   = {}
loo_bu_kdes    = {}
loo_usage_kdes = {}
loo_bu_usage_kdes = {}
for _t in _teams_kde:
    _other    = [tt for tt in _teams_kde if tt != _t and tt in _team_pva_kdes]
    _other_bu = [tt for tt in _teams_kde if tt != _t and tt in _team_bu_kdes]
    _other_u  = [tt for tt in _teams_kde if tt != _t and tt in _team_usage_kdes]
    _other_bu_u = [tt for tt in _teams_kde if tt != _t and tt in _team_bu_usage_kdes]
    if _other:
        loo_pva_kdes[_t] = np.mean(np.stack([_team_pva_kdes[tt] for tt in _other], axis=0), axis=0)
    if _other_bu:
        loo_bu_kdes[_t]  = np.mean(np.stack([_team_bu_kdes[tt]  for tt in _other_bu], axis=0), axis=0)
    if _other_u:
        loo_usage_kdes[_t] = np.mean(np.stack([_team_usage_kdes[tt] for tt in _other_u], axis=0), axis=0)
    if _other_bu_u:
        loo_bu_usage_kdes[_t] = np.mean(np.stack([_team_bu_usage_kdes[tt] for tt in _other_bu_u], axis=0), axis=0)

print(f"  [OK] LOO baselines ready: {len(loo_pva_kdes)} teams (possession), "
      f"{len(loo_bu_kdes)} teams (buildup), {len(loo_usage_kdes)} teams (usage), "
      f"{len(loo_bu_usage_kdes)} teams (buildup usage)")

# ==============================================================================
# VIZ 1: LEAGUE-WIDE POSSESSION VALUES (KDE)
# ==============================================================================

print("\n[Creating KDE Viz 1: League-Wide Zone Possession Values...]")

fig, ax = plt.subplots(figsize=(16, 10))
pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='white',
              linewidth=2, goal_type='box')
pitch.draw(ax=ax)

x_coords = actions_df_viz['x'].values
y_coords = actions_df_viz['y'].values
pva_values = actions_df_viz['pva'].values

print("  Computing value KDE...")
Xi, Yi, _ = create_value_kde(x_coords, y_coords, pva_values, gridsize=120, grid_bins=24)
Zi_league = league_avg_kde_pva

from matplotlib.colors import TwoSlopeNorm, Normalize
levels = 20

norm = Normalize(vmin=Zi_league.min(), vmax=Zi_league.max())

contourf = ax.contourf(Xi, Yi, Zi_league, levels=levels, cmap='RdYlGn', norm=norm, alpha=0.75)

cbar = plt.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Average PVA per action', rotation=270, labelpad=20, fontsize=20)
cbar.ax.tick_params(labelsize=16)

add_third_markers(ax)

ax.set_title(f'{LEAGUE_CONFIG["name"]} — Mean PVA (Continuous)',
             fontsize=15, fontweight='bold', pad=25)

plt.tight_layout()
plt.savefig(os.path.join(kde_dir, '01_league_possession_values_kde.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  [OK] Saved: 01_league_possession_values_kde.png")
plt.close()

# ==============================================================================
# VIZ 1B: BUILD-UP PLAY - POSSESSION VALUES (excluding danger zone)
# ==============================================================================

print("\n[Creating KDE Viz 1B: Build-Up Play (excluding danger zone)...]")
danger_zone_count = len(actions_df_viz) - len(actions_buildup)
print(f"  Filtered {danger_zone_count:,} actions from danger zone (18x20 area in front of goal)")

fig, ax = plt.subplots(figsize=(16, 10))
pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='white',
              linewidth=2, goal_type='box')
pitch.draw(ax=ax)

x_buildup = actions_buildup['x'].values
y_buildup = actions_buildup['y'].values
pva_buildup = actions_buildup['pva'].values

print("  Computing build-up play value KDE...")
Xi, Yi, _ = create_value_kde(x_buildup, y_buildup, pva_buildup, gridsize=120, grid_bins=24)
Zi_buildup_league = league_avg_kde_bu

norm = Normalize(vmin=Zi_buildup_league.min(), vmax=Zi_buildup_league.max())

contourf = ax.contourf(Xi, Yi, Zi_buildup_league, levels=levels, cmap='RdYlGn', norm=norm, alpha=0.75)

from matplotlib.patches import Rectangle
danger_rect = Rectangle((DANGER_ZONE_X_MIN, DANGER_ZONE_Y_MIN),
                         DANGER_ZONE_X_MAX - DANGER_ZONE_X_MIN,
                         DANGER_ZONE_Y_MAX - DANGER_ZONE_Y_MIN,
                         linewidth=2, edgecolor='white', facecolor='gray',
                         alpha=0.5, hatch='///', linestyle='--')
ax.add_patch(danger_rect)

cbar = plt.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Average PVA per action', rotation=270, labelpad=20, fontsize=20)
cbar.ax.tick_params(labelsize=16)

add_third_markers(ax)

ax.set_title(f'{LEAGUE_CONFIG["name"]} — Build-Up Play PVA (Continuous)\n' +
             'Excluding danger zone',
             fontsize=15, fontweight='bold', pad=25)

plt.tight_layout()
plt.savefig(os.path.join(kde_dir, '01b_league_buildup_play_kde.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  [OK] Saved: 01b_league_buildup_play_kde.png")
plt.close()

# ==============================================================================
# VIZ 2: LEAGUE-WIDE USAGE (KDE)
# ==============================================================================

print("\n[Creating KDE Viz 2: League-Wide Zone Usage...]")

fig, ax = plt.subplots(figsize=(16, 10))
pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='white',
              linewidth=2, goal_type='box')
pitch.draw(ax=ax)

usage_weights = np.ones(len(x_coords)) / len(x_coords)

print("  Computing usage KDE...")
Xi, Yi, Zi = create_weighted_kde(x_coords, y_coords, usage_weights, gridsize=120)

contourf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap='YlOrRd', alpha=0.75)

cbar = plt.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Action Density', rotation=270, labelpad=20, fontsize=20)
cbar.ax.tick_params(labelsize=16)

add_third_markers(ax)

ax.set_title(f'{LEAGUE_CONFIG["name"]} — Action Density (Continuous)',
             fontsize=15, fontweight='bold', pad=25)

plt.tight_layout()
plt.savefig(os.path.join(kde_dir, '02_league_usage_kde.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  [OK] Saved: 02_league_usage_kde.png")
plt.close()

# ==============================================================================
# VIZ 3: THIRDS-NORMALIZED POSSESSION VALUES (KDE)
# ==============================================================================

print("\n[Creating KDE Viz 3: Thirds-Normalized Possession Values...]")

fig, ax = plt.subplots(figsize=(16, 10))
pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='white',
              linewidth=2, goal_type='box')
pitch.draw(ax=ax)

defensive_cmap = LinearSegmentedColormap.from_list('defensive',
    ['#08519c', '#f7f7f7', '#6baed6'])
midfield_cmap = LinearSegmentedColormap.from_list('midfield',
    ['#006d2c', '#f7f7f7', '#74c476'])
attacking_cmap = LinearSegmentedColormap.from_list('attacking',
    ['#d94801', '#f7f7f7', '#fdae6b'])

print("  Computing KDE by third...")

for third_name, x_min, x_max, cmap in [
    ('defensive', 0, DEFENSIVE_THIRD_END, defensive_cmap),
    ('midfield', DEFENSIVE_THIRD_END, MIDFIELD_THIRD_END, midfield_cmap),
    ('attacking', MIDFIELD_THIRD_END, PITCH_LENGTH, attacking_cmap)
]:
    third_actions = actions_df_viz[(actions_df_viz['x'] >= x_min) &
                                    (actions_df_viz['x'] < x_max)]

    if len(third_actions) == 0:
        continue

    x_third = third_actions['x'].values
    y_third = third_actions['y'].values
    pva_third = third_actions['pva'].values

    pva_third_norm = (pva_third - pva_third.mean()) / pva_third.std()
    pva_third_shifted = pva_third_norm - pva_third_norm.min() + 0.001
    weights_third = pva_third_shifted / pva_third_shifted.sum()

    Xi, Yi, Zi = create_weighted_kde(x_third, y_third, weights_third, gridsize=120)

    mask = (Xi >= x_min) & (Xi < x_max)
    Zi_masked = np.where(mask, Zi, np.nan)

    contourf = ax.contourf(Xi, Yi, Zi_masked, levels=15, cmap=cmap, alpha=0.75)

add_third_markers(ax)

ax.set_title(f'{LEAGUE_CONFIG["name"]} — PVA Normalized within Thirds (Continuous)',
             fontsize=15, fontweight='bold', pad=25)

plt.tight_layout()
plt.savefig(os.path.join(kde_dir, '03_league_possession_thirds_normalized_kde.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  [OK] Saved: 03_league_possession_thirds_normalized_kde.png")
plt.close()

print(f"\n[OK] Created 4 league-wide KDE visualizations")

# ==============================================================================
# VIZ 4: THIRDS-NORMALIZED USAGE (KDE)
# ==============================================================================

print("\n[Creating KDE Viz 4: Thirds-Normalized Usage...]")

fig, ax = plt.subplots(figsize=(16, 10))
pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='white',
              linewidth=2, goal_type='box')
pitch.draw(ax=ax)

print("  Computing usage KDE by third...")

for third_name, x_min, x_max, cmap in [
    ('defensive', 0, DEFENSIVE_THIRD_END, defensive_cmap),
    ('midfield', DEFENSIVE_THIRD_END, MIDFIELD_THIRD_END, midfield_cmap),
    ('attacking', MIDFIELD_THIRD_END, PITCH_LENGTH, attacking_cmap)
]:
    third_actions = actions_df_viz[(actions_df_viz['x'] >= x_min) &
                                    (actions_df_viz['x'] < x_max)]

    if len(third_actions) == 0:
        continue

    x_third = third_actions['x'].values
    y_third = third_actions['y'].values

    weights_third = np.ones(len(x_third)) / len(x_third)

    Xi, Yi, Zi = create_weighted_kde(x_third, y_third, weights_third, gridsize=120)

    mask = (Xi >= x_min) & (Xi < x_max)
    Zi_masked = np.where(mask, Zi, np.nan)

    contourf = ax.contourf(Xi, Yi, Zi_masked, levels=15, cmap=cmap, alpha=0.75)

add_third_markers(ax)

ax.set_title(f'{LEAGUE_CONFIG["name"]} — Usage Normalized within Thirds (Continuous)',
             fontsize=15, fontweight='bold', pad=25)

plt.tight_layout()
plt.savefig(os.path.join(kde_dir, '04_league_usage_thirds_normalized_kde.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
print("  [OK] Saved: 04_league_usage_thirds_normalized_kde.png")
plt.close()

# ==============================================================================
# TEAM-SPECIFIC KDE VISUALIZATIONS
# ==============================================================================

print("\n[Creating team-specific KDE visualizations...]")

teams = sorted(actions_df_viz['team'].unique())

team_kde_data = {
    'possession': {},
    'buildup': {},
    'usage': {},
    'vs_league': {},
    'buildup_vs_league': {},
    'zva_contribution': {},
    'zva_buildup': {},
}

for team_idx, team_name in enumerate(teams, 1):
    print(f"\n  [{team_idx}/{len(teams)}] {team_name}...")

    team_actions = actions_df_viz[actions_df_viz['team'] == team_name]
    safe_team_name = team_name.replace(' ', '_').replace('/', '_')

    if len(team_actions) < 50:
        print(f"    Skipped (too few actions)")
        continue

    fig, ax = plt.subplots(figsize=(16, 10))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='white',
                  linewidth=2, goal_type='box')
    pitch.draw(ax=ax)

    x_team = team_actions['x'].values
    y_team = team_actions['y'].values
    pva_team = team_actions['pva'].values

    Xi, Yi, Zi = create_value_kde(x_team, y_team, pva_team, gridsize=120, grid_bins=24)

    team_kde_data['possession'][team_name] = (Xi, Yi, Zi)

    from matplotlib.colors import Normalize
    norm = Normalize(vmin=Zi.min(), vmax=Zi.max())

    contourf = ax.contourf(Xi, Yi, Zi, levels=20, cmap='RdYlGn', norm=norm, alpha=0.75)
    cbar = plt.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Possession Value Density', rotation=270, labelpad=20, fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    add_third_markers(ax)
    ax.set_title(f'{team_name} — Mean PVA (n={len(team_actions):,})',
                 fontsize=15, fontweight='bold', pad=25)

    plt.tight_layout()
    plt.savefig(os.path.join(kde_team_dir, f'{safe_team_name}_possession_kde.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    team_buildup_mask = ~team_actions.apply(lambda row: is_in_danger_zone(row['x'], row['y']), axis=1)
    team_buildup = team_actions[team_buildup_mask].copy()

    if len(team_buildup) >= 50:
        fig, ax = plt.subplots(figsize=(16, 10))
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='white',
                      linewidth=2, goal_type='box')
        pitch.draw(ax=ax)

        x_buildup_team = team_buildup['x'].values
        y_buildup_team = team_buildup['y'].values
        pva_buildup_team = team_buildup['pva'].values

        Xi, Yi, Zi = create_value_kde(x_buildup_team, y_buildup_team, pva_buildup_team, gridsize=120, grid_bins=24)

        team_kde_data['buildup'][team_name] = (Xi, Yi, Zi)

        norm = Normalize(vmin=Zi.min(), vmax=Zi.max())

        contourf = ax.contourf(Xi, Yi, Zi, levels=20, cmap='RdYlGn', norm=norm, alpha=0.75)

        from matplotlib.patches import Rectangle
        danger_rect = Rectangle((DANGER_ZONE_X_MIN, DANGER_ZONE_Y_MIN),
                                 DANGER_ZONE_X_MAX - DANGER_ZONE_X_MIN,
                                 DANGER_ZONE_Y_MAX - DANGER_ZONE_Y_MIN,
                                 linewidth=2, edgecolor='white', facecolor='gray',
                                 alpha=0.5, hatch='///', linestyle='--')
        ax.add_patch(danger_rect)

        cbar = plt.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Average PVA per action', rotation=270, labelpad=20, fontsize=20)
        cbar.ax.tick_params(labelsize=16)

        add_third_markers(ax)
        ax.set_title(f'{team_name} — Build-Up Play PVA (n={len(team_buildup):,})\nExcluding danger zone',
                     fontsize=15, fontweight='bold', pad=25)

        plt.tight_layout()
        plt.savefig(os.path.join(kde_team_dir, f'{safe_team_name}_buildup_kde.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    fig, ax = plt.subplots(figsize=(16, 10))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='white',
                  linewidth=2, goal_type='box')
    pitch.draw(ax=ax)

    weights_usage = np.ones(len(x_team)) / len(x_team)
    Xi, Yi, Zi = create_weighted_kde(x_team, y_team, weights_usage, gridsize=120)

    team_kde_data['usage'][team_name] = (Xi, Yi, Zi)

    contourf = ax.contourf(Xi, Yi, Zi, levels=20, cmap='YlOrRd', alpha=0.75)
    cbar = plt.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Action Density', rotation=270, labelpad=20, fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    add_third_markers(ax)
    ax.set_title(f'{team_name} — Action Density (n={len(team_actions):,})',
                 fontsize=15, fontweight='bold', pad=25)

    plt.tight_layout()
    plt.savefig(os.path.join(kde_team_dir, f'{safe_team_name}_usage_kde.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    if team_name in loo_usage_kdes and team_name in loo_pva_kdes:
        Zi_usage_diff = team_kde_data['usage'][team_name][2] - loo_usage_kdes[team_name]
        Xi_poss, Yi_poss, _ = team_kde_data['possession'][team_name]
        Zi_zva = Zi_usage_diff * loo_pva_kdes[team_name]

        team_kde_data['zva_contribution'][team_name] = (Xi_poss, Yi_poss, Zi_zva)

        fig, ax = plt.subplots(figsize=(16, 10))
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='white',
                      linewidth=2, goal_type='box')
        pitch.draw(ax=ax)

        vmax_zva = max(abs(Zi_zva.min()), abs(Zi_zva.max())) or 0.001
        norm_zva = TwoSlopeNorm(vmin=-vmax_zva, vcenter=0, vmax=vmax_zva)
        cf = ax.contourf(Xi_poss, Yi_poss, Zi_zva, levels=20, cmap=zva_cmap, norm=norm_zva, alpha=0.75)
        cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('ZVA Contribution', rotation=270, labelpad=20, fontsize=20)
        cbar.ax.tick_params(labelsize=16)

        add_third_markers(ax)
        ax.set_title(f'{team_name} — ZVA Contribution (n={len(team_actions):,})',
                     fontsize=15, fontweight='bold', pad=25)
        plt.tight_layout()
        plt.savefig(os.path.join(kde_team_dir, f'{safe_team_name}_zva_contribution_kde.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    if (team_name in loo_bu_usage_kdes and team_name in team_kde_data['buildup']
            and team_name in loo_bu_kdes):
        Zi_bu_usage_diff = _team_bu_usage_kdes[team_name] - loo_bu_usage_kdes[team_name]
        Xi_bu, Yi_bu, _ = team_kde_data['buildup'][team_name]
        Zi_bu_zva = Zi_bu_usage_diff * loo_bu_kdes[team_name]

        team_kde_data['zva_buildup'][team_name] = (Xi_bu, Yi_bu, Zi_bu_zva)

        fig, ax = plt.subplots(figsize=(16, 10))
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='white',
                      linewidth=2, goal_type='box')
        pitch.draw(ax=ax)

        vmax_bu = max(abs(Zi_bu_zva.min()), abs(Zi_bu_zva.max())) or 0.001
        norm_bu = TwoSlopeNorm(vmin=-vmax_bu, vcenter=0, vmax=vmax_bu)
        cf_bu = ax.contourf(Xi_bu, Yi_bu, Zi_bu_zva, levels=20, cmap=zva_cmap, norm=norm_bu, alpha=0.75)
        cbar_bu = plt.colorbar(cf_bu, ax=ax, fraction=0.046, pad=0.04)
        cbar_bu.set_label('Build-Up ZVA Contribution', rotation=270, labelpad=20, fontsize=20)
        cbar_bu.ax.tick_params(labelsize=16)

        danger_rect = Rectangle((DANGER_ZONE_X_MIN, DANGER_ZONE_Y_MIN),
                                DANGER_ZONE_X_MAX - DANGER_ZONE_X_MIN,
                                DANGER_ZONE_Y_MAX - DANGER_ZONE_Y_MIN,
                                linewidth=2, edgecolor='white', facecolor='gray',
                                alpha=0.5, hatch='///', linestyle='--')
        ax.add_patch(danger_rect)

        add_third_markers(ax)
        ax.set_title(f'{team_name} — Build-Up ZVA Contribution (n={len(team_buildup):,})\nExcluding danger zone',
                     fontsize=15, fontweight='bold', pad=25)
        plt.tight_layout()
        plt.savefig(os.path.join(kde_team_dir, f'{safe_team_name}_zva_buildup_kde.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    fig, ax = plt.subplots(figsize=(16, 10))
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='white',
                  linewidth=2, goal_type='box')
    pitch.draw(ax=ax)

    Xi_team, Yi_team, Zi_team = team_kde_data['possession'][team_name]
    Zi_diff = Zi_team - loo_pva_kdes[team_name]

    team_kde_data['vs_league'][team_name] = (Xi_team, Yi_team, Zi_diff)

    vmax_diff = max(abs(Zi_diff.min()), abs(Zi_diff.max())) or 0.001
    norm = TwoSlopeNorm(vmin=-vmax_diff, vcenter=0, vmax=vmax_diff)

    contourf = ax.contourf(Xi_team, Yi_team, Zi_diff, levels=20, cmap=buildup_vs_league_cmap, norm=norm, alpha=0.75)
    cbar = plt.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('PVA vs League Avg', rotation=270, labelpad=20, fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    add_third_markers(ax)
    ax.set_title(f'{team_name} — PVA vs League Average (n={len(team_actions):,})',
                 fontsize=15, fontweight='bold', pad=25)

    plt.tight_layout()
    plt.savefig(os.path.join(kde_team_dir, f'{safe_team_name}_vs_league_kde.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    if len(team_buildup) >= 50:
        fig, ax = plt.subplots(figsize=(16, 10))
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='white',
                      linewidth=2, goal_type='box')
        pitch.draw(ax=ax)

        Xi_team_buildup, Yi_team_buildup, Zi_team_buildup = team_kde_data['buildup'][team_name]
        Zi_buildup_diff = Zi_team_buildup - loo_bu_kdes[team_name]

        team_kde_data['buildup_vs_league'][team_name] = (Xi_team_buildup, Yi_team_buildup, Zi_buildup_diff)

        vmax_bu_diff = max(abs(Zi_buildup_diff.min()), abs(Zi_buildup_diff.max())) or 0.001
        norm = TwoSlopeNorm(vmin=-vmax_bu_diff, vcenter=0, vmax=vmax_bu_diff)

        contourf = ax.contourf(Xi_team_buildup, Yi_team_buildup, Zi_buildup_diff,
                              levels=20, cmap=buildup_vs_league_cmap, norm=norm, alpha=0.75)

        danger_rect = Rectangle((DANGER_ZONE_X_MIN, DANGER_ZONE_Y_MIN),
                                DANGER_ZONE_X_MAX - DANGER_ZONE_X_MIN,
                                DANGER_ZONE_Y_MAX - DANGER_ZONE_Y_MIN,
                                linewidth=2, edgecolor='white', facecolor='gray',
                                alpha=0.5, hatch='///', linestyle='--')
        ax.add_patch(danger_rect)

        cbar = plt.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Build-Up PVA vs League Avg', rotation=270, labelpad=20, fontsize=20)
        cbar.ax.tick_params(labelsize=16)

        add_third_markers(ax)
        ax.set_title(f'{team_name} — Build-Up PVA vs League Average (n={len(team_buildup):,})\nExcluding danger zone',
                     fontsize=15, fontweight='bold', pad=25)

        plt.tight_layout()
        plt.savefig(os.path.join(kde_team_dir, f'{safe_team_name}_buildup_vs_league_kde.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    print(f"    [OK] Created 7 KDE visualizations (incl. ZVA + buildup ZVA)")

# ==============================================================================
# FACET GRID COMPARISONS - ALL TEAMS BY CATEGORY
# ==============================================================================

print("\n" + "="*80)
print("CREATING FACET GRID COMPARISONS")
print("="*80)
print("\nGenerating 6 multi-team comparison visualizations...")

kde_base_dir = os.path.join(viz_dir, 'KDE')
comparison_dir = os.path.join(kde_base_dir, 'team_comparison_by_category')
os.makedirs(comparison_dir, exist_ok=True)

n_teams = len(teams)
n_cols = 4
n_rows = (n_teams + n_cols - 1) // n_cols  # Ceiling division

def create_facet_grid(data_dict, title, filename, cmap='RdYlGn', show_danger_zone=False,
                      use_symmetric_norm=True, norm_percentile=None):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 6*n_rows))
    axes = axes.flatten() if n_teams > 1 else [axes]

    all_z_values = []
    for team_name in teams:
        if team_name in data_dict:
            _, _, Zi = data_dict[team_name]
            all_z_values.extend(Zi.flatten())

    if len(all_z_values) > 0:
        all_z_arr = np.array(all_z_values)

        if use_symmetric_norm:
            if norm_percentile is not None:
                vmax = np.percentile(np.abs(all_z_arr), norm_percentile) or 0.001
            else:
                vmax = max(abs(all_z_arr.min()), abs(all_z_arr.max())) or 0.001
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        else:
            if norm_percentile is not None:
                global_min = np.percentile(all_z_arr, 100 - norm_percentile)
                global_max = np.percentile(all_z_arr, norm_percentile)
            else:
                global_min = all_z_arr.min()
                global_max = all_z_arr.max()
            norm = Normalize(vmin=global_min, vmax=global_max)
    else:
        norm = None

    for idx, team_name in enumerate(teams):
        ax = axes[idx]

        if team_name not in data_dict:
            ax.axis('off')
            continue

        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='white',
                      linewidth=1.5, goal_type='box')
        pitch.draw(ax=ax)

        Xi, Yi, Zi = data_dict[team_name]

        contourf = ax.contourf(Xi, Yi, Zi, levels=30, cmap=cmap, norm=norm, alpha=0.75)

        if show_danger_zone:
            danger_rect = Rectangle((DANGER_ZONE_X_MIN, DANGER_ZONE_Y_MIN),
                                    DANGER_ZONE_X_MAX - DANGER_ZONE_X_MIN,
                                    DANGER_ZONE_Y_MAX - DANGER_ZONE_Y_MIN,
                                    linewidth=1, edgecolor='white', facecolor='gray',
                                    alpha=0.3, hatch='///', linestyle='--')
            ax.add_patch(danger_rect)

        add_third_markers(ax)

        ax.set_title(team_name, fontsize=22, fontweight='bold', pad=10)

    for idx in range(n_teams, len(axes)):
        axes[idx].axis('off')

    if norm is not None:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(contourf, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=16)

    fig.suptitle(title, fontsize=42, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig(os.path.join(comparison_dir, filename), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: {filename}")

create_facet_grid(
    team_kde_data['possession'],
    f'{LEAGUE_CONFIG["name"]} — Mean PVA (All Teams)',
    '01_all_teams_possession_values.png',
    cmap='RdYlGn',
    use_symmetric_norm=False
)

create_facet_grid(
    team_kde_data['buildup'],
    f'{LEAGUE_CONFIG["name"]} — Build-Up Play PVA (All Teams)',
    '02_all_teams_buildup_play.png',
    cmap='RdYlGn',
    show_danger_zone=True,
    use_symmetric_norm=False
)

create_facet_grid(
    team_kde_data['usage'],
    f'{LEAGUE_CONFIG["name"]} — Action Density (All Teams)',
    '03_all_teams_usage_density.png',
    cmap='YlOrRd',
    use_symmetric_norm=False,
    norm_percentile=98
)

create_facet_grid(
    team_kde_data['vs_league'],
    f'{LEAGUE_CONFIG["name"]} — PVA vs League Average (All Teams)',
    '04_all_teams_vs_league_avg.png',
    cmap=buildup_vs_league_cmap,
    use_symmetric_norm=True
)

create_facet_grid(
    team_kde_data['buildup_vs_league'],
    f'{LEAGUE_CONFIG["name"]} — Build-Up PVA vs League Average (All Teams)',
    '05_all_teams_buildup_vs_league_avg.png',
    cmap=buildup_vs_league_cmap,
    show_danger_zone=True,
    use_symmetric_norm=True,
    norm_percentile=98
)

create_facet_grid(
    team_kde_data['zva_contribution'],
    f'{LEAGUE_CONFIG["name"]} — Zone Value Added Contribution (All Teams)',
    '06_all_teams_zva_contribution.png',
    cmap=zva_cmap,
    use_symmetric_norm=True
)

create_facet_grid(
    team_kde_data['zva_buildup'],
    f'{LEAGUE_CONFIG["name"]} — Build-Up ZVA Contribution (All Teams)',
    '07_all_teams_zva_buildup.png',
    cmap=zva_cmap,
    show_danger_zone=True,
    use_symmetric_norm=True
)

print(f"\n[OK] Created 7 facet grid comparison visualizations")
print(f"Saved to: {comparison_dir}")

print("\n" + "="*80)
print("KDE VISUALIZATION GENERATION COMPLETE")
print("="*80)
print(f"\nCreated:")
print(f"  - 5 league-wide KDE visualizations")
print(f"  - {len(teams)*7} team-specific KDE visualizations (incl. ZVA + buildup ZVA)")
print(f"  - 7 facet grid comparison visualizations (all teams by category)")
print(f"Saved to: {kde_dir}")
print("="*80)
