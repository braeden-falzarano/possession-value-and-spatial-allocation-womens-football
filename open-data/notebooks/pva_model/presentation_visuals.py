"""
One-off figures for the capstone slide deck.
  python presentation_visuals.py
"""

import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from mplsoccer import Pitch
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# PATHS
# ==============================================================================

script_dir    = os.path.dirname(os.path.abspath(__file__))
project_root  = os.path.dirname(os.path.dirname(script_dir))
processed_dir = os.path.join(project_root, 'data', 'processed')
results_dir   = os.path.join(project_root, 'results')
output_dir    = os.path.join(results_dir, 'presentation')
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# FIGURE 1: Metric Benchmark Dot Plot
# ==============================================================================

print("Creating Figure 1: Metric benchmark dot plot...")

pooled = pd.read_csv(os.path.join(results_dir, 'multi_league', 'metric_benchmark_pooled.csv'))

# GD/Game outcome only
gd = pooled[pooled['outcome'] == 'GD/Game'].copy()
gd = gd.sort_values('spearman_r', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(gd))
colors = ['#2c7bb6' if m != 'PVA/Game' else '#d7191c' for m in gd['metric']]
sizes = [100 if m != 'PVA/Game' else 180 for m in gd['metric']]
edge_widths = [0.5 if m != 'PVA/Game' else 2.0 for m in gd['metric']]

for i, (_, row) in enumerate(gd.iterrows()):
    lc = '#d7191c' if row['metric'] == 'PVA/Game' else '#cccccc'
    ax.plot([0, row['spearman_r']], [i, i], color=lc, linewidth=1.5, alpha=0.5, zorder=1)

ax.scatter(gd['spearman_r'], y_pos, c=colors, s=sizes, zorder=3,
           edgecolors='black', linewidths=edge_widths)

ax.set_yticks(y_pos)
ax.set_yticklabels(gd['metric'], fontsize=10)
ax.set_xlabel('Spearman r  (vs GD/Game)', fontsize=12)
ax.set_title('PVA Correlates with Season Outcomes on Par with xT and xG\n'
             f'(Pooled: N = {gd.iloc[0]["n"]} team-seasons across 2 leagues)',
             fontsize=19, fontweight='bold')
ax.set_xlim(0.55, 1.0)
ax.axvline(x=0.9, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='x', alpha=0.3, linestyle='--')

pva_dot = mpatches.Patch(color='#d7191c', label='PVA (this study)')
other_dot = mpatches.Patch(color='#2c7bb6', label='Baseline metrics')
ax.legend(handles=[pva_dot, other_dot], loc='lower right', fontsize=9,
          framealpha=0.9)

plt.tight_layout()
out_path = os.path.join(output_dir, 'metric_benchmark_dotplot.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {out_path}")

# ==============================================================================
# FIGURE 2: Possession-to-goal vector visualization
# ==============================================================================

print("Creating Figure 2: Possession-to-goal vector visualization...")

actions_df = pd.read_csv(
    os.path.join(processed_dir, 'fa_wsl_2018-19_actions_with_pva.csv'),
    low_memory=False
)
xt_df = pd.read_csv(os.path.join(processed_dir, 'fa_wsl_2018-19_xt_grid.csv'))
xt_grid = xt_df.set_index('cell_idx')['xt_value'].to_dict()

# PVA decomposition helpers
PITCH_LENGTH, PITCH_WIDTH = 120, 80
X_BINS, Y_BINS = 9, 6
PROGRESSIVE_PASS_BONUS_PER_10_YARDS = 0.000247
CARRY_PROGRESSIVE_BONUS_PER_10_YARDS = 0.001420
LINE_BREAKING_BONUS = 0.000771
PENALTY_BOX_ENTRY_BONUS = 0.008225

def _cell_idx(loc):
    if loc is None or not isinstance(loc, (list, tuple)) or len(loc) < 2:
        return None
    cx = min(int(loc[0] / (PITCH_LENGTH / X_BINS)), X_BINS - 1)
    cy = min(int(loc[1] / (PITCH_WIDTH / Y_BINS)), Y_BINS - 1)
    return cx * Y_BINS + cy

def _pitch_third(x):
    if x < 40: return 'defensive'
    if x < 80: return 'middle'
    return 'attacking'

def _crosses_third(s, e):
    if s is None or e is None: return False
    st, et = _pitch_third(s[0]), _pitch_third(e[0])
    return (st == 'defensive' and et in ('middle', 'attacking')) or \
           (st == 'middle' and et == 'attacking')

def _enters_pen_box(s, e):
    if s is None or e is None: return False
    in_box = lambda l: l[0] >= 102 and 18 <= l[1] <= 62
    return not in_box(s) and in_box(e)

def decompose_pva(action):
    """Return (base_delta_xt, total_bonus, list_of_bonus_names)."""
    atype = action['type']
    bonuses = []

    if atype == 'Pass':
        s = parse_loc(action.get('location'))
        e = parse_loc(action.get('pass_end_location'))
        if s is None or e is None: return (0, 0, [])
        sc, ec = _cell_idx(list(s)), _cell_idx(list(e))
        if sc is None or ec is None: return (0, 0, [])
        base = xt_grid.get(ec, 0) - xt_grid.get(sc, 0)
        bonus = 0
        fwd = e[0] - s[0]
        if fwd >= 10:
            avg_xt = (xt_grid.get(sc, 0) + xt_grid.get(ec, 0)) / 2
            mean_xt = np.mean(list(xt_grid.values()))
            b = (fwd / 10) * PROGRESSIVE_PASS_BONUS_PER_10_YARDS * (avg_xt / mean_xt)
            bonus += b
            bonuses.append('Progressive')
        if _crosses_third(list(s), list(e)):
            bonus += LINE_BREAKING_BONUS
            bonuses.append('Line-break')
        if pd.notna(action.get('pass_shot_assist')):
            bonus += 0.040554
            bonuses.append('Shot assist')
        return (base, bonus, bonuses)

    if atype == 'Carry':
        s = parse_loc(action.get('location'))
        e = parse_loc(action.get('carry_end_location'))
        if s is None or e is None: return (0, 0, [])
        sc, ec = _cell_idx(list(s)), _cell_idx(list(e))
        if sc is None or ec is None: return (0, 0, [])
        base = xt_grid.get(ec, 0) - xt_grid.get(sc, 0)
        bonus = 0
        fwd = e[0] - s[0]
        if fwd >= 10:
            avg_xt = (xt_grid.get(sc, 0) + xt_grid.get(ec, 0)) / 2
            mean_xt = np.mean(list(xt_grid.values()))
            b = (fwd / 10) * CARRY_PROGRESSIVE_BONUS_PER_10_YARDS * (avg_xt / mean_xt)
            bonus += b
            bonuses.append('Progressive')
        if _enters_pen_box(list(s), list(e)):
            bonus += PENALTY_BOX_ENTRY_BONUS
            bonuses.append('Pen box entry')
        if _crosses_third(list(s), list(e)):
            bonus += LINE_BREAKING_BONUS
            bonuses.append('Line-break')
        return (base, bonus, bonuses)

    if atype == 'Shot':
        s = parse_loc(action.get('location'))
        if s is None: return (action['pva'], 0, [])
        sc = _cell_idx(list(s))
        if sc is None: return (action['pva'], 0, [])
        xg = action.get('shot_statsbomb_xg', 0)
        xg = xg if pd.notna(xg) else 0
        base = xg - xt_grid.get(sc, 0)
        return (base, 0, ['xG-based'])

    # Defensive actions (Ball Recovery, Block, etc.)
    return (action['pva'], 0, ['Defensive'])

# Find goal-scoring possessions with enough buildup
shots = actions_df[actions_df['type'] == 'Shot'].copy()
goals = shots[shots['shot_outcome'] == 'Goal'].copy()

best_poss = None
best_score = 0

for _, goal in goals.iterrows():
    mid = goal['match_id']
    pid = goal['possession']
    team = goal['team']

    poss = actions_df[(actions_df['match_id'] == mid) &
                      (actions_df['possession'] == pid) &
                      (actions_df['team'] == team)].copy()
    poss = poss.sort_values('index')

    # 6-12 actions (readable on a slide)
    n_acts = len(poss)
    if n_acts < 6 or n_acts > 12:
        continue

    # at least 3 passes/carries
    buildup_types = poss['type'].isin(['Pass', 'Carry'])
    n_buildup = buildup_types.sum()
    if n_buildup < 3:
        continue

    # prefer possessions with a visible carry
    carries = poss[poss['type'] == 'Carry']
    max_carry_dist = 0
    for _, c in carries.iterrows():
        end_loc_str = c.get('carry_end_location')
        if pd.notna(end_loc_str):
            try:
                parsed = ast.literal_eval(str(end_loc_str))
                if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                    cd = np.sqrt((parsed[0] - c['x'])**2 + (parsed[1] - c['y'])**2)
                    max_carry_dist = max(max_carry_dist, cd)
            except (ValueError, SyntaxError):
                pass

    # need at least one 10+ yard carry
    if max_carry_dist < 10:
        continue

    # scoring: long carries + forward progression + PVA variety
    pva_range = poss['pva'].max() - poss['pva'].min()
    x_range = poss['x'].max() - poss['x'].min()
    start_depth = 120 - poss['x'].min()
    score = pva_range * 300 + x_range * 2 + start_depth + max_carry_dist * 20

    if score > best_score:
        best_score = score
        best_poss = poss.copy()

if best_poss is None:
    print("  No suitable possession found!")
else:
    poss = best_poss.reset_index(drop=True)
    team_name = poss.iloc[0]['team']
    match_id = poss.iloc[0]['match_id']

    print(f"  Selected: {team_name}, match {match_id}, {len(poss)} actions")

    def parse_loc(loc_str):
        if pd.isna(loc_str):
            return None
        try:
            parsed = ast.literal_eval(str(loc_str))
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                return (float(parsed[0]), float(parsed[1]))
        except (ValueError, SyntaxError):
            pass
        return None

    # pitch on left, action table on right
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    pitch = Pitch(pitch_type='statsbomb', pitch_color='#2d2d2d',
                  line_color='white', linewidth=1.5, goal_type='box')
    pitch.draw(ax=ax)

    pva_vals = poss['pva'].values
    # Use median of absolute non-zero values as scale anchor so the shot
    # doesn't squash all other values into the yellow center
    nonzero = np.abs(pva_vals[pva_vals != 0])
    scale = np.median(nonzero) * 3 if len(nonzero) > 0 else 0.001
    pva_norm = TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale)
    pva_cmap = LinearSegmentedColormap.from_list('RedYellowGreen', [
        (0.0,   '#67000d'),  # dark red (most negative)
        (0.35,  '#e53935'),  # red
        (0.46,  '#e53935'),  # red up to near zero
        (0.48,  '#ffdd00'),  # yellow band start
        (0.52,  '#ffdd00'),  # yellow band end
        (0.54,  '#66bb6a'),  # light green from near zero
        (0.65,  '#66bb6a'),  # light green
        (1.0,   '#2e7d32'),  # bright deep green (most positive)
    ], N=256)

    table_rows = []

    action_coords = []
    for i, (_, action) in enumerate(poss.iterrows()):
        start_x, start_y = action['x'], action['y']
        act_type = action['type']

        end_x, end_y = None, None
        if act_type == 'Pass':
            end_loc = parse_loc(action.get('pass_end_location'))
            if end_loc:
                end_x, end_y = end_loc
        elif act_type == 'Carry':
            end_loc = parse_loc(action.get('carry_end_location'))
            if end_loc:
                end_x, end_y = end_loc
        elif act_type == 'Shot':
            end_x, end_y = 120, start_y

        action_coords.append((start_x, start_y, end_x, end_y))

    # connector lines between consecutive actions
    for i in range(len(action_coords) - 1):
        sx, sy, ex, ey = action_coords[i]
        next_sx, next_sy = action_coords[i + 1][0], action_coords[i + 1][1]
        from_x = ex if ex is not None else sx
        from_y = ey if ey is not None else sy
        dist = np.sqrt((next_sx - from_x)**2 + (next_sy - from_y)**2)
        if dist > 1.0:
            ax.plot([from_x, next_sx], [from_y, next_sy],
                    color='#888888', linewidth=1.5, linestyle=':', alpha=0.6, zorder=2)

    # action arrows + numbered markers
    for i, (_, action) in enumerate(poss.iterrows()):
        start_x, start_y = action['x'], action['y']
        pva = action['pva']
        color = pva_cmap(pva_norm(pva))
        act_type = action['type']
        end_x, end_y = action_coords[i][2], action_coords[i][3]

        if end_x is not None:
            ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=3.0, mutation_scale=15),
                        zorder=4)

        ax.scatter(start_x, start_y, color=color, s=280, zorder=5,
                   edgecolors='black', linewidths=1.2, marker='o')

        ax.text(start_x, start_y, str(i + 1),
                fontsize=13, color='white', fontweight='bold',
                ha='center', va='center', zorder=7)

        base, bonus, bonus_names = decompose_pva(action)
        table_rows.append((i + 1, act_type, pva, color, base, bonus, bonus_names))

    total_pva = poss['pva'].sum()
    ax.set_title(f'{team_name} — Goal-Scoring Possession\n'
                 f'{len(poss)} actions  |  Total PVA: {total_pva:+.4f}',
                 fontsize=20, fontweight='bold', color='white', pad=10)

    sm = plt.cm.ScalarMappable(cmap=pva_cmap, norm=pva_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                        shrink=0.5, pad=0.06, aspect=30)
    cbar.set_label('PVA per Action', fontsize=16, color='white')
    cbar.ax.xaxis.set_tick_params(color='white', labelsize=14)
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color='white', fontsize=14)

    ax_table.set_facecolor('#2d2d2d')
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)
    ax_table.axis('off')

    n_rows = len(table_rows)
    row_height = min(0.055, 0.85 / (n_rows * 1.6))
    detail_gap = row_height * 0.55
    top = 0.95

    ax_table.text(0.08, top, '#', fontsize=14, color='#aaaaaa',
                  fontweight='bold', va='center')
    ax_table.text(0.18, top, 'Action', fontsize=14, color='#aaaaaa',
                  fontweight='bold', va='center')
    ax_table.text(0.58, top, 'Base', fontsize=14, color='#aaaaaa',
                  fontweight='bold', va='center', ha='right')
    ax_table.text(0.80, top, 'Bonus', fontsize=14, color='#aaaaaa',
                  fontweight='bold', va='center', ha='right')
    ax_table.text(0.98, top, 'PVA', fontsize=14, color='#aaaaaa',
                  fontweight='bold', va='center', ha='right')

    y = top - row_height
    for j, (num, atype, pva, color, base, bonus, bonus_names) in enumerate(table_rows):
        ax_table.add_patch(plt.Rectangle((0.0, y - row_height * 0.3), 0.035,
                           row_height * 0.6, facecolor=color, edgecolor='none',
                           transform=ax_table.transAxes, zorder=3))

        ax_table.text(0.08, y, str(num), fontsize=14, color='white',
                      fontweight='bold', va='center')
        ax_table.text(0.18, y, atype, fontsize=14, color='white',
                      va='center')

        ax_table.text(0.58, y, f'{base:+.4f}', fontsize=13, color=color,
                      va='center', ha='right', fontfamily='monospace')

        if abs(bonus) > 0.00005:
            ax_table.text(0.80, y, f'{bonus:+.4f}', fontsize=13, color='#00e5ff',
                          fontweight='bold', va='center', ha='right', fontfamily='monospace')
        else:
            ax_table.text(0.69, y, '—', fontsize=13, color='#555555',
                          va='center', ha='center')

        ax_table.text(0.98, y, f'{pva:+.4f}', fontsize=14, color=color,
                      fontweight='bold', va='center', ha='right', fontfamily='monospace')

        if bonus_names and bonus_names != ['Defensive'] and bonus_names != ['xG-based']:
            y -= detail_gap
            detail = ', '.join(bonus_names)
            ax_table.text(0.18, y, f'  └ {detail}', fontsize=11, color='#00e5ff',
                          va='center', alpha=0.8)
        elif bonus_names == ['xG-based']:
            y -= detail_gap
            ax_table.text(0.18, y, f'  └ xG − xT(zone)', fontsize=11, color='#aaaaaa',
                          va='center', alpha=0.8)

        y -= row_height

    y -= 0.005
    ax_table.plot([0.0, 1.0], [y + row_height * 0.4] * 2,
                  color='#aaaaaa', linewidth=0.8, transform=ax_table.transAxes)
    ax_table.text(0.18, y, 'Total', fontsize=14, color='white',
                  fontweight='bold', va='center')
    total_base = sum(r[4] for r in table_rows)
    total_bonus = sum(r[5] for r in table_rows)
    total_base_color = pva_cmap(pva_norm(total_base))
    total_pva_color = pva_cmap(pva_norm(total_pva))
    ax_table.text(0.58, y, f'{total_base:+.4f}', fontsize=13, color=total_base_color,
                  va='center', ha='right', fontfamily='monospace')
    ax_table.text(0.80, y, f'{total_bonus:+.4f}', fontsize=13, color='#00e5ff',
                  fontweight='bold', va='center', ha='right', fontfamily='monospace')
    ax_table.text(0.98, y, f'{total_pva:+.4f}', fontsize=14,
                  color=total_pva_color, fontweight='bold', va='center',
                  ha='right', fontfamily='monospace')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'possession_to_goal_vector.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#2d2d2d')
    plt.close()
    print(f"  Saved: {out_path}")

# ==============================================================================
# FIGURE 3: League-average xT Grid
# ==============================================================================

print("Creating Figure 3: League-average xT grid (6x3 zones)...")

xt_viz = pd.read_csv(os.path.join(processed_dir, 'fa_wsl_2018-19_xt_grid.csv'))
xt_9x6 = xt_viz['xt_value'].values.reshape(9, 6)

XT_PITCH_L, XT_PITCH_W = 120, 80
ZONE_X, ZONE_Y = 6, 3
zone_xt = np.zeros((ZONE_X, ZONE_Y))
zone_count = np.zeros((ZONE_X, ZONE_Y))
for cx9 in range(9):
    for cy6 in range(6):
        center_x = (cx9 + 0.5) * (XT_PITCH_L / 9)
        center_y = (cy6 + 0.5) * (XT_PITCH_W / 6)
        zx = min(int(center_x / (XT_PITCH_L / ZONE_X)), ZONE_X - 1)
        zy = min(int(center_y / (XT_PITCH_W / ZONE_Y)), ZONE_Y - 1)
        zone_xt[zx, zy] += xt_9x6[cx9, cy6]
        zone_count[zx, zy] += 1
zone_xt_avg = zone_xt / zone_count

cell_w = XT_PITCH_L / ZONE_X
cell_h = XT_PITCH_W / ZONE_Y

fig, ax = plt.subplots(figsize=(14, 9))
pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b',
              line_color='white', linewidth=1.5, goal_type='box')
pitch.draw(ax=ax)

xt_norm = Normalize(vmin=zone_xt_avg.min(), vmax=zone_xt_avg.max())
xt_cmap = plt.cm.YlOrRd

for cx in range(ZONE_X):
    for cy in range(ZONE_Y):
        val = zone_xt_avg[cx, cy]
        color = xt_cmap(xt_norm(val))
        x0 = cx * cell_w
        y0 = cy * cell_h

        rect = plt.Rectangle((x0, y0), cell_w, cell_h,
                              facecolor=color, edgecolor='white',
                              linewidth=1.2, alpha=0.75, zorder=2)
        ax.add_patch(rect)

        ax.text(x0 + cell_w / 2, y0 + cell_h / 2 + 2, f'{val:.4f}',
                fontsize=14, color='white', fontweight='bold',
                ha='center', va='center', zorder=3)
        zone_lbl = f'C{cx+1}_R{cy+1}'
        ax.text(x0 + cell_w / 2, y0 + cell_h / 2 - 6, zone_lbl,
                fontsize=9, color='white', ha='center', va='center',
                zorder=3, alpha=0.7)

ax.set_title('Expected Threat (xT) Grid — FA WSL League Average',
             fontsize=23, fontweight='bold', color='white', pad=12)

ax.annotate('Attacking Direction →', xy=(0.75, -0.04), xycoords='axes fraction',
            fontsize=11, color='white', fontweight='bold', ha='center')

sm = plt.cm.ScalarMappable(cmap=xt_cmap, norm=xt_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                    shrink=0.5, pad=0.08, aspect=30)
cbar.set_label('xT Value', fontsize=11, color='white')
cbar.ax.xaxis.set_tick_params(color='white')
plt.setp(cbar.ax.xaxis.get_ticklabels(), color='white')

plt.tight_layout()
out_path = os.path.join(output_dir, 'xt_grid_league_average.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#22312b')
plt.close()
print(f"  Saved: {out_path}")

print(f"\nAll presentation visuals saved to: {output_dir}")
