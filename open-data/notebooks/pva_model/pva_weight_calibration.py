"""
Empirical calibration of PVA bonus weights via matched comparison on
terminal xG (actions with vs without each feature, matched on end cell).
  python pva_weight_calibration.py
"""

import os
import ast
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PITCH_LENGTH = 120
PITCH_WIDTH  = 80
X_BINS = 9
Y_BINS = 6
DEFENSIVE_THIRD_END = 40
MIDFIELD_THIRD_END  = 80

MIN_OBS = 10

LEAGUES = [
    {'name': 'NWSL 2018',       'slug': 'nwsl_2018',       'csv_suffix': '_actions_with_pva_clean.csv'},
    {'name': 'FA WSL 2018-19',  'slug': 'fa_wsl_2018-19',  'csv_suffix': '_actions_with_pva.csv'},
    {'name': 'FA WSL 2019-20',  'slug': 'fa_wsl_2019-20',  'csv_suffix': '_actions_with_pva.csv'},
    {'name': 'FA WSL 2020-21',  'slug': 'fa_wsl_2020-21',  'csv_suffix': '_actions_with_pva.csv'},
]

HAND_TUNED = {
    'prog_pass':   0.025,
    'prog_carry':  0.015,
    'line_break':  0.020,
    'pen_box':     0.030,
    'shot_assist': 0.060,
    'pressure':    0.25,   # PRESSURE_MULTIPLIER - 1
    'dribble':     0.015,
}

# ==============================================================================
# PATHS
# ==============================================================================

script_dir    = os.path.dirname(os.path.abspath(__file__))
project_root  = os.path.dirname(os.path.dirname(script_dir))
processed_dir = os.path.join(project_root, 'data', 'processed')
results_dir   = os.path.join(project_root, 'results', 'multi_league')
os.makedirs(results_dir, exist_ok=True)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def parse_location(loc_str):
    if pd.isna(loc_str):
        return None
    try:
        parsed = ast.literal_eval(str(loc_str))
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return (float(parsed[0]), float(parsed[1]))
    except (ValueError, SyntaxError):
        pass
    return None

def compute_cell_idx(x, y):
    col = min(int(x / (PITCH_LENGTH / X_BINS)), X_BINS - 1)
    row = min(int(y / (PITCH_WIDTH / Y_BINS)), Y_BINS - 1)
    return col * Y_BINS + row

def get_pitch_third(x):
    if x < DEFENSIVE_THIRD_END: return 'defensive'
    if x < MIDFIELD_THIRD_END:  return 'middle'
    return 'attacking'

def crosses_third_forward(start_x, end_x):
    s = get_pitch_third(start_x)
    e = get_pitch_third(end_x)
    return ((s == 'defensive' and e in ['middle', 'attacking']) or
            (s == 'middle' and e == 'attacking'))

def enters_penalty_box(start_x, start_y, end_x, end_y):
    in_start = start_x >= 102 and 18 <= start_y <= 62
    in_end   = end_x >= 102 and 18 <= end_y <= 62
    return (not in_start) and in_end

def section(title):
    print(f"\n{'=' * 80}\n  {title}\n{'=' * 80}")


def matched_comparison(feature_df, control_df, cell_col, outcome_col, min_obs=MIN_OBS):
    """Weighted average outcome difference, matched by cell."""
    feat_mean = feature_df.groupby(cell_col)[outcome_col].agg(['mean', 'count'])
    ctrl_mean = control_df.groupby(cell_col)[outcome_col].agg(['mean', 'count'])

    common = feat_mean.index.intersection(ctrl_mean.index)
    diffs, weights = [], []
    details = []
    for c in common:
        nf = feat_mean.loc[c, 'count']
        nc = ctrl_mean.loc[c, 'count']
        if nf < min_obs or nc < min_obs:
            continue
        d = feat_mean.loc[c, 'mean'] - ctrl_mean.loc[c, 'mean']
        diffs.append(d)
        weights.append(nf)
        details.append({'cell': c, 'diff': d, 'n_feat': int(nf), 'n_ctrl': int(nc)})

    if not diffs:
        return 0.0, 0, 0, []
    bonus = float(np.average(diffs, weights=weights))
    return bonus, len(diffs), int(sum(weights)), details

# ==============================================================================
# LOAD DATA & TAG WITH TERMINAL xG
# ==============================================================================

section("PVA WEIGHT CALIBRATION — Empirical Feature Consequence Model")

all_actions = []

for league in LEAGUES:
    section(f"Processing {league['name']}")

    csv_path = os.path.join(processed_dir, league['slug'] + league['csv_suffix'])
    df = pd.read_csv(csv_path, low_memory=False)

    xt_path = os.path.join(processed_dir, league['slug'] + '_xt_grid.csv')
    xt_df = pd.read_csv(xt_path)
    xt_grid = xt_df.set_index('cell_idx')['xt_value'].to_dict()
    mean_xt = np.mean(list(xt_grid.values()))

    print(f"  {len(df):,} actions, {df['match_id'].nunique()} matches")

    # Possession terminal xG
    shots = df[df['type'] == 'Shot'].copy()
    shots_sorted = shots.sort_values('index') if 'index' in shots.columns else shots
    term_xg = shots_sorted.groupby(['match_id', 'possession']).agg(
        terminal_xg=('shot_statsbomb_xg', 'last')
    ).reset_index()

    # Parse end locations
    passes = df[df['type'] == 'Pass'].copy()
    passes['end_xy'] = passes['pass_end_location'].apply(parse_location)
    passes['end_x'] = passes['end_xy'].apply(lambda xy: xy[0] if xy else None)
    passes['end_y'] = passes['end_xy'].apply(lambda xy: xy[1] if xy else None)
    passes['end_cell'] = passes['end_xy'].apply(
        lambda xy: compute_cell_idx(xy[0], xy[1]) if xy else None)
    passes_complete = passes[passes['pass_outcome'].isna()].copy()

    carries = df[df['type'] == 'Carry'].copy()
    carries['end_xy'] = carries['carry_end_location'].apply(parse_location)
    carries['end_x'] = carries['end_xy'].apply(lambda xy: xy[0] if xy else None)
    carries['end_y'] = carries['end_xy'].apply(lambda xy: xy[1] if xy else None)
    carries['end_cell'] = carries['end_xy'].apply(
        lambda xy: compute_cell_idx(xy[0], xy[1]) if xy else None)

    dribbles_complete = df[(df['type'] == 'Dribble') & (df['dribble_outcome'] == 'Complete')].copy()
    dribbles_incomplete = df[(df['type'] == 'Dribble') & (df['dribble_outcome'] != 'Complete')].copy()

    # Tag features and merge terminal xG
    pc = passes_complete[passes_complete['end_cell'].notna()].copy()
    pc['fwd_dist'] = pc['end_x'] - pc['x']
    pc['is_progressive'] = pc['fwd_dist'] >= 10
    pc['is_line_break'] = pc.apply(
        lambda r: crosses_third_forward(r['x'], r['end_x']), axis=1)
    pc['is_pen_box'] = pc.apply(
        lambda r: enters_penalty_box(r['x'], r['y'], r['end_x'], r['end_y'])
        if r['end_x'] is not None and r['end_y'] is not None else False, axis=1)
    pc['is_shot_assist'] = pc['pass_shot_assist'].notna()
    pc = pc.merge(term_xg, on=['match_id', 'possession'], how='left')
    pc['terminal_xg'] = pc['terminal_xg'].fillna(0.0)

    # Carries
    ca = carries[carries['end_cell'].notna()].copy()
    ca['fwd_dist'] = ca['end_x'] - ca['x']
    ca['is_progressive'] = ca['fwd_dist'] >= 10
    ca['is_line_break'] = ca.apply(
        lambda r: crosses_third_forward(r['x'], r['end_x']), axis=1)
    ca['is_pen_box'] = ca.apply(
        lambda r: enters_penalty_box(r['x'], r['y'], r['end_x'], r['end_y'])
        if r['end_x'] is not None and r['end_y'] is not None else False, axis=1)
    ca['is_under_pressure'] = ca['under_pressure'] == True
    ca = ca.merge(term_xg, on=['match_id', 'possession'], how='left')
    ca['terminal_xg'] = ca['terminal_xg'].fillna(0.0)

    # Dribbles
    dc = dribbles_complete.copy()
    dc = dc.merge(term_xg, on=['match_id', 'possession'], how='left')
    dc['terminal_xg'] = dc['terminal_xg'].fillna(0.0)

    di = dribbles_incomplete.copy()
    di = di.merge(term_xg, on=['match_id', 'possession'], how='left')
    di['terminal_xg'] = di['terminal_xg'].fillna(0.0)

    all_actions.append({
        'league': league['name'],
        'passes_complete': pc,
        'carries': ca,
        'dribbles_complete': dc,
        'dribbles_incomplete': di,
        'xt_grid': xt_grid,
        'mean_xt': mean_xt,
    })

    n_poss = df.groupby(['match_id', 'possession']).ngroups
    n_shot_poss = len(term_xg)
    print(f"  {n_poss:,} possessions ({n_shot_poss} with terminal shot)")

# ==============================================================================
# POOL ALL LEAGUES
# ==============================================================================

section("POOLING ALL LEAGUES")

pc_all = pd.concat([a['passes_complete'] for a in all_actions], ignore_index=True)
ca_all = pd.concat([a['carries'] for a in all_actions], ignore_index=True)
dc_all = pd.concat([a['dribbles_complete'] for a in all_actions], ignore_index=True)
di_all = pd.concat([a['dribbles_incomplete'] for a in all_actions], ignore_index=True)

print(f"  Complete passes:     {len(pc_all):>8,}")
print(f"  Carries:             {len(ca_all):>8,}")
print(f"  Complete dribbles:   {len(dc_all):>8,}")
print(f"  Incomplete dribbles: {len(di_all):>8,}")

# ==============================================================================
# EMPIRICAL FEATURE CONSEQUENCE ESTIMATES
# ==============================================================================

section("EMPIRICAL FEATURE CONSEQUENCE ESTIMATES")

print(f"\n  Method: E[terminal_xG | feature, cell] − E[terminal_xG | no feature, cell]")
print(f"  Matched on end cell (controls for spatial position / ΔxT)")
print(f"  Min observations per cell group: {MIN_OBS}\n")

results = {}

# 1. Progressive pass
feat_group = pc_all[pc_all['is_progressive']].copy()
ctrl_group = pc_all[~pc_all['is_progressive']].copy()
bonus, n_cells, n_feat, _ = matched_comparison(
    feat_group, ctrl_group, 'end_cell', 'terminal_xg')
results['prog_pass'] = {'bonus': bonus, 'n_cells': n_cells, 'n_feat': n_feat}
print(f"  Progressive pass:  bonus = {bonus:+.5f}  (matched on {n_cells} cells, "
      f"N_feat = {n_feat:,}, N_ctrl = {len(ctrl_group):,})")

# 2. Progressive carry
feat_group = ca_all[ca_all['is_progressive']].copy()
ctrl_group = ca_all[~ca_all['is_progressive']].copy()
bonus, n_cells, n_feat, _ = matched_comparison(
    feat_group, ctrl_group, 'end_cell', 'terminal_xg')
results['prog_carry'] = {'bonus': bonus, 'n_cells': n_cells, 'n_feat': n_feat}
print(f"  Progressive carry: bonus = {bonus:+.5f}  (matched on {n_cells} cells, "
      f"N_feat = {n_feat:,}, N_ctrl = {len(ctrl_group):,})")

# 3. Line-breaking
passes_lb = pc_all[['end_cell', 'is_line_break', 'terminal_xg']].copy()
carries_lb = ca_all[['end_cell', 'is_line_break', 'terminal_xg']].copy()
lb_all = pd.concat([passes_lb, carries_lb], ignore_index=True)
feat_group = lb_all[lb_all['is_line_break']].copy()
ctrl_group = lb_all[~lb_all['is_line_break']].copy()
bonus, n_cells, n_feat, _ = matched_comparison(
    feat_group, ctrl_group, 'end_cell', 'terminal_xg')
results['line_break'] = {'bonus': bonus, 'n_cells': n_cells, 'n_feat': n_feat}
print(f"  Line-breaking:     bonus = {bonus:+.5f}  (matched on {n_cells} cells, "
      f"N_feat = {n_feat:,}, N_ctrl = {len(ctrl_group):,})")

# 4. Penalty box entry
passes_pb = pc_all[['end_cell', 'is_pen_box', 'terminal_xg']].copy()
carries_pb = ca_all[['end_cell', 'is_pen_box', 'terminal_xg']].copy()
pb_all = pd.concat([passes_pb, carries_pb], ignore_index=True)
feat_group = pb_all[pb_all['is_pen_box']].copy()
ctrl_group = pb_all[~pb_all['is_pen_box']].copy()
bonus, n_cells, n_feat, _ = matched_comparison(
    feat_group, ctrl_group, 'end_cell', 'terminal_xg')
results['pen_box'] = {'bonus': bonus, 'n_cells': n_cells, 'n_feat': n_feat}
print(f"  Pen box entry:     bonus = {bonus:+.5f}  (matched on {n_cells} cells, "
      f"N_feat = {n_feat:,}, N_ctrl = {len(ctrl_group):,})")

# 5. Shot assist
feat_group = pc_all[pc_all['is_shot_assist']].copy()
ctrl_group = pc_all[~pc_all['is_shot_assist']].copy()
bonus, n_cells, n_feat, _ = matched_comparison(
    feat_group, ctrl_group, 'end_cell', 'terminal_xg')
results['shot_assist'] = {'bonus': bonus, 'n_cells': n_cells, 'n_feat': n_feat}
print(f"  Shot assist:       bonus = {bonus:+.5f}  (matched on {n_cells} cells, "
      f"N_feat = {n_feat:,}, N_ctrl = {len(ctrl_group):,})")

# 6. Under pressure (carries)
feat_group = ca_all[ca_all['is_under_pressure']].copy()
ctrl_group = ca_all[~ca_all['is_under_pressure']].copy()
bonus, n_cells, n_feat, _ = matched_comparison(
    feat_group, ctrl_group, 'end_cell', 'terminal_xg')
results['pressure'] = {'bonus': bonus, 'n_cells': n_cells, 'n_feat': n_feat}
print(f"  Under pressure:    bonus = {bonus:+.5f}  (matched on {n_cells} cells, "
      f"N_feat = {n_feat:,}, N_ctrl = {len(ctrl_group):,})")

# 7. Complete dribble (matched on start cell)
bonus, n_cells, n_feat, _ = matched_comparison(
    dc_all, di_all, 'cell_idx', 'terminal_xg')
results['dribble'] = {'bonus': bonus, 'n_cells': n_cells, 'n_feat': n_feat}
print(f"  Complete dribble:  bonus = {bonus:+.5f}  (matched on {n_cells} cells, "
      f"N_feat = {n_feat:,}, N_ctrl = {len(di_all):,})")

# ==============================================================================
# COMPARISON
# ==============================================================================

section("COMPARISON: EMPIRICAL vs HAND-TUNED WEIGHTS")

print(f"\n  {'Feature':<20} {'Empirical':>12} {'Hand-tuned':>12} {'Ratio':>8}")
print(f"  {'-' * 56}")
for feat_name in HAND_TUNED:
    emp = results[feat_name]['bonus']
    ht = HAND_TUNED[feat_name]
    ratio = emp / ht if ht != 0 else float('inf')
    print(f"  {feat_name:<20} {emp:>12.5f} {ht:>12.3f} {ratio:>8.2f}x")

# ==============================================================================
# PER-LEAGUE BREAKDOWN
# ==============================================================================

section("PER-LEAGUE FEATURE CONSEQUENCES")

for i, league in enumerate(LEAGUES):
    a = all_actions[i]
    pc_l = a['passes_complete']
    ca_l = a['carries']
    dc_l = a['dribbles_complete']
    di_l = a['dribbles_incomplete']

    print(f"\n  {league['name']}")
    print(f"  {'Feature':<20} {'Bonus':>10} {'Cells':>6} {'N_feat':>8}")
    print(f"  {'-' * 48}")

    # Progressive pass
    b, nc, nf, _ = matched_comparison(
        pc_l[pc_l['is_progressive']], pc_l[~pc_l['is_progressive']],
        'end_cell', 'terminal_xg')
    print(f"  {'prog_pass':<20} {b:>10.5f} {nc:>6} {nf:>8,}")

    # Progressive carry
    b, nc, nf, _ = matched_comparison(
        ca_l[ca_l['is_progressive']], ca_l[~ca_l['is_progressive']],
        'end_cell', 'terminal_xg')
    print(f"  {'prog_carry':<20} {b:>10.5f} {nc:>6} {nf:>8,}")

    # Line-breaking
    lb_l = pd.concat([
        pc_l[['end_cell', 'is_line_break', 'terminal_xg']],
        ca_l[['end_cell', 'is_line_break', 'terminal_xg']],
    ], ignore_index=True)
    b, nc, nf, _ = matched_comparison(
        lb_l[lb_l['is_line_break']], lb_l[~lb_l['is_line_break']],
        'end_cell', 'terminal_xg')
    print(f"  {'line_break':<20} {b:>10.5f} {nc:>6} {nf:>8,}")

    # Pen box entry
    pb_l = pd.concat([
        pc_l[['end_cell', 'is_pen_box', 'terminal_xg']],
        ca_l[['end_cell', 'is_pen_box', 'terminal_xg']],
    ], ignore_index=True)
    b, nc, nf, _ = matched_comparison(
        pb_l[pb_l['is_pen_box']], pb_l[~pb_l['is_pen_box']],
        'end_cell', 'terminal_xg')
    print(f"  {'pen_box':<20} {b:>10.5f} {nc:>6} {nf:>8,}")

    # Shot assist
    b, nc, nf, _ = matched_comparison(
        pc_l[pc_l['is_shot_assist']], pc_l[~pc_l['is_shot_assist']],
        'end_cell', 'terminal_xg')
    print(f"  {'shot_assist':<20} {b:>10.5f} {nc:>6} {nf:>8,}")

    # Pressure
    b, nc, nf, _ = matched_comparison(
        ca_l[ca_l['is_under_pressure']], ca_l[~ca_l['is_under_pressure']],
        'end_cell', 'terminal_xg')
    print(f"  {'pressure':<20} {b:>10.5f} {nc:>6} {nf:>8,}")

    # Dribble
    b, nc, nf, _ = matched_comparison(dc_l, di_l, 'cell_idx', 'terminal_xg')
    print(f"  {'dribble':<20} {b:>10.5f} {nc:>6} {nf:>8,}")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

section("SAVING RESULTS")

rows = []
for feat_name in HAND_TUNED:
    rows.append({
        'feature': feat_name,
        'empirical_bonus': results[feat_name]['bonus'],
        'hand_tuned_weight': HAND_TUNED[feat_name],
        'n_cells_matched': results[feat_name]['n_cells'],
        'n_feature_actions': results[feat_name]['n_feat'],
    })
results_df = pd.DataFrame(rows)
out_path = os.path.join(results_dir, 'pva_weight_calibration.csv')
results_df.to_csv(out_path, index=False)
print(f"  Saved: {out_path}")

# ==============================================================================
# EMPIRICAL CONSTANTS
# ==============================================================================

section("EMPIRICAL CONSTANTS")

# Floor negative bonuses at 0 (already captured by base ΔxT)
for feat_name in HAND_TUNED:
    val = results[feat_name]['bonus']
    floored = max(0.0, val)
    flag = "" if val >= 0 else "  (floored from {:.5f})".format(val)
    print(f"  {feat_name:<20} = {floored:.6f}{flag}")

print(f"\n  Negative values floored at 0 (already captured by base ΔxT).")

prog_pass = max(0.0, results['prog_pass']['bonus'])
prog_carry = max(0.0, results['prog_carry']['bonus'])
line_break = max(0.0, results['line_break']['bonus'])
pen_box = max(0.0, results['pen_box']['bonus'])
shot_assist = max(0.0, results['shot_assist']['bonus'])
pressure_add = max(0.0, results['pressure']['bonus'])
dribble = max(0.0, results['dribble']['bonus'])

print(f"\n  # Paste into pva_model_generic.py:")
print(f"  PROGRESSIVE_PASS_BONUS_PER_10_YARDS = {prog_pass:.6f}")
print(f"  LINE_BREAKING_BONUS = {line_break:.6f}")
print(f"  CARRY_PROGRESSIVE_BONUS_PER_10_YARDS = {prog_carry:.6f}")
print(f"  PRESSURE_MULTIPLIER = {1.0 + pressure_add:.6f}")
print(f"  PENALTY_BOX_ENTRY_BONUS = {pen_box:.6f}")
print(f"  # shot_assist_bonus = {shot_assist:.6f}  (line 469)")
print(f"  # dribble_bonus = {dribble:.6f}  (line 452)")

print(f"\n{'=' * 80}")
print(f"  CALIBRATION COMPLETE")
print(f"{'=' * 80}")
