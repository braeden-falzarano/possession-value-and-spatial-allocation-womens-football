"""
PVA model for any StatsBomb competition/season. Outputs actions_with_pva.csv.
  python pva_model_generic.py --comp_name "FA WSL 2018-19" --comp_id 37 --season_id 4
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from statsbombpy import sb
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CLI ARGUMENTS
# ==============================================================================

parser = argparse.ArgumentParser(description='Run PVA model on StatsBomb data')
parser.add_argument('--comp_name', required=True, help='Human-readable name (e.g. "FA WSL 2018-19")')
parser.add_argument('--comp_id', required=True, type=int, help='StatsBomb competition_id')
parser.add_argument('--season_id', required=True, type=int, help='StatsBomb season_id')
args = parser.parse_args()

COMP_NAME = args.comp_name
COMP_ID   = args.comp_id
SEASON_ID = args.season_id

SLUG = COMP_NAME.lower().replace(' ', '_').replace('/', '-').replace('--', '-')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PITCH_LENGTH = 120
PITCH_WIDTH = 80
X_BINS = 9
Y_BINS = 6
N_CELLS = X_BINS * Y_BINS

DEFENSIVE_THIRD_END = 40
MIDFIELD_THIRD_END = 80

# Empirically calibrated weights (see pva_weight_calibration.py)
PROGRESSIVE_PASS_BONUS_PER_10_YARDS = 0.000247
LINE_BREAKING_BONUS = 0.000771
CARRY_PROGRESSIVE_BONUS_PER_10_YARDS = 0.001420
PRESSURE_MULTIPLIER = 1.0       # floored from negative
PENALTY_BOX_ENTRY_BONUS = 0.008225

print("=" * 80)
print(f"PVA MODEL — {COMP_NAME}")
print(f"  competition_id={COMP_ID}  season_id={SEASON_ID}")
print("=" * 80)
print(f"\nGrid: {X_BINS}×{Y_BINS} = {N_CELLS} cells")
print(f"Bonuses: prog_pass={PROGRESSIVE_PASS_BONUS_PER_10_YARDS}, "
      f"line_break={LINE_BREAKING_BONUS}, carry_prog={CARRY_PROGRESSIVE_BONUS_PER_10_YARDS}, "
      f"pressure={PRESSURE_MULTIPLIER}x, pen_box={PENALTY_BOX_ENTRY_BONUS}")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def assign_grid_cell(x, y):
    cell_x = min(int(x / (PITCH_LENGTH / X_BINS)), X_BINS - 1)
    cell_y = min(int(y / (PITCH_WIDTH / Y_BINS)), Y_BINS - 1)
    return cell_x, cell_y

def get_cell_idx(location):
    if location is None or not isinstance(location, (list, tuple)) or len(location) < 2:
        return None
    cell_x, cell_y = assign_grid_cell(location[0], location[1])
    return cell_x * Y_BINS + cell_y

def get_cell_center(cell_idx):
    cell_x = cell_idx // Y_BINS
    cell_y = cell_idx % Y_BINS
    center_x = cell_x * (PITCH_LENGTH / X_BINS) + (PITCH_LENGTH / X_BINS / 2)
    center_y = cell_y * (PITCH_WIDTH / Y_BINS) + (PITCH_WIDTH / Y_BINS / 2)
    return (center_x, center_y)

def get_pitch_third(x):
    if x < DEFENSIVE_THIRD_END:
        return 'defensive'
    elif x < MIDFIELD_THIRD_END:
        return 'middle'
    else:
        return 'attacking'

def is_turnover(event):
    if event['type'] in ['Interception', 'Tackle', 'Clearance', 'Dispossessed', 'Dribbled Past']:
        return True
    if event['type'] == 'Pass':
        outcome = event.get('pass_outcome')
        if pd.notna(outcome):
            return True
    return False

def calculate_distance(loc1, loc2):
    if loc1 is None or loc2 is None:
        return 0
    if not isinstance(loc1, (list, tuple)) or not isinstance(loc2, (list, tuple)):
        return 0
    if len(loc1) < 2 or len(loc2) < 2:
        return 0
    return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def is_in_penalty_box(location):
    if location is None or not isinstance(location, (list, tuple)) or len(location) < 2:
        return False
    return location[0] >= 102 and 18 <= location[1] <= 62

def enters_penalty_box(start_loc, end_loc):
    if start_loc is None or end_loc is None:
        return False
    return not is_in_penalty_box(start_loc) and is_in_penalty_box(end_loc)

def crosses_third_boundary(start_loc, end_loc):
    if start_loc is None or end_loc is None:
        return False
    start_third = get_pitch_third(start_loc[0])
    end_third = get_pitch_third(end_loc[0])
    return (start_third == 'defensive' and end_third in ['middle', 'attacking']) or \
           (start_third == 'middle' and end_third == 'attacking')

# ==============================================================================
# PHASE 1: DATA LOADING
# ==============================================================================

print("\n" + "-" * 80)
print(f"PHASE 1: LOADING {COMP_NAME} DATA")
print("-" * 80)

matches = sb.matches(competition_id=COMP_ID, season_id=SEASON_ID)
match_ids = matches['match_id'].tolist()
print(f"  {len(match_ids)} matches found")

all_events = []
for i, match_id in enumerate(match_ids):
    if (i + 1) % 20 == 0 or i == 0 or i == len(match_ids) - 1:
        print(f"  Loading match {i+1}/{len(match_ids)}...")
    try:
        events = sb.events(match_id=match_id)
        events['match_id'] = match_id
        all_events.append(events)
    except Exception as e:
        print(f"  WARNING: Error loading match {match_id}: {e}")

events_df = pd.concat(all_events, ignore_index=True)
print(f"\n  {len(events_df):,} total events")
print(f"  Teams: {events_df['team'].nunique()}")
print(f"  Players: {events_df['player'].nunique()}")

# ==============================================================================
# PHASE 2: EXPECTED THREAT (xT) MODEL
# ==============================================================================

print("\n" + "-" * 80)
print("PHASE 2: BUILDING xT MODEL")
print("-" * 80)

transition_matrix = np.zeros((N_CELLS, N_CELLS))
shots_from_cell = np.zeros(N_CELLS)
turnovers_from_cell = np.zeros(N_CELLS)
cell_action_counts = np.zeros(N_CELLS)

for match_id in events_df['match_id'].unique():
    match_events = events_df[events_df['match_id'] == match_id].copy()
    match_events = match_events.sort_values('index').reset_index(drop=True)

    current_team = None
    possession_start = 0

    for i, (idx, event) in enumerate(match_events.iterrows()):
        team = event.get('possession_team')
        if team != current_team:
            if current_team is not None and i > possession_start:
                possession = match_events.iloc[possession_start:i]
                for j in range(len(possession) - 1):
                    curr = possession.iloc[j]
                    next_event = possession.iloc[j + 1]
                    curr_loc = curr.get('location')
                    if curr_loc is None:
                        continue
                    start_idx = get_cell_idx(curr_loc)
                    if start_idx is None:
                        continue
                    cell_action_counts[start_idx] += 1
                    if next_event['type'] == 'Shot':
                        shots_from_cell[start_idx] += 1
                    elif is_turnover(next_event):
                        turnovers_from_cell[start_idx] += 1
                    else:
                        next_loc = next_event.get('location')
                        if next_loc is not None:
                            end_idx = get_cell_idx(next_loc)
                            if end_idx is not None:
                                transition_matrix[start_idx, end_idx] += 1
            current_team = team
            possession_start = i

print(f"  Processed {cell_action_counts.sum():,.0f} actions in possessions")

shot_prob = np.zeros(N_CELLS)
turnover_prob = np.zeros(N_CELLS)
for i in range(N_CELLS):
    if cell_action_counts[i] > 0:
        transition_matrix[i, :] /= cell_action_counts[i]
        shot_prob[i] = shots_from_cell[i] / cell_action_counts[i]
        turnover_prob[i] = turnovers_from_cell[i] / cell_action_counts[i]

shots_df = events_df[events_df['type'] == 'Shot'].copy()
cell_xg_sum = np.zeros(N_CELLS)
cell_xg_count = np.zeros(N_CELLS)
for idx, shot in shots_df.iterrows():
    xg = shot.get('shot_statsbomb_xg')
    if pd.notna(xg):
        cell_idx = get_cell_idx(shot.get('location'))
        if cell_idx is not None:
            cell_xg_sum[cell_idx] += xg
            cell_xg_count[cell_idx] += 1

league_avg_xg = np.sum(cell_xg_sum) / max(np.sum(cell_xg_count), 1)
shot_xg_by_cell = np.where(cell_xg_count > 0, cell_xg_sum / cell_xg_count, league_avg_xg)
print(f"  {len(shots_df)} shots, league avg xG: {league_avg_xg:.4f}")

xt = np.zeros(N_CELLS)
for iteration in range(10):
    xt_new = np.zeros(N_CELLS)
    for i in range(N_CELLS):
        shoot_value = shot_prob[i] * shot_xg_by_cell[i]
        move_value = np.sum(transition_matrix[i, :] * xt)
        xt_new[i] = shoot_value + move_value
    if np.allclose(xt, xt_new, atol=1e-5):
        print(f"  xT converged after {iteration + 1} iterations")
        break
    xt = xt_new

print(f"  xT range: [{xt.min():.4f}, {xt.max():.4f}], mean: {xt.mean():.4f}")

# ==============================================================================
# PHASE 2.5: EMPIRICAL TURNOVER CONSEQUENCE MODEL
# ==============================================================================

print("\n" + "-" * 80)
print("PHASE 2.5: EMPIRICAL TURNOVER CONSEQUENCE MODEL")
print("-" * 80)

turnover_threat_sum = np.zeros(N_CELLS)
turnover_count = np.zeros(N_CELLS)
turnover_shots = np.zeros(N_CELLS)
turnover_xg_sum = np.zeros(N_CELLS)
MAX_ACTIONS_TO_TRACK = 10

for match_id in events_df['match_id'].unique():
    match_events = events_df[events_df['match_id'] == match_id].copy()
    match_events = match_events.sort_values('index').reset_index(drop=True)

    for i in range(len(match_events) - 1):
        event = match_events.iloc[i]
        is_turnover_event = False
        turnover_loc = None

        if event['type'] == 'Pass' and pd.notna(event.get('pass_outcome')):
            is_turnover_event = True
            turnover_loc = event.get('location')
        elif event['type'] in ['Dispossessed', 'Dribbled Past', 'Miscontrol']:
            is_turnover_event = True
            turnover_loc = event.get('location')

        if not is_turnover_event or turnover_loc is None:
            continue

        turnover_cell = get_cell_idx(turnover_loc)
        if turnover_cell is None:
            continue

        losing_team = event.get('team')
        opponent_threat = 0.0
        opponent_shot = False
        opponent_xg = 0.0

        for j in range(i + 1, min(i + 1 + MAX_ACTIONS_TO_TRACK, len(match_events))):
            next_event = match_events.iloc[j]
            if next_event.get('team') == losing_team:
                break
            next_loc = next_event.get('location')
            if next_loc is None:
                continue
            next_cell = get_cell_idx(next_loc)
            if next_cell is not None:
                opponent_threat += xt[next_cell]
            if next_event['type'] == 'Shot':
                opponent_shot = True
                shot_xg = next_event.get('shot_statsbomb_xg', 0)
                if pd.notna(shot_xg):
                    opponent_xg += shot_xg

        turnover_threat_sum[turnover_cell] += opponent_threat
        turnover_count[turnover_cell] += 1
        if opponent_shot:
            turnover_shots[turnover_cell] += 1
            turnover_xg_sum[turnover_cell] += opponent_xg

avg_turnover_consequence = np.zeros(N_CELLS)
for i in range(N_CELLS):
    if turnover_count[i] > 0:
        avg_turnover_consequence[i] = turnover_threat_sum[i] / turnover_count[i]

for i in range(N_CELLS):
    if turnover_count[i] == 0:
        cell_x = i // Y_BINS
        cell_y = i % Y_BINS
        flipped_x = X_BINS - 1 - cell_x
        flipped_cell = flipped_x * Y_BINS + cell_y
        avg_turnover_consequence[i] = xt[flipped_cell]

total_turnovers = int(turnover_count.sum())
print(f"  {total_turnovers:,} turnovers analyzed")

# ==============================================================================
# PHASE 3: PVA CALCULATION
# ==============================================================================

print("\n" + "-" * 80)
print("PHASE 3: CALCULATING PVA")
print("-" * 80)

def calculate_progressive_bonus(start_loc, end_loc, xt_grid):
    if start_loc is None or end_loc is None:
        return 0.0
    if not isinstance(start_loc, (list, tuple)) or not isinstance(end_loc, (list, tuple)):
        return 0.0
    if len(start_loc) < 2 or len(end_loc) < 2:
        return 0.0
    forward_distance = end_loc[0] - start_loc[0]
    if forward_distance < 10:
        return 0.0
    start_cell = get_cell_idx(start_loc)
    end_cell = get_cell_idx(end_loc)
    if start_cell is None or end_cell is None:
        return (forward_distance / 10) * PROGRESSIVE_PASS_BONUS_PER_10_YARDS
    avg_xt = (xt_grid[start_cell] + xt_grid[end_cell]) / 2
    mean_xt = xt_grid.mean()
    context_multiplier = avg_xt / mean_xt
    base_bonus = (forward_distance / 10) * PROGRESSIVE_PASS_BONUS_PER_10_YARDS
    return base_bonus * context_multiplier

def calculate_line_breaking_bonus(start_loc, end_loc):
    if crosses_third_boundary(start_loc, end_loc):
        return LINE_BREAKING_BONUS
    return 0.0

def calculate_turnover_cost(turnover_location, xt_grid, empirical_consequence=None):
    if turnover_location is None:
        return 0.0
    turnover_cell = get_cell_idx(turnover_location)
    if turnover_cell is None:
        return 0.0
    if empirical_consequence is not None:
        return -empirical_consequence[turnover_cell]
    opponent_x = PITCH_LENGTH - turnover_location[0]
    opponent_y = turnover_location[1]
    opponent_cell = get_cell_idx([opponent_x, opponent_y])
    if opponent_cell is None:
        return 0.0
    return -xt_grid[opponent_cell]

def calculate_defensive_value(action, xt_grid, turnover_consequence=None):
    action_location = action.get('location')
    if action_location is None:
        return 0.0
    recovery_cell = get_cell_idx(action_location)
    if recovery_cell is None:
        return 0.0
    if turnover_consequence is not None:
        our_threat_created = turnover_consequence[recovery_cell]
    else:
        our_threat_created = xt_grid[recovery_cell]
    opponent_x = PITCH_LENGTH - action_location[0]
    opponent_y = action_location[1]
    opponent_cell = get_cell_idx([opponent_x, opponent_y])
    if opponent_cell is None:
        opponent_threat_denied = 0.0
    else:
        if turnover_consequence is not None:
            opponent_threat_denied = turnover_consequence[opponent_cell]
        else:
            opponent_threat_denied = xt_grid[opponent_cell]
    return opponent_threat_denied + our_threat_created

def calculate_pva(action, xt_grid, turnover_consequence=None):
    action_type = action['type']

    if action_type in ['Interception', 'Tackle', 'Block', 'Ball Recovery']:
        return calculate_defensive_value(action, xt_grid, turnover_consequence)

    if action_type == 'Carry':
        start_loc = action.get('location')
        end_loc = action.get('carry_end_location')
        if start_loc is None or end_loc is None:
            return 0.0
        start_cell = get_cell_idx(start_loc)
        end_cell = get_cell_idx(end_loc)
        if start_cell is None or end_cell is None:
            return 0.0
        base_value = xt_grid[end_cell] - xt_grid[start_cell]
        forward_distance = end_loc[0] - start_loc[0]
        if forward_distance >= 10:
            avg_xt = (xt_grid[start_cell] + xt_grid[end_cell]) / 2
            mean_xt = xt_grid.mean()
            context_multiplier = avg_xt / mean_xt
            distance_bonus = (forward_distance / 10) * CARRY_PROGRESSIVE_BONUS_PER_10_YARDS * context_multiplier
        else:
            distance_bonus = 0.0
        under_pressure = action.get('under_pressure', False)
        if under_pressure and (base_value + distance_bonus) > 0:
            pressure_mult = PRESSURE_MULTIPLIER
        else:
            pressure_mult = 1.0
        pen_box_bonus = PENALTY_BOX_ENTRY_BONUS if enters_penalty_box(start_loc, end_loc) else 0.0
        line_break_bonus = calculate_line_breaking_bonus(start_loc, end_loc)
        total_value = ((base_value + distance_bonus) * pressure_mult) + pen_box_bonus + line_break_bonus
        return total_value

    if action_type == 'Dribble':
        outcome = action.get('dribble_outcome')
        location = action.get('location')
        if location is None:
            return 0.0
        cell = get_cell_idx(location)
        if cell is None:
            return 0.0
        if outcome == 'Complete':
            cell_xt = xt_grid[cell]
            mean_xt = xt_grid.mean()
            context_multiplier = cell_xt / mean_xt
            return 0.017781 * context_multiplier
        return 0.0

    if action_type == 'Pass':
        start_loc = action.get('location')
        end_loc = action.get('pass_end_location')
        outcome = action.get('pass_outcome')
        if pd.notna(outcome):
            return calculate_turnover_cost(start_loc, xt_grid, turnover_consequence)
        if start_loc and end_loc:
            start_cell = get_cell_idx(start_loc)
            end_cell = get_cell_idx(end_loc)
            if start_cell is not None and end_cell is not None:
                base_value = xt_grid[end_cell] - xt_grid[start_cell]
                progressive_bonus = calculate_progressive_bonus(start_loc, end_loc, xt_grid)
                line_breaking_bonus = calculate_line_breaking_bonus(start_loc, end_loc)
                shot_assist = action.get('pass_shot_assist')
                shot_assist_bonus = 0.040554 if pd.notna(shot_assist) else 0.0
                return base_value + progressive_bonus + line_breaking_bonus + shot_assist_bonus
        return 0.0

    if action_type == 'Shot':
        xg = action.get('shot_statsbomb_xg', 0)
        location = action.get('location')
        if location:
            cell = get_cell_idx(location)
            if cell is not None:
                return xg - xt_grid[cell]
        return xg

    if action_type in ['Dispossessed', 'Miscontrol']:
        location = action.get('location')
        return calculate_turnover_cost(location, xt_grid, turnover_consequence)

    if action_type == 'Dribbled Past':
        location = action.get('location')
        if location is None:
            return 0.0
        recovery_cell = get_cell_idx(location)
        if recovery_cell is None:
            return 0.0
        if turnover_consequence is not None:
            return -turnover_consequence[recovery_cell]
        else:
            return -xt_grid[recovery_cell]

    return 0.0

action_types = ['Pass', 'Carry', 'Dribble', 'Shot',
                'Interception', 'Tackle', 'Block', 'Ball Recovery',
                'Dispossessed', 'Miscontrol', 'Dribbled Past']
actions_df = events_df[events_df['type'].isin(action_types)].copy()

start_of_play_types = ['Kick Off', 'Goal Kick', 'Throw-in', 'Corner', 'Free Kick']
if 'pass_type' in actions_df.columns:
    before_count = len(actions_df)
    actions_df = actions_df[~actions_df['pass_type'].isin(start_of_play_types)]
    print(f"  Removed {before_count - len(actions_df):,} start-of-play events")

print(f"  Calculating PVA for {len(actions_df):,} actions...")

pva_values = []
total = len(actions_df)
for count, (idx, action) in enumerate(actions_df.iterrows()):
    if count % 20000 == 0 and count > 0:
        print(f"    {count:,}/{total:,} ({count/total*100:.0f}%)...")
    pva_values.append(calculate_pva(action, xt, avg_turnover_consequence))

actions_df['pva'] = pva_values
actions_df['cell_idx'] = actions_df['location'].apply(get_cell_idx)
actions_df['x'] = actions_df['location'].apply(lambda loc: loc[0] if loc and len(loc) >= 2 else None)
actions_df['y'] = actions_df['location'].apply(lambda loc: loc[1] if loc and len(loc) >= 2 else None)
actions_df['third'] = actions_df['x'].apply(lambda x: get_pitch_third(x) if pd.notna(x) else None)
actions_df = actions_df.dropna(subset=['x', 'y', 'cell_idx'])

print(f"\n  {len(actions_df):,} actions with PVA")
print(f"  PVA range: [{actions_df['pva'].min():.4f}, {actions_df['pva'].max():.4f}]")
print(f"  Mean PVA: {actions_df['pva'].mean():.4f}")
print(f"  Total PVA: {actions_df['pva'].sum():.2f}")

# ==============================================================================
# PHASE 4: VALIDATION
# ==============================================================================

print("\n" + "-" * 80)
print("PHASE 4: VALIDATION")
print("-" * 80)

print("\nValue Distribution by Action Type:")
for at in action_types:
    type_df = actions_df[actions_df['type'] == at]
    if len(type_df) > 0:
        print(f"  {at:15s}: n={len(type_df):6,} | mean={type_df['pva'].mean():7.4f} | total={type_df['pva'].sum():8.2f}")

print("\nValue Distribution by Pitch Third:")
for third in ['defensive', 'middle', 'attacking']:
    third_df = actions_df[actions_df['third'] == third]
    if len(third_df) > 0:
        print(f"  {third:15s}: n={len(third_df):6,} | mean={third_df['pva'].mean():7.4f} | total={third_df['pva'].sum():8.2f}")

# ==============================================================================
# PHASE 5: SAVE
# ==============================================================================

print("\n" + "-" * 80)
print("PHASE 5: SAVING RESULTS")
print("-" * 80)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
processed_dir = os.path.join(project_root, 'data', 'processed')
os.makedirs(processed_dir, exist_ok=True)

output_file = os.path.join(processed_dir, f'{SLUG}_actions_with_pva.csv')
actions_df.to_csv(output_file, index=False)
print(f"  Saved: {output_file}")
print(f"  {len(actions_df):,} actions with PVA values")

xt_df = pd.DataFrame({'cell_idx': range(N_CELLS), 'xt_value': xt})
xt_output = os.path.join(processed_dir, f'{SLUG}_xt_grid.csv')
xt_df.to_csv(xt_output, index=False)
print(f"  Saved: {xt_output}")

print("\n" + "=" * 80)
print(f"PVA MODEL COMPLETE — {COMP_NAME}")
print(f"  {len(actions_df):,} actions  |  {len(match_ids)} matches  |  {events_df['team'].nunique()} teams")
print("=" * 80)
