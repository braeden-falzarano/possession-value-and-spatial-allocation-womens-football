"""
Phase 1: extract possession paths, resample to fixed-length waypoints.
Run: python 01_extract_possessions.py
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PITCH_LENGTH = 120
PITCH_WIDTH  = 80
N_RESAMPLE   = 20   # 40D feature vector
MIN_ACTIONS  = 7

# ==============================================================================
# PATHS
# ==============================================================================

script_dir    = os.path.dirname(os.path.abspath(__file__))
project_root  = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
processed_dir = os.path.join(project_root, 'data', 'processed')
stats_dir     = os.path.join(project_root, 'results')
output_dir    = os.path.join(stats_dir, 'possession_patterns')
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# LEAGUE-SEASON DEFINITIONS
# ==============================================================================

LEAGUES = [
    {
        'name': 'NWSL 2018',
        'actions_csv': os.path.join(processed_dir, 'nwsl_2018_actions_with_pva_clean.csv'),
    },
    {
        'name': 'FA WSL 2018-19',
        'actions_csv': os.path.join(processed_dir, 'fa_wsl_2018-19_actions_with_pva.csv'),
    },
    {
        'name': 'FA WSL 2019-20',
        'actions_csv': os.path.join(processed_dir, 'fa_wsl_2019-20_actions_with_pva.csv'),
    },
    {
        'name': 'FA WSL 2020-21',
        'actions_csv': os.path.join(processed_dir, 'fa_wsl_2020-21_actions_with_pva.csv'),
    },
]

# ==============================================================================
# HELPERS
# ==============================================================================

def get_pitch_third(x):
    if x < 40:  return 'Defensive'
    if x < 80:  return 'Midfield'
    return 'Attacking'

def resample_path(xy_coords, n_points=10):
    """Resample path to n_points via arc-length interpolation. Returns normalized flat array."""
    xy = np.array(xy_coords, dtype=np.float64)

    xy[:, 0] /= PITCH_LENGTH
    xy[:, 1] /= PITCH_WIDTH

    if len(xy) == 1:
        return np.tile(xy[0], n_points)

    diffs = np.diff(xy, axis=0)
    cum_length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])

    if cum_length[-1] < 1e-10:
        return np.tile(xy[0], n_points)

    targets = np.linspace(0.0, cum_length[-1], n_points)
    fx = interp1d(cum_length, xy[:, 0])
    fy = interp1d(cum_length, xy[:, 1])
    return np.column_stack([fx(targets), fy(targets)]).flatten()

def section(title):
    print()
    print('=' * 80)
    print(f'  {title}')
    print('=' * 80)

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

all_possession_rows = []

for league in LEAGUES:
    section(f"Processing {league['name']}")

    if not os.path.exists(league['actions_csv']):
        print(f"  SKIP: {league['actions_csv']} not found")
        continue

    actions_df = pd.read_csv(league['actions_csv'])
    print(f"  Loaded {len(actions_df):,} actions")

    required = ['match_id', 'possession', 'possession_team', 'team', 'x', 'y', 'pva', 'index']
    missing = [c for c in required if c not in actions_df.columns]
    if missing:
        print(f"  SKIP: missing columns {missing}")
        continue

    actions_df = actions_df.dropna(subset=['x', 'y', 'pva'])

    actions_df = actions_df.sort_values(['match_id', 'possession', 'index'])

    n_before = 0
    grouped = actions_df.groupby(['match_id', 'possession'])

    for (mid, poss_id), poss_df in grouped:
        poss_team = poss_df['possession_team'].iloc[0]

        team_actions = poss_df[poss_df['team'] == poss_team]

        if len(team_actions) == 0:
            continue

        team_actions = team_actions.dropna(subset=['x', 'y'])
        if len(team_actions) < MIN_ACTIONS:
            continue

        xy_seq = team_actions[['x', 'y']].values.tolist()
        n_actions = len(xy_seq)

        total_pva = team_actions['pva'].sum()
        mean_pva  = team_actions['pva'].mean()

        start_third = get_pitch_third(xy_seq[0][0])
        end_third   = get_pitch_third(xy_seq[-1][0])

        resampled = resample_path(xy_seq, N_RESAMPLE)

        row = {
            'league':          league['name'],
            'match_id':        mid,
            'possession':      poss_id,
            'possession_team': poss_team,
            'n_actions':       n_actions,
            'total_pva':       round(total_pva, 6),
            'mean_pva':        round(mean_pva, 6),
            'start_third':     start_third,
            'end_third':       end_third,
        }
        for i in range(N_RESAMPLE):
            row[f'p{i}_x'] = round(resampled[2 * i], 6)
            row[f'p{i}_y'] = round(resampled[2 * i + 1], 6)

        all_possession_rows.append(row)

    league_count = sum(1 for r in all_possession_rows if r['league'] == league['name'])
    print(f"  Extracted {league_count:,} possessions")

# ==============================================================================
# SAVE
# ==============================================================================

section("SAVING OUTPUT")

possessions_df = pd.DataFrame(all_possession_rows)
output_path = os.path.join(output_dir, 'possessions_resampled.csv')
possessions_df.to_csv(output_path, index=False)

print(f"  {len(possessions_df):,} total possessions across {possessions_df['league'].nunique()} leagues")
print(f"  Possession length stats: median={possessions_df['n_actions'].median():.0f}, "
      f"mean={possessions_df['n_actions'].mean():.1f}, "
      f"min={possessions_df['n_actions'].min()}, max={possessions_df['n_actions'].max()}")
print(f"  Saved: {output_path}")

for ln in possessions_df['league'].unique():
    sub = possessions_df[possessions_df['league'] == ln]
    print(f"    {ln}: {len(sub):,} possessions, "
          f"{sub['possession_team'].nunique()} teams")

print("\n" + "=" * 80)
print("  PHASE 1 COMPLETE — possessions extracted and resampled")
print("=" * 80)
