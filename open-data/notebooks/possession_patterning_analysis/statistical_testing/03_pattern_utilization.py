"""
Phase 3: compute Routing Value Added (RVA) per team from cluster usage diffs.
Run: python 03_pattern_utilization.py
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MIN_POSS = 10

STRATA = {
    'Defensive': 'D',
    'Midfield':  'M',
    'Attacking': 'A',
}

ANALYSIS_LEAGUES = {'FA WSL 2018-19', 'FA WSL 2019-20', 'FA WSL 2020-21'}

COMPLETE_SEASONS = {
    'FA WSL 2018-19': ('37', '4'),
    'FA WSL 2020-21': ('37', '90'),
}

# ==============================================================================
# PATHS
# ==============================================================================

script_dir    = os.path.dirname(os.path.abspath(__file__))
project_root  = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
data_dir      = os.path.join(project_root, 'data')
output_dir    = os.path.join(project_root, 'results', 'possession_patterns')
os.makedirs(output_dir, exist_ok=True)

input_path = os.path.join(output_dir, 'possessions_clustered.csv')

# ==============================================================================
# HELPERS
# ==============================================================================

def section(title):
    print()
    print('=' * 80)
    print(f'  {title}')
    print('=' * 80)

def load_outcomes(matches_json_path):
    with open(matches_json_path) as f:
        matches = json.load(f)
    record = {}
    for m in matches:
        for team_name, scored, conceded in [
            (m['home_team']['home_team_name'], m['home_score'], m['away_score']),
            (m['away_team']['away_team_name'], m['away_score'], m['home_score']),
        ]:
            r = record.setdefault(team_name, dict(gf=0, ga=0, gp=0))
            r['gf'] += scored; r['ga'] += conceded; r['gp'] += 1
    return pd.DataFrame([{
        'team': t,
        'goals_per_game': round(r['gf'] / r['gp'], 4),
        'goal_diff_per_game': round((r['gf'] - r['ga']) / r['gp'], 4),
    } for t, r in record.items()])

# ==============================================================================
# LOAD CLUSTERED DATA
# ==============================================================================

section("LOADING CLUSTERED POSSESSIONS")

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Run 02_cluster_patterns.py first. Missing: {input_path}")

df = pd.read_csv(input_path)
df['cluster_id'] = df['cluster_id'].astype(str)
n_total = len(df)
n_unassigned_total = (df['cluster_id'] == '-1').sum()

all_clusters = sorted([c for c in df['cluster_id'].unique() if c != '-1'])
n_clusters = len(all_clusters)
print(f"  Loaded {n_total:,} possessions, {n_clusters} clusters, "
      f"{df['league'].nunique()} leagues")
print(f"  Unassigned possessions: {n_unassigned_total:,} "
      f"({n_unassigned_total/n_total*100:.1f}%)")

# ==============================================================================
# PROCESS EACH LEAGUE
# ==============================================================================

all_team_rows = []

for league_name in sorted(ANALYSIS_LEAGUES & set(df['league'].unique())):
    section(f"Processing {league_name}")

    ldf = df[df['league'] == league_name].copy()
    teams = sorted(ldf['possession_team'].unique())
    n_teams = len(teams)
    print(f"  {len(ldf):,} possessions, {n_teams} teams")

    team_usage = {}
    team_pva   = {}
    team_n     = {}
    team_unassigned_pct = {}
    for t in teams:
        tdf = ldf[ldf['possession_team'] == t]
        total_poss = len(tdf)
        n_unassigned = (tdf['cluster_id'] == '-1').sum()
        team_unassigned_pct[t] = n_unassigned / total_poss * 100 if total_poss > 0 else 0

        tdf_assigned = tdf[tdf['cluster_id'] != '-1']
        tc = tdf_assigned.groupby('cluster_id').size()
        tp = tdf_assigned.groupby('cluster_id')['total_pva'].mean()
        team_usage[t] = {c: tc.get(c, 0) / total_poss for c in all_clusters}
        team_pva[t]   = {c: tp.get(c, np.nan) for c in all_clusters}
        team_n[t]     = {c: tc.get(c, 0) for c in all_clusters}

    team_loo_usage = {}
    team_loo_pva   = {}
    for t in teams:
        other_teams = [o for o in teams if o != t]
        loo_usage = {}
        loo_pva   = {}
        for c in all_clusters:
            other_usages = [team_usage[o][c] for o in other_teams]
            loo_usage[c] = np.mean(other_usages) if other_usages else 0
            other_pvas = [team_pva[o][c] for o in other_teams
                          if not np.isnan(team_pva[o][c])]
            loo_pva[c] = np.mean(other_pvas) if other_pvas else np.nan
        team_loo_usage[t] = loo_usage
        team_loo_pva[t]   = loo_pva

    for t in teams:
        tdf = ldf[ldf['possession_team'] == t]
        total_poss = len(tdf)

        def compute_rva(cluster_subset):
            rva = 0.0
            n_used = 0
            for c in cluster_subset:
                if team_n[t][c] < MIN_POSS:
                    continue
                league_pva_c = team_loo_pva[t][c]
                if np.isnan(league_pva_c):
                    continue
                usage_diff = team_usage[t][c] - team_loo_usage[t][c]
                rva += usage_diff * league_pva_c
                n_used += 1
            return rva, n_used

        stratum_results = {}
        for stratum_name, prefix in STRATA.items():
            stratum_clusters = [c for c in all_clusters if c.startswith(prefix)]
            rva_val, n_used = compute_rva(stratum_clusters)
            stratum_results[stratum_name] = (rva_val, n_used)

        rva_all = sum(stratum_results[s][0] for s in STRATA)
        mean_pva = tdf['total_pva'].mean()

        row = {
            'league':           league_name,
            'team':             t,
            'n_possessions':    total_poss,
            'pct_unassigned':   round(team_unassigned_pct[t], 2),
            'rva_all':          round(rva_all, 6),
            'mean_pva':         round(mean_pva, 6),
        }
        for stratum_name in STRATA:
            rva_val, n_used = stratum_results[stratum_name]
            key = stratum_name.lower()[:3]  # def, mid, att
            row[f'rva_{key}'] = round(rva_val, 6)
            row[f'n_clusters_{key}'] = n_used
        all_team_rows.append(row)

    league_rows = [r for r in all_team_rows if r['league'] == league_name]
    league_df = pd.DataFrame(league_rows).sort_values('rva_all', ascending=False)

    print(f"\n  Routing Value Added (RVA) per team:")
    print(f"  {'Team':<30} {'Def':>10} {'Mid':>10} {'Att':>10} {'Total':>10} "
          f"{'MeanPVA':>9} {'%Unasgn':>8}")
    print(f"  {'-'*92}")
    for _, row in league_df.iterrows():
        print(f"  {row['team']:<30} "
              f"{row['rva_def']:>10.6f} "
              f"{row['rva_mid']:>10.6f} "
              f"{row['rva_att']:>10.6f} "
              f"{row['rva_all']:>10.6f} "
              f"{row['mean_pva']:>9.6f} {row['pct_unassigned']:>8.1f}")

# ==============================================================================
# RVA → GOAL DIFFERENCE TRANSLATION
# ==============================================================================

all_team_df = pd.DataFrame(all_team_rows)

section("RVA → GOAL DIFFERENCE TRANSLATION")

outcome_dfs = []
for league_name, (comp_id, season_id) in COMPLETE_SEASONS.items():
    path = os.path.join(data_dir, 'matches', comp_id, f'{season_id}.json')
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping {league_name}")
        continue
    odf = load_outcomes(path)
    odf['league'] = league_name
    outcome_dfs.append(odf)

pooled_outcomes = pd.concat(outcome_dfs, ignore_index=True)
pooled = all_team_df[all_team_df['league'].isin(COMPLETE_SEASONS)].merge(
    pooled_outcomes, on=['league', 'team'])

slope, intercept, r_value, p_value, std_err = linregress(
    pooled['mean_pva'], pooled['goal_diff_per_game'])

print(f"  OLS: GD/Game = {slope:.2f} × Mean_PVA + ({intercept:.4f})")
print(f"    R² = {r_value**2:.3f}, p = {p_value:.4f}, N = {len(pooled)} team-seasons")
print(f"    1 PVA unit = {slope:.2f} GD/Game")

all_team_df['rva_gd'] = round(all_team_df['rva_all'] * slope, 4)
all_team_df['rva_pct_of_pva'] = round(
    all_team_df['rva_all'] / all_team_df['mean_pva'] * 100, 2)

r_sq = r_value ** 2
print(f"\n  Caveat: Mean PVA explains {r_sq*100:.1f}% of GD/Game variance (R²={r_sq:.3f}).")
print(f"  GD/Game values below are upper bounds — the slope captures the total")
print(f"  PVA–GD association including confounds beyond routing.")

print(f"\n  RVA translated to GD/Game:")
for league_name in sorted(all_team_df['league'].unique()):
    ldf = all_team_df[all_team_df['league'] == league_name].sort_values('rva_gd', ascending=False)
    print(f"\n  {league_name}:")
    print(f"  {'Team':<30} {'RVA (PVA)':>12} {'% of PVA':>9} {'RVA (GD/Game)':>14}")
    print(f"  {'-'*69}")
    for _, row in ldf.iterrows():
        print(f"  {row['team']:<30} {row['rva_all']:>12.6f} "
              f"{row['rva_pct_of_pva']:>+8.1f}% {row['rva_gd']:>+14.4f}")

# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================

section("SAVING OUTPUTS")

all_team_df.to_csv(os.path.join(output_dir, 'team_pattern_metrics.csv'), index=False)
print(f"  Saved: team_pattern_metrics.csv ({len(all_team_df)} team-seasons)")

print("\n" + "=" * 80)
print("  PHASE 3 COMPLETE — Routing Value Added (RVA) analysis done")
print("=" * 80)
