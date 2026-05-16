"""
Zone statistical testing: chi-square usage, t-test efficiency, and ZVA analysis.
Run: python zone_statistical_testing.py
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import chisquare, ttest_1samp, linregress
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PITCH_LENGTH = 120
PITCH_WIDTH  = 80
X_BINS = 6
Y_BINS = 3
N_ZONES = X_BINS * Y_BINS   # 18

cell_width  = PITCH_LENGTH / X_BINS
cell_height = PITCH_WIDTH  / Y_BINS

N_MIN  = 30
ALPHA  = 0.05

LEAGUES = [
    {
        'name': 'FA WSL 2018-19',
        'slug': 'fa_wsl_2018-19',
        'matches_json': 'data/matches/37/4.json',
    },
    {
        'name': 'FA WSL 2020-21',
        'slug': 'fa_wsl_2020-21',
        'matches_json': 'data/matches/37/90.json',
    },
]

# ==============================================================================
# PATHS
# ==============================================================================

script_dir    = os.path.dirname(os.path.abspath(__file__))
project_root  = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
processed_dir = os.path.join(project_root, 'data', 'processed')
output_dir    = os.path.join(project_root, 'results', 'multi_league')
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# HELPERS
# ==============================================================================

def compute_cell_idx(x, y):
    cx = min(int(x / cell_width),  X_BINS - 1)
    cy = min(int(y / cell_height), Y_BINS - 1)
    return cx * Y_BINS + cy

def zone_label(cell_idx):
    col = int(cell_idx) // Y_BINS + 1
    row = int(cell_idx) % Y_BINS + 1
    return f"C{col}_R{row}"

def sig_stars(p):
    if pd.isna(p): return 'ns'
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def section(title):
    print(f"\n{'=' * 80}\n  {title}\n{'=' * 80}")


# ==============================================================================
# MAIN LOOP — process each complete league-season
# ==============================================================================

section("ZONE STATISTICAL TESTING")

all_chisq_rows   = []
all_pairwise_rows = []
all_ttest_rows   = []
all_zva_rows     = []

for league in LEAGUES:
    section(f"{league['name']}")

    csv_path = os.path.join(processed_dir, league['slug'] + '_actions_with_pva.csv')
    actions_df = pd.read_csv(csv_path)
    actions_df['cell_6x3'] = actions_df.apply(
        lambda r: compute_cell_idx(r['x'], r['y']), axis=1)

    actions_df = actions_df[
        ~((actions_df['x'] >= 114) &
          (actions_df['y'] >= 30) &
          (actions_df['y'] <= 50))
    ].copy()

    teams = sorted(actions_df['team'].unique())
    n_teams = len(teams)
    zones = list(range(N_ZONES))
    print(f"  {len(actions_df):,} actions, {n_teams} teams")

    matches_path = os.path.join(project_root, league['matches_json'])
    with open(matches_path) as f:
        matches = json.load(f)
    print(f"  {len(matches)} matches")

    record = {}
    for m in matches:
        for team_name, scored, conceded in [
            (m['home_team']['home_team_name'], m['home_score'], m['away_score']),
            (m['away_team']['away_team_name'], m['away_score'], m['home_score']),
        ]:
            r = record.setdefault(team_name, dict(gf=0, ga=0, gp=0))
            r['gf'] += scored
            r['ga'] += conceded
            r['gp'] += 1

    outcomes_df = pd.DataFrame([
        {
            'team': t,
            'gf_per_game': round(r['gf'] / r['gp'], 4),
            'ga_per_game': round(r['ga'] / r['gp'], 4),
            'gd_per_game': round((r['gf'] - r['ga']) / r['gp'], 4),
        }
        for t, r in record.items()
    ])

    team_pva_means  = {}
    team_zone_props = {}
    team_zone_n     = {}
    team_zone_pvas  = {}

    for t in teams:
        ta = actions_df[actions_df['team'] == t]
        tc = ta.groupby('cell_6x3').size()
        team_zone_n[t]     = tc.to_dict()
        team_zone_props[t] = (tc / tc.sum()).to_dict()
        team_pva_means[t]  = ta.groupby('cell_6x3')['pva'].mean().to_dict()
        team_zone_pvas[t]  = {z: grp['pva'].values
                              for z, grp in ta.groupby('cell_6x3')}

    loo_pva   = {}
    loo_usage = {}
    for t in teams:
        loo_pva[t]  = {}
        loo_usage[t] = {}
        others = [o for o in teams if o != t]
        for z in zones:
            other_pvas  = [team_pva_means[o].get(z, np.nan) for o in others]
            other_usage = [team_zone_props[o].get(z, 0.0)   for o in others]
            valid = [v for v in other_pvas if not np.isnan(v)]
            loo_pva[t][z]  = np.mean(valid) if valid else np.nan
            loo_usage[t][z] = np.mean(other_usage)

    # ==================================================================
    # SECTION 1: ZONE USAGE PATTERNS — Chi-square
    # ==================================================================

    print(f"\n  --- Zone Usage Patterns (Chi-square vs LOO baseline) ---\n")
    print(f"  {'Team':<30} {'chi2':>10} {'p':>10} {'sig':>5} {'N':>8}")
    print(f"  {'-' * 67}")

    for t in teams:
        observed = np.array([team_zone_n[t].get(z, 0) for z in zones], dtype=float)
        expected_prop = np.array([loo_usage[t][z] for z in zones])
        total = observed.sum()
        expected = expected_prop * total

        mask = expected >= 5
        if mask.sum() < 2:
            continue
        obs_f = observed[mask]
        exp_f = expected[mask]
        chi2, p = chisquare(obs_f, exp_f)

        all_chisq_rows.append({
            'league': league['name'], 'team': t,
            'chi2': round(chi2, 2), 'p': round(p, 6),
            'sig': sig_stars(p), 'n_actions': int(total),
            'zones_tested': int(mask.sum()),
        })
        print(f"  {t:<30} {chi2:>10.2f} {p:>10.4f} {sig_stars(p):>5} {int(total):>8,}")

    n_sig_usage = sum(1 for r in all_chisq_rows
                      if r['league'] == league['name'] and r['p'] < ALPHA)
    print(f"\n  {n_sig_usage}/{n_teams} teams have significantly different zone usage")

    # ==================================================================
    # SECTION 1b: PAIRWISE ZONE USAGE — Chi-square (team vs team)
    # ==================================================================

    print(f"\n  --- Pairwise Zone Usage (Chi-square, all team pairs) ---\n")
    n_pairs = 0
    n_sig_pairs = 0

    for team_a, team_b in combinations(teams, 2):
        obs_a = np.array([team_zone_n[team_a].get(z, 0) for z in zones], dtype=float)
        obs_b = np.array([team_zone_n[team_b].get(z, 0) for z in zones], dtype=float)
        total_a = obs_a.sum()
        total_b = obs_b.sum()

        pooled_prop = (obs_a + obs_b) / (total_a + total_b)
        exp_a = pooled_prop * total_a
        exp_b = pooled_prop * total_b

        mask = (exp_a >= 5) & (exp_b >= 5)
        if mask.sum() < 2:
            continue

        chi2, p = chisquare(obs_a[mask], exp_a[mask])
        n_pairs += 1
        if p < ALPHA:
            n_sig_pairs += 1

        all_pairwise_rows.append({
            'league': league['name'],
            'team_a': team_a,
            'team_b': team_b,
            'chi2': round(chi2, 2),
            'p': round(p, 6),
            'sig': sig_stars(p),
            'n_actions_a': int(total_a),
            'n_actions_b': int(total_b),
            'zones_tested': int(mask.sum()),
        })

    print(f"  {n_sig_pairs}/{n_pairs} pairs significantly different (p < {ALPHA})")

    # ==================================================================
    # SECTION 2: ZONE EFFICIENCY — One-sample t-test, FDR-corrected
    # ==================================================================

    print(f"\n  --- Zone Efficiency (t-test vs LOO baseline, FDR-corrected) ---\n")

    league_ttest = []
    for t in teams:
        for z in zones:
            n_z = team_zone_n[t].get(z, 0)
            if n_z < N_MIN:
                continue
            baseline = loo_pva[t][z]
            if np.isnan(baseline):
                continue
            pva_arr = team_zone_pvas[t].get(z, np.array([]))
            if len(pva_arr) < N_MIN:
                continue
            t_stat, p_raw = ttest_1samp(pva_arr, baseline)
            mean_team = np.mean(pva_arr)
            std_team = np.std(pva_arr, ddof=1)
            cohen_d = (mean_team - baseline) / std_team if std_team > 0 else 0.0
            direction = 'above' if mean_team > baseline else 'below'
            league_ttest.append({
                'league': league['name'], 'team': t,
                'zone': zone_label(z), 'zone_idx': z,
                'n': n_z, 'mean_team': round(mean_team, 6),
                'mean_league': round(baseline, 6),
                'direction': direction, 'cohen_d': round(cohen_d, 4),
                't_stat': round(t_stat, 4), 'p_raw': p_raw,
            })

    if league_ttest:
        p_vals = [r['p_raw'] for r in league_ttest]
        _, q_vals, _, _ = multipletests(p_vals, alpha=ALPHA, method='fdr_bh')
        for i, r in enumerate(league_ttest):
            r['q_fdr'] = round(q_vals[i], 6)
            r['sig'] = sig_stars(q_vals[i])

    all_ttest_rows.extend(league_ttest)

    print(f"  {'Team':<30} {'Tested':>7} {'Above':>7} {'Below':>7}")
    print(f"  {'-' * 55}")
    for t in teams:
        t_rows = [r for r in league_ttest if r['team'] == t]
        tested = len(t_rows)
        above = sum(1 for r in t_rows if r['q_fdr'] < ALPHA and r['direction'] == 'above')
        below = sum(1 for r in t_rows if r['q_fdr'] < ALPHA and r['direction'] == 'below')
        print(f"  {t:<30} {tested:>7} {above:>7} {below:>7}")

    n_sig_eff = sum(1 for r in league_ttest if r['q_fdr'] < ALPHA)
    print(f"\n  {n_sig_eff} significant out of {len(league_ttest)} tests (FDR < {ALPHA})")

    # ==================================================================
    # SECTION 3: ZVA (Zone Value Added)
    # ==================================================================

    print(f"\n  --- ZVA: Zone Value Added ---\n")
    print(f"  {'Team':<30} {'ZVA':>10} {'zones':>6} {'Mean PVA':>10}")
    print(f"  {'-' * 60}")

    for t in teams:
        zva = 0.0
        n_used = 0
        for z in zones:
            if team_zone_n[t].get(z, 0) < N_MIN:
                continue
            league_pva_z = loo_pva[t][z]
            if np.isnan(league_pva_z):
                continue
            usage_diff = team_zone_props[t].get(z, 0.0) - loo_usage[t][z]
            zva += usage_diff * league_pva_z
            n_used += 1

        mean_pva = actions_df[actions_df['team'] == t]['pva'].mean()

        all_zva_rows.append({
            'league': league['name'], 'team': t,
            'ZVA': round(zva, 6),
            'zones_used': n_used,
            'mean_pva': round(mean_pva, 6),
        })

        print(f"  {t:<30} {zva:>10.6f} {n_used:>6} {mean_pva:>10.6f}")

# ==============================================================================
# ZVA → GOAL DIFFERENCE TRANSLATION
# ==============================================================================

zva_df = pd.DataFrame(all_zva_rows)

section("ZVA → GOAL DIFFERENCE TRANSLATION")

pooled_outcomes = []
for league in LEAGUES:
    matches_path = os.path.join(project_root, league['matches_json'])
    with open(matches_path) as f:
        matches = json.load(f)
    record = {}
    for m in matches:
        for team_name, scored, conceded in [
            (m['home_team']['home_team_name'], m['home_score'], m['away_score']),
            (m['away_team']['away_team_name'], m['away_score'], m['home_score']),
        ]:
            r = record.setdefault(team_name, dict(gf=0, ga=0, gp=0))
            r['gf'] += scored; r['ga'] += conceded; r['gp'] += 1
    for t, r in record.items():
        pooled_outcomes.append({
            'league': league['name'], 'team': t,
            'gd_per_game': round((r['gf'] - r['ga']) / r['gp'], 4),
        })

pooled_outcomes_df = pd.DataFrame(pooled_outcomes)
pooled = zva_df.merge(pooled_outcomes_df, on=['league', 'team'])

slope, intercept, r_value, p_value, std_err = linregress(
    pooled['mean_pva'], pooled['gd_per_game'])

print(f"  OLS: GD/Game = {slope:.2f} × Mean_PVA + ({intercept:.4f})")
print(f"    R² = {r_value**2:.3f}, p = {p_value:.4f}, N = {len(pooled)} team-seasons")
print(f"    1 PVA unit = {slope:.2f} GD/Game")

zva_df['ZVA_gd'] = round(zva_df['ZVA'] * slope, 4)
zva_df['ZVA_pct_of_pva'] = round(
    zva_df['ZVA'] / zva_df['mean_pva'] * 100, 2)

r_sq = r_value ** 2
print(f"\n  Caveat: Mean PVA explains {r_sq*100:.1f}% of GD/Game variance (R²={r_sq:.3f}).")
print(f"  GD/Game values below are upper bounds — the slope captures the total")
print(f"  PVA–GD association including confounds beyond routing.")

print(f"\n  ZVA translated to GD/Game:")
for ln in sorted(zva_df['league'].unique()):
    ldf = zva_df[zva_df['league'] == ln].sort_values('ZVA_gd', ascending=False)
    print(f"\n  {ln}:")
    print(f"  {'Team':<30} {'ZVA (PVA)':>12} {'% of PVA':>9} {'ZVA (GD/Game)':>14}")
    print(f"  {'-'*69}")
    for _, row in ldf.iterrows():
        print(f"  {row['team']:<30} {row['ZVA']:>12.6f} "
              f"{row['ZVA_pct_of_pva']:>+8.1f}% {row['ZVA_gd']:>+14.4f}")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

section("SAVING RESULTS")

pd.DataFrame(all_chisq_rows).to_csv(
    os.path.join(output_dir, 'zone_usage_chisquare.csv'), index=False)
print(f"  Saved: zone_usage_chisquare.csv")

pd.DataFrame(all_pairwise_rows).to_csv(
    os.path.join(output_dir, 'zone_usage_pairwise_chisquare.csv'), index=False)
print(f"  Saved: zone_usage_pairwise_chisquare.csv")

pd.DataFrame(all_ttest_rows).to_csv(
    os.path.join(output_dir, 'zone_efficiency_ttest.csv'), index=False)
print(f"  Saved: zone_efficiency_ttest.csv")

zva_df.to_csv(os.path.join(output_dir, 'zva_per_team.csv'), index=False)
print(f"  Saved: zva_per_team.csv")

print(f"\n{'=' * 80}")
print(f"  ZONE STATISTICAL TESTING COMPLETE")
print(f"{'=' * 80}")
