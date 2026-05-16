"""
Benchmarks PVA against xT, xG, and simple baselines for predicting season outcomes.
  python metric_benchmark.py
"""

import os
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
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

METRICS = [
    ('pva_per_game',              'PVA/Game'),
    ('xt_per_game',               'xT/Game'),
    ('xg_per_game',               'xG/Game'),
    ('possession_pct',            'Possession %'),
    ('prog_passes_per_game',      'Prog Passes/Game'),
    ('prog_carries_per_game',     'Prog Carries/Game'),
    ('line_breaks_per_game',      'Line Breaks/Game'),
    ('pen_box_entries_per_game',  'Pen Box Entries/Game'),
    ('pressure_actions_per_game', 'Pressure Actions/Game'),
    ('ft_entries_per_game',       'Final 3rd Entries/Game'),
]

OUTCOMES = [
    ('goals_per_game',     'Goals/Game'),
    ('goal_diff_per_game', 'GD/Game'),
]

MATCH_METRICS = [
    ('pva',              'PVA'),
    ('xt',               'xT'),
    ('xg',               'xG'),
    ('possession_pct',   'Possession %'),
    ('prog_passes',      'Prog Passes'),
    ('prog_carries',     'Prog Carries'),
    ('line_breaks',      'Line Breaks'),
    ('pen_box_entries',  'Pen Box Entries'),
    ('pressure_actions', 'Pressure Actions'),
    ('ft_entries',       'Final 3rd Entries'),
]

# ==============================================================================
# PATHS
# ==============================================================================

script_dir    = os.path.dirname(os.path.abspath(__file__))
project_root  = os.path.dirname(os.path.dirname(script_dir))
processed_dir = os.path.join(project_root, 'data', 'processed')
data_dir      = os.path.join(project_root, 'data')
results_dir   = os.path.join(project_root, 'results')
output_dir    = os.path.join(results_dir, 'multi_league')
os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# HELPERS
# ==============================================================================

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def section(title):
    print(f"\n{'=' * 80}\n  {title}\n{'=' * 80}")

def compute_cell_idx(x, y):
    col = min(int(x / (PITCH_LENGTH / X_BINS)), X_BINS - 1)
    row = min(int(y / (PITCH_WIDTH / Y_BINS)), Y_BINS - 1)
    return col * Y_BINS + row

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

def get_pitch_third(x):
    if x < DEFENSIVE_THIRD_END: return 'defensive'
    if x < MIDFIELD_THIRD_END:  return 'middle'
    return 'attacking'

def crosses_third_forward(start_x, end_x):
    s = get_pitch_third(start_x)
    e = get_pitch_third(end_x)
    return ((s == 'defensive' and e in ['middle', 'attacking']) or
            (s == 'middle' and e == 'attacking'))

def is_in_penalty_box(x, y):
    return x >= 102 and 18 <= y <= 62

def enters_penalty_box(start_x, start_y, end_x, end_y):
    return not is_in_penalty_box(start_x, start_y) and is_in_penalty_box(end_x, end_y)


# ==============================================================================
# COMPUTE METRICS FOR ONE LEAGUE
# ==============================================================================

def compute_league_metrics(actions_df, xt_grid, outcomes_df):
    """Compute all benchmark metrics. Returns (season_df, match_df)."""
    all_team_matches = actions_df[['team', 'match_id']].drop_duplicates()

    def match_count(event_df, col_name):
        cts = event_df.groupby(['team', 'match_id']).size().rename(col_name).reset_index()
        merged = all_team_matches.merge(cts, on=['team', 'match_id'], how='left')
        merged[col_name] = merged[col_name].fillna(0)
        return merged[['team', 'match_id', col_name]]

    passes = actions_df[actions_df['type'] == 'Pass'].copy()
    passes['end_xy'] = passes['pass_end_location'].apply(parse_location)
    passes['end_x'] = passes['end_xy'].apply(lambda xy: xy[0] if xy else None)
    passes['end_y'] = passes['end_xy'].apply(lambda xy: xy[1] if xy else None)
    passes['end_cell'] = passes['end_xy'].apply(
        lambda xy: compute_cell_idx(xy[0], xy[1]) if xy else None)
    passes_complete = passes[passes['pass_outcome'].isna()].copy()

    carries = actions_df[actions_df['type'] == 'Carry'].copy()
    carries['end_xy'] = carries['carry_end_location'].apply(parse_location)
    carries['end_x'] = carries['end_xy'].apply(lambda xy: xy[0] if xy else None)
    carries['end_y'] = carries['end_xy'].apply(lambda xy: xy[1] if xy else None)
    carries['end_cell'] = carries['end_xy'].apply(
        lambda xy: compute_cell_idx(xy[0], xy[1]) if xy else None)

    shots = actions_df[actions_df['type'] == 'Shot'].copy()

    m_pva = actions_df.groupby(['team', 'match_id'])['pva'].sum().rename('pva').reset_index()

    passes_complete['xt_delta'] = passes_complete.apply(
        lambda r: (xt_grid.get(r['end_cell'], 0) - xt_grid.get(r['cell_idx'], 0))
        if r['end_cell'] is not None else 0.0, axis=1)
    carries['xt_delta'] = carries.apply(
        lambda r: (xt_grid.get(r['end_cell'], 0) - xt_grid.get(r['cell_idx'], 0))
        if r['end_cell'] is not None else 0.0, axis=1)
    shots['xt_delta'] = shots.apply(
        lambda r: r['shot_statsbomb_xg'] - xt_grid.get(r['cell_idx'], 0), axis=1)
    xt_actions = pd.concat([
        passes_complete[['team', 'match_id', 'xt_delta']],
        carries[['team', 'match_id', 'xt_delta']],
        shots[['team', 'match_id', 'xt_delta']],
    ])
    m_xt = xt_actions.groupby(['team', 'match_id'])['xt_delta'].sum().rename('xt').reset_index()

    m_xg = shots.groupby(['team', 'match_id'])['shot_statsbomb_xg'].sum().rename('xg').reset_index()
    m_xg = all_team_matches.merge(m_xg, on=['team', 'match_id'], how='left')
    m_xg['xg'] = m_xg['xg'].fillna(0)

    mt_actions = actions_df.groupby(['match_id', 'team']).size().rename('team_actions').reset_index()
    tot_actions = actions_df.groupby('match_id').size().rename('total_actions').reset_index()
    m_poss = mt_actions.merge(tot_actions, on='match_id')
    m_poss['possession_pct'] = m_poss['team_actions'] / m_poss['total_actions']

    passes_complete['forward_dist'] = passes_complete['end_x'] - passes_complete['x']
    m_pp = match_count(passes_complete[passes_complete['forward_dist'] >= 10], 'prog_passes')

    carries['forward_dist'] = carries['end_x'] - carries['x']
    m_pc = match_count(carries[carries['forward_dist'] >= 10], 'prog_carries')

    lb_passes = passes_complete[passes_complete.apply(
        lambda r: crosses_third_forward(r['x'], r['end_x'])
        if r['end_x'] is not None else False, axis=1)]
    lb_carries = carries[carries.apply(
        lambda r: crosses_third_forward(r['x'], r['end_x'])
        if r['end_x'] is not None else False, axis=1)]
    lb_all = pd.concat([lb_passes[['team', 'match_id']], lb_carries[['team', 'match_id']]])
    m_lb = match_count(lb_all, 'line_breaks')

    pb_passes = passes_complete[passes_complete.apply(
        lambda r: enters_penalty_box(r['x'], r['y'], r['end_x'], r['end_y'])
        if r['end_x'] is not None and r['end_y'] is not None else False, axis=1)]
    pb_carries = carries[carries.apply(
        lambda r: enters_penalty_box(r['x'], r['y'], r['end_x'], r['end_y'])
        if r['end_x'] is not None and r['end_y'] is not None else False, axis=1)]
    pb_all = pd.concat([pb_passes[['team', 'match_id']], pb_carries[['team', 'match_id']]])
    m_pb = match_count(pb_all, 'pen_box_entries')

    m_pr = match_count(actions_df[actions_df['under_pressure'] == True], 'pressure_actions')

    ft_passes = passes_complete[(passes_complete['x'] < 80) & (passes_complete['end_x'] >= 80)]
    ft_carries = carries[(carries['x'] < 80) & (carries['end_x'] >= 80)]
    ft_all = pd.concat([ft_passes[['team', 'match_id']], ft_carries[['team', 'match_id']]])
    m_ft = match_count(ft_all, 'ft_entries')

    match_metrics = all_team_matches.copy()
    for mdf in [m_pva, m_xt, m_xg[['team', 'match_id', 'xg']],
                m_poss[['team', 'match_id', 'possession_pct']],
                m_pp, m_pc, m_lb, m_pb, m_pr, m_ft]:
        match_metrics = match_metrics.merge(mdf, on=['team', 'match_id'])

    metric_cols = [c for c, _ in MATCH_METRICS]
    season = match_metrics.groupby('team')[metric_cols].mean().reset_index()
    season = season.rename(columns={
        'pva': 'pva_per_game', 'xt': 'xt_per_game', 'xg': 'xg_per_game',
        'prog_passes': 'prog_passes_per_game', 'prog_carries': 'prog_carries_per_game',
        'line_breaks': 'line_breaks_per_game', 'pen_box_entries': 'pen_box_entries_per_game',
        'pressure_actions': 'pressure_actions_per_game', 'ft_entries': 'ft_entries_per_game',
    })

    team_metrics = outcomes_df[['team', 'goals_per_game', 'goal_diff_per_game']].merge(season, on='team')

    return team_metrics, match_metrics

# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

section("METRIC BENCHMARK — PVA vs Components & Baselines")

all_corr_rows  = []
all_league_dfs = []
all_match_diffs     = []
all_match_corr_rows = []

for league in LEAGUES:
    section(f"Processing {league['name']}")

    csv_path = os.path.join(processed_dir, league['slug'] + '_actions_with_pva.csv')
    actions_df = pd.read_csv(csv_path, low_memory=False)

    xt_path = os.path.join(processed_dir, league['slug'] + '_xt_grid.csv')
    xt_df = pd.read_csv(xt_path)
    xt_grid = xt_df.set_index('cell_idx')['xt_value'].to_dict()

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
            r['gf'] += scored
            r['ga'] += conceded
            r['gp'] += 1

    outcomes_rows = []
    for team_name, r in record.items():
        outcomes_rows.append({
            'team': team_name,
            'goals_per_game': round(r['gf'] / r['gp'], 4),
            'goal_diff_per_game': round((r['gf'] - r['ga']) / r['gp'], 4),
        })
    outcomes_df = pd.DataFrame(outcomes_rows)

    n_teams = len(outcomes_df)
    print(f"  {len(actions_df):,} actions, {n_teams} teams, {len(matches)} matches")

    team_metrics, match_metrics = compute_league_metrics(actions_df, xt_grid, outcomes_df)
    team_metrics['league'] = league['name']
    all_league_dfs.append(team_metrics)

    print(f"\n  Season-level Spearman correlations (N={n_teams}):")
    print(f"  {'Metric':<25} {'Outcome':<14} {'r':>7}  {'p':>8}  {'sig':>4}")
    print(f"  {'-'*62}")

    for m_col, m_label in METRICS:
        for o_col, o_label in OUTCOMES:
            r_s, p_s = spearmanr(team_metrics[m_col], team_metrics[o_col])
            all_corr_rows.append({
                'league': league['name'],
                'metric': m_label, 'metric_col': m_col,
                'outcome': o_label, 'outcome_col': o_col,
                'n': n_teams,
                'spearman_r': round(r_s, 4),
                'spearman_p': round(p_s, 4),
                'sig': sig_stars(p_s),
            })
            print(f"  {m_label:<25} {o_label:<14} {r_s:>7.3f}  {p_s:>8.4f}  {sig_stars(p_s):>4}")

    match_diff_rows = []
    for m in matches:
        mid = m['match_id']
        home_name = m['home_team']['home_team_name']
        away_name = m['away_team']['away_team_name']
        goal_diff = m['home_score'] - m['away_score']

        home = match_metrics[(match_metrics['match_id'] == mid) & (match_metrics['team'] == home_name)]
        away = match_metrics[(match_metrics['match_id'] == mid) & (match_metrics['team'] == away_name)]
        if len(home) == 0 or len(away) == 0:
            continue

        row = {'match_id': mid, 'goal_diff': goal_diff}
        for mc, _ in MATCH_METRICS:
            row[mc + '_diff'] = home.iloc[0][mc] - away.iloc[0][mc]
        match_diff_rows.append(row)

    match_diffs = pd.DataFrame(match_diff_rows)
    match_diffs['league'] = league['name']
    all_match_diffs.append(match_diffs)

    n_matches = len(match_diffs)
    print(f"\n  Match-level Spearman: metric_diff vs goal_diff (N={n_matches} matches):")
    print(f"  {'Metric':<25} {'Outcome':<14} {'r':>7}  {'p':>8}  {'sig':>4}")
    print(f"  {'-'*62}")

    for mc, ml in MATCH_METRICS:
        r_s, p_s = spearmanr(match_diffs[mc + '_diff'], match_diffs['goal_diff'])
        all_match_corr_rows.append({
            'league': league['name'],
            'metric': ml, 'metric_col': mc + '_diff',
            'n': n_matches,
            'spearman_r': round(r_s, 4),
            'spearman_p': round(p_s, 4),
            'sig': sig_stars(p_s),
        })
        print(f"  {ml:<25} {'Goal Diff':<14} {r_s:>7.3f}  {p_s:>8.4f}  {sig_stars(p_s):>4}")

# ==============================================================================
# POOLED ANALYSIS
# ==============================================================================

all_corr_df = pd.DataFrame(all_corr_rows)

pooled_df = pd.concat(all_league_dfs, ignore_index=True)
n_pooled = len(pooled_df)

section(f"POOLED SEASON-LEVEL ANALYSIS (N={n_pooled} team-seasons)")

print(f"\n  {'Metric':<25} {'Outcome':<14} {'r':>7}  {'p':>8}  {'sig':>4}")
print(f"  {'-'*62}")

pooled_corr_rows = []
for m_col, m_label in METRICS:
    for o_col, o_label in OUTCOMES:
        r_s, p_s = spearmanr(pooled_df[m_col], pooled_df[o_col])
        pooled_corr_rows.append({
            'metric': m_label, 'outcome': o_label,
            'spearman_r': round(r_s, 4), 'spearman_p': round(p_s, 4),
            'sig': sig_stars(p_s), 'n': n_pooled,
        })
        print(f"  {m_label:<25} {o_label:<14} {r_s:>7.3f}  {p_s:>8.4f}  {sig_stars(p_s):>4}")

pooled_match_diffs = pd.concat(all_match_diffs, ignore_index=True)
n_pooled_matches = len(pooled_match_diffs)

section(f"POOLED MATCH-LEVEL ANALYSIS (N={n_pooled_matches} matches)")

print(f"\n  {'Metric':<25} {'Outcome':<14} {'r':>7}  {'p':>8}  {'sig':>4}")
print(f"  {'-'*62}")

pooled_match_corr_rows = []
for mc, ml in MATCH_METRICS:
    r_s, p_s = spearmanr(pooled_match_diffs[mc + '_diff'], pooled_match_diffs['goal_diff'])
    pooled_match_corr_rows.append({
        'metric': ml, 'outcome': 'Goal Diff',
        'spearman_r': round(r_s, 4), 'spearman_p': round(p_s, 4),
        'sig': sig_stars(p_s), 'n': n_pooled_matches,
    })
    print(f"  {ml:<25} {'Goal Diff':<14} {r_s:>7.3f}  {p_s:>8.4f}  {sig_stars(p_s):>4}")

# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================

section("SAVING OUTPUTS")

all_corr_df.to_csv(os.path.join(output_dir, 'metric_benchmark_per_league.csv'), index=False)
print(f"  Saved: metric_benchmark_per_league.csv")

pd.DataFrame(pooled_corr_rows).to_csv(os.path.join(output_dir, 'metric_benchmark_pooled.csv'), index=False)
print(f"  Saved: metric_benchmark_pooled.csv")

pd.DataFrame(all_match_corr_rows).to_csv(os.path.join(output_dir, 'metric_benchmark_match_level_per_league.csv'), index=False)
print(f"  Saved: metric_benchmark_match_level_per_league.csv")

pd.DataFrame(pooled_match_corr_rows).to_csv(os.path.join(output_dir, 'metric_benchmark_match_level_pooled.csv'), index=False)
print(f"  Saved: metric_benchmark_match_level_pooled.csv")

# ==============================================================================
# FIGURES
# ==============================================================================

for league in LEAGUES:
    league_df = pd.concat(all_league_dfs, ignore_index=True)
    league_df = league_df[league_df['league'] == league['name']]
    league_corrs = all_corr_df[all_corr_df['league'] == league['name']]

    league_stats_dir = os.path.join(results_dir, league['slug'])
    os.makedirs(league_stats_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 7))
    metric_labels = [m[1] for m in METRICS]
    n_metrics = len(METRICS)
    bar_width = 0.25
    x = np.arange(n_metrics)
    colors = ['#2c7bb6', '#d7191c']

    for i, (o_col, o_label) in enumerate(OUTCOMES):
        vals = [league_corrs[(league_corrs['metric'] == m[1]) &
                             (league_corrs['outcome'] == o_label)]['spearman_r'].iloc[0]
                for m in METRICS]
        ax.bar(x + i * bar_width, vals, bar_width, label=o_label,
               color=colors[i], edgecolor='black', linewidth=0.5)
        for j, v in enumerate(vals):
            p = league_corrs[(league_corrs['metric'] == METRICS[j][1]) &
                             (league_corrs['outcome'] == o_label)]['spearman_p'].iloc[0]
            stars = sig_stars(p)
            if stars != 'ns':
                y_off = 0.02 if v >= 0 else -0.02
                va = 'bottom' if v >= 0 else 'top'
                ax.text(x[j] + i * bar_width, v + y_off, stars,
                        ha='center', va=va, fontsize=7, fontweight='bold')

    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(metric_labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Spearman r', fontsize=11)
    ax.set_title(f'{league["name"]} — Metric Correlations with Season Outcomes  (N={len(league_df)})',
                 fontsize=13, fontweight='bold')
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    fig_path = os.path.join(league_stats_dir, 'metric_benchmark_bars.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {league['slug']}/metric_benchmark_bars.png")

    fig, axes = plt.subplots(len(OUTCOMES), n_metrics, figsize=(4 * n_metrics, 4 * len(OUTCOMES)))
    fig.suptitle(f'{league["name"]} — Metric vs Outcome Scatter  (N={len(league_df)})',
                 fontsize=14, fontweight='bold', y=1.01)

    for j, (m_col, m_label) in enumerate(METRICS):
        for i, (o_col, o_label) in enumerate(OUTCOMES):
            ax = axes[i, j]
            xs = league_df[m_col].values
            ys = league_df[o_col].values
            r_s, p_s = spearmanr(xs, ys)
            ax.scatter(xs, ys, s=50, c='steelblue', edgecolors='black', linewidth=0.5, zorder=3)
            for _, row in league_df.iterrows():
                ax.annotate(row['team'][:3], (row[m_col], row[o_col]),
                            textcoords='offset points', xytext=(4, 3), fontsize=6, color='gray')
            z = np.polyfit(xs, ys, 1)
            x_line = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.5, linewidth=1)
            ax.set_title(f'r={r_s:.2f} {sig_stars(p_s)}', fontsize=8)
            if i == len(OUTCOMES) - 1:
                ax.set_xlabel(m_label, fontsize=7)
            if j == 0:
                ax.set_ylabel(o_label, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    fig_path = os.path.join(league_stats_dir, 'metric_benchmark_scatter.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {league['slug']}/metric_benchmark_scatter.png")

print(f"\n{'=' * 80}")
print(f"  METRIC BENCHMARK COMPLETE")
print(f"{'=' * 80}")
