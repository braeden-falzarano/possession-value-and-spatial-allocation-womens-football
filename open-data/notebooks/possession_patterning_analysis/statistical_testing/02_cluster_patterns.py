"""
Phase 2: stratified K-Medoids clustering (FasterPAM) by starting third.
Route scouting sets k bounds; silhouette selects final k per stratum.
Run: python 02_cluster_patterns.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.spatial.distance import squareform, pdist
import kmedoids
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

RANDOM_STATE = 42

STRATA_CONFIG = {
    'Defensive': {'prefix': 'D', 'k_ceiling': 18},
    'Midfield':  {'prefix': 'M', 'k_ceiling': 18},
    'Attacking': {'prefix': 'A', 'k_ceiling': 9},
}

K_FLOOR = 6
ROUTE_THRESHOLDS = [0.25, 0.5, 1.0]
DISTANCE_PERCENTILE = 75
N_RESAMPLE = 20

# ==============================================================================
# PATHS
# ==============================================================================

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
data_dir     = os.path.join(project_root, 'data')
stats_dir    = os.path.join(project_root, 'results')
output_dir   = os.path.join(stats_dir, 'possession_patterns')
os.makedirs(output_dir, exist_ok=True)

input_path = os.path.join(output_dir, 'possessions_resampled.csv')

# ==============================================================================
# HELPERS
# ==============================================================================

PITCH_LENGTH = 120
PITCH_WIDTH  = 80

def section(title):
    print()
    print('=' * 80)
    print(f'  {title}')
    print('=' * 80)

def get_pitch_third(x_norm):
    x = x_norm * PITCH_LENGTH
    if x < 40:  return 'Defensive'
    if x < 80:  return 'Midfield'
    return 'Attacking'

def compute_grid_route(path_coords, n_resample=20):
    """Map path to 3x3 grid route fingerprint (e.g., 'DL-MC-AR')."""
    cells = []
    for i in range(n_resample):
        x, y = path_coords[2 * i], path_coords[2 * i + 1]  # normalized [0,1]
        depth = 'D' if x < 1/3 else ('M' if x < 2/3 else 'A')
        lat   = 'L' if y < 1/3 else ('C' if y < 2/3 else 'R')
        cell = depth + lat
        if not cells or cells[-1] != cell:
            cells.append(cell)
    return '-'.join(cells)

def route_scout(X_features, n_resample, thresholds=ROUTE_THRESHOLDS):
    """Count distinct grid routes at frequency thresholds. Returns (k_anchors, route_counts)."""
    routes = [compute_grid_route(row, n_resample) for row in X_features]
    route_counts = pd.Series(routes).value_counts()
    n = len(X_features)
    k_anchors = {t: int((route_counts >= n * t / 100).sum()) for t in thresholds}
    return k_anchors, route_counts

def run_k_sweep(D, X_scaled, k_range, stratum_name):
    metrics = []
    for k in k_range:
        result = kmedoids.fasterpam(D, k, random_state=RANDOM_STATE)
        labels = result.labels

        sil = silhouette_score(
            X_scaled, labels,
            sample_size=min(10000, len(X_scaled)),
            random_state=RANDOM_STATE,
        )
        ch = calinski_harabasz_score(X_scaled, labels)
        total_dist = result.loss

        metrics.append({
            'stratum': stratum_name,
            'k': k,
            'silhouette': round(sil, 6),
            'calinski_harabasz': round(ch, 2),
            'total_distance': round(total_dist, 2),
        })
        print(f"    k={k:>3}  Sil={sil:.4f}  CH={ch:>10.1f}  TotalDist={total_dist:>12.1f}")
    return pd.DataFrame(metrics)

def profile_clusters(df, cluster_col, centroids_dict, n_resample):
    cluster_ids = sorted(df[cluster_col].unique())
    if -1 in cluster_ids:
        cluster_ids = [c for c in cluster_ids if c != -1]
    cluster_ids = sorted(cluster_ids, key=lambda x: (x[0], int(x[1:])))
    rows = []
    for cid in cluster_ids:
        cluster = df[df[cluster_col] == cid]
        n = len(cluster)
        total_pva_vals = cluster['total_pva']
        mean_pva_vals  = cluster['mean_pva']
        centroid = centroids_dict[cid]
        start_x, start_y = centroid[0], centroid[1]
        end_x, end_y = centroid[-2], centroid[-1]
        net_x_progress = end_x - start_x
        start_mode = cluster['start_third'].mode().iloc[0] if n > 0 else 'N/A'
        end_mode   = cluster['end_third'].mode().iloc[0] if n > 0 else 'N/A'
        row = {
            'cluster_id':       cid,
            'stratum':          cluster['stratum'].iloc[0] if 'stratum' in cluster.columns else 'all',
            'n_possessions':    n,
            'pct_of_total':     round(n / len(df) * 100, 2),
            'mean_total_pva':   round(total_pva_vals.mean(), 6),
            'median_total_pva': round(total_pva_vals.median(), 6),
            'std_total_pva':    round(total_pva_vals.std(), 6),
            'mean_mean_pva':    round(mean_pva_vals.mean(), 6),
            'mean_n_actions':   round(cluster['n_actions'].mean(), 1),
            'median_n_actions': round(cluster['n_actions'].median(), 0),
            'start_third_mode': start_mode,
            'end_third_mode':   end_mode,
            'net_x_progress':   round(net_x_progress, 4),
            'centroid_start_x': round(start_x, 4),
            'centroid_start_y': round(start_y, 4),
            'centroid_end_x':   round(end_x, 4),
            'centroid_end_y':   round(end_y, 4),
        }
        for i in range(n_resample):
            row[f'centroid_p{i}_x'] = round(centroid[2*i], 6)
            row[f'centroid_p{i}_y'] = round(centroid[2*i+1], 6)
        rows.append(row)
    profiles = pd.DataFrame(rows)
    profiles = profiles.sort_values('mean_total_pva', ascending=False).reset_index(drop=True)
    return profiles

def print_profiles(profiles_df, total_n):
    print(f"\n  {'Cluster':>8} {'Stratum':>10} {'N':>7} {'%':>6} {'Mean PVA':>10} {'Median PVA':>11} "
          f"{'Actions':>8} {'Start→End':>15} {'ΔX':>7}")
    print(f"  {'-'*86}")
    for _, row in profiles_df.iterrows():
        arrow = f"{row['start_third_mode'][:3]}→{row['end_third_mode'][:3]}"
        pct = row['n_possessions'] / total_n * 100
        print(f"  {row['cluster_id']:>8} {row['stratum']:>10} {row['n_possessions']:>7,} {pct:>5.1f}% "
              f"{row['mean_total_pva']:>10.4f} {row['median_total_pva']:>11.4f} "
              f"{row['mean_n_actions']:>8.1f} {arrow:>15} {row['net_x_progress']:>7.3f}")

# ==============================================================================
# LOAD DATA
# ==============================================================================

section("LOADING RESAMPLED POSSESSIONS")

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Run 01_extract_possessions.py first. Missing: {input_path}")

df = pd.read_csv(input_path)
print(f"  Loaded {len(df):,} possessions across {df['league'].nunique()} leagues")

feature_cols = [f'p{i}_{c}' for i in range(N_RESAMPLE) for c in ('x', 'y')]
print(f"  Feature dimensionality: {len(feature_cols)}D")

# ==============================================================================
# SPLIT BY STARTING THIRD (STRATIFICATION)
# ==============================================================================

section("STRATIFICATION BY STARTING THIRD")

strata_counts = df['start_third'].value_counts()
for stratum in ['Defensive', 'Midfield', 'Attacking']:
    n = strata_counts.get(stratum, 0)
    print(f"  {stratum:>12}: {n:>7,} possessions ({n/len(df)*100:.1f}%)")

df['stratum'] = df['start_third']

# ==============================================================================
# PER-STRATUM K SELECTION + CLUSTERING (K-MEDOIDS)
# ==============================================================================

all_k_metrics = []
all_stratum_meta = []

df['cluster_id'] = ''
centroids_dict = {}

for stratum_name in ['Defensive', 'Midfield', 'Attacking']:
    config = STRATA_CONFIG[stratum_name]
    prefix = config['prefix']

    stratum_mask = df['stratum'] == stratum_name
    stratum_df = df[stratum_mask]
    stratum_idx = stratum_df.index
    n_stratum = len(stratum_df)

    X_stratum = stratum_df[feature_cols].values.astype(np.float64)

    k_anchors, route_counts = route_scout(X_stratum, N_RESAMPLE)
    k_ceiling = config['k_ceiling']
    k_lo = max(K_FLOOR, min(k_anchors[0.5], k_ceiling))
    k_hi = min(max(k_anchors.values()), k_ceiling)
    k_lo = min(k_lo, k_hi)
    k_range = list(range(k_lo, k_hi + 1))

    section(f"STRATUM: {stratum_name} (prefix={prefix}, K sweep {k_lo}..{k_hi})")
    print(f"  {n_stratum:,} possessions, feature matrix {X_stratum.shape}")

    print(f"\n  Route scouting ({len(route_counts)} unique grid routes):")
    for t_pct, k_a in sorted(k_anchors.items()):
        print(f"    >={t_pct}% freq: {k_a} routes")
    print(f"    Top 5 routes:")
    for route, count in route_counts.head(5).items():
        print(f"      {route}: {count:,} ({count / n_stratum * 100:.1f}%)")
    print(f"    K sweep range: {k_lo}..{k_hi}")

    route_counts.to_csv(
        os.path.join(output_dir, f'route_scouting_{stratum_name.lower()}.csv'),
        header=['count'],
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_stratum)

    print(f"\n  Computing pairwise distance matrix...")
    D = squareform(pdist(X_scaled, 'euclidean'))
    print(f"  Distance matrix: {D.shape}, {D.nbytes / 1e6:.0f} MB")

    print(f"\n  K sweep ({k_range[0]}..{k_range[-1]}) via FasterPAM:")
    metrics_df = run_k_sweep(D, X_scaled, k_range, stratum_name)
    all_k_metrics.append(metrics_df)

    best_idx = metrics_df['silhouette'].idxmax()
    best_k = int(metrics_df.loc[best_idx, 'k'])
    best_sil = metrics_df.loc[best_idx, 'silhouette']
    ch_best_k = int(metrics_df.loc[metrics_df['calinski_harabasz'].idxmax(), 'k'])

    print(f"\n  Silhouette winner: k = {best_k} (score = {best_sil:.4f})")
    print(f"  CH winner:         k = {ch_best_k}")
    if best_k == ch_best_k:
        print(f"  Both metrics agree on k = {best_k}")
    else:
        print(f"  Using silhouette winner (k = {best_k})")

    print(f"\n  K-Medoids with k = {best_k} (FasterPAM)...")
    result = kmedoids.fasterpam(D, best_k, random_state=RANDOM_STATE)
    raw_labels = result.labels
    medoid_indices = result.medoids

    sil_before = silhouette_score(
        X_scaled, raw_labels,
        sample_size=min(10000, n_stratum),
        random_state=RANDOM_STATE,
    )
    print(f"  Silhouette (before filtering): {sil_before:.4f}")
    print(f"  Medoid indices: {medoid_indices}")

    distances = np.array([D[i, medoid_indices[raw_labels[i]]] for i in range(n_stratum)])

    keep_mask = np.zeros(n_stratum, dtype=bool)
    for cid in range(best_k):
        cluster_mask = raw_labels == cid
        cluster_dists = distances[cluster_mask]
        threshold = np.percentile(cluster_dists, DISTANCE_PERCENTILE)
        keep_mask[cluster_mask] = distances[cluster_mask] <= threshold

    n_assigned = keep_mask.sum()
    n_unassigned = n_stratum - n_assigned
    print(f"  Distance filtering (keep {DISTANCE_PERCENTILE}th pct):")
    print(f"    Assigned: {n_assigned:,} ({n_assigned/n_stratum*100:.1f}%)")
    print(f"    Unassigned: {n_unassigned:,} ({n_unassigned/n_stratum*100:.1f}%)")

    if n_assigned > best_k:
        sil_after = silhouette_score(
            X_scaled[keep_mask], raw_labels[keep_mask],
            sample_size=min(10000, n_assigned),
            random_state=RANDOM_STATE,
        )
        print(f"    Silhouette (assigned only): {sil_after:.4f}")
    else:
        sil_after = sil_before

    global_labels = []
    for i, lab in enumerate(raw_labels):
        if keep_mask[i]:
            global_labels.append(f'{prefix}{lab}')
        else:
            global_labels.append('-1')

    df.loc[stratum_idx, 'cluster_id'] = global_labels

    for cid in range(best_k):
        global_id = f'{prefix}{cid}'
        medoid_idx = medoid_indices[cid]
        centroids_dict[global_id] = X_stratum[medoid_idx]

    all_stratum_meta.append({
        'stratum': stratum_name,
        'prefix': prefix,
        'method': 'kmedoids_fasterpam',
        'n_possessions': n_stratum,
        'n_unique_routes': len(route_counts),
        'k_anchor_0.25pct': k_anchors[0.25],
        'k_anchor_0.5pct':  k_anchors[0.5],
        'k_anchor_1.0pct':  k_anchors[1.0],
        'k_range': f'{k_lo}-{k_hi}',
        'selected_k': best_k,
        'silhouette_winner_k': best_k,
        'ch_winner_k': ch_best_k,
        'silhouette_before_filter': round(sil_before, 6),
        'silhouette_after_filter': round(sil_after, 6),
        'n_assigned': int(n_assigned),
        'n_unassigned': int(n_unassigned),
        'unassigned_pct': round(n_unassigned / n_stratum * 100, 2),
    })

    print(f"\n  Cluster sizes:")
    for cid in range(best_k):
        n_in = (np.array(global_labels) == f'{prefix}{cid}').sum()
        print(f"    {prefix}{cid}: {n_in:,}")

    del D

# ==============================================================================
# K SELECTION DIAGNOSTICS (COMBINED PLOT)
# ==============================================================================

section("K SELECTION DIAGNOSTICS")

all_k_metrics_df = pd.concat(all_k_metrics, ignore_index=True)
all_k_metrics_df.to_csv(os.path.join(output_dir, 'k_selection_metrics.csv'), index=False)
print(f"  Saved: k_selection_metrics.csv")

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
strata_order = ['Defensive', 'Midfield', 'Attacking']
metric_cols = ['silhouette', 'calinski_harabasz', 'total_distance']
metric_labels = ['Silhouette Score', 'Calinski-Harabasz', 'Total Distance (Elbow)']
colors = ['#2166ac', '#b2182b', '#1b7837']

for row_i, stratum_name in enumerate(strata_order):
    sdata = all_k_metrics_df[all_k_metrics_df['stratum'] == stratum_name]
    meta = [m for m in all_stratum_meta if m['stratum'] == stratum_name][0]
    sel_k = meta['selected_k']

    for col_i, (mcol, mlabel, mcolor) in enumerate(zip(metric_cols, metric_labels, colors)):
        ax = axes[row_i, col_i]
        ax.plot(sdata['k'], sdata[mcol], 'o-', color=mcolor, linewidth=2, markersize=4)
        ax.axvline(x=sel_k, color='red', linestyle='--', alpha=0.7, label=f'k={sel_k}')
        ax.set_xlabel('k')
        if row_i == 0:
            ax.set_title(mlabel, fontsize=10, fontweight='bold')
        if col_i == 0:
            ax.set_ylabel(f'{stratum_name}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.set_xticks(sdata['k'].values)
        ax.tick_params(labelsize=7)

fig.suptitle('Per-Stratum K Selection — K-Medoids (Multi-Metric Panel)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'k_selection_panel.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: k_selection_panel.png")

# ==============================================================================
# PROFILE ALL CLUSTERS
# ==============================================================================

section("CLUSTER PROFILES")

assigned_df = df[df['cluster_id'] != '-1'].copy()
n_assigned_total = len(assigned_df)
n_unassigned_total = len(df) - n_assigned_total
n_clusters_total = assigned_df['cluster_id'].nunique()

print(f"  Total: {n_assigned_total:,} assigned, {n_unassigned_total:,} unassigned, "
      f"{n_clusters_total} clusters")
print(f"  (Cluster prototypes are real possessions — medoids, not averages)")

profiles_df = profile_clusters(assigned_df, 'cluster_id', centroids_dict, N_RESAMPLE)
print_profiles(profiles_df, n_assigned_total)

for stratum_name in strata_order:
    stratum_profiles = profiles_df[profiles_df['stratum'] == stratum_name]
    n_cls = len(stratum_profiles)
    n_poss = stratum_profiles['n_possessions'].sum()
    print(f"\n  {stratum_name}: {n_cls} clusters, {n_poss:,} assigned possessions")
    pva_range = stratum_profiles['mean_total_pva']
    print(f"    PVA range: [{pva_range.min():.4f}, {pva_range.max():.4f}]")

# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================

section("SAVING OUTPUTS")

clustered_path = os.path.join(output_dir, 'possessions_clustered.csv')
df.drop(columns=['stratum'], errors='ignore').to_csv(clustered_path, index=False)
print(f"  Saved: {clustered_path}")
print(f"    {len(df):,} total possessions ({n_assigned_total:,} assigned, "
      f"{n_unassigned_total:,} unassigned)")

profiles_path = os.path.join(output_dir, 'cluster_profiles.csv')
profiles_df.to_csv(profiles_path, index=False)
print(f"  Saved: {profiles_path}")
print(f"    {n_clusters_total} clusters profiled (medoid-based prototypes)")

stratum_meta_df = pd.DataFrame(all_stratum_meta)
stratum_meta_df.to_csv(os.path.join(output_dir, 'clustering_metadata.csv'), index=False)
print(f"  Saved: clustering_metadata.csv")

# Summary
print("\n" + "=" * 80)
print("  PHASE 2 COMPLETE — Stratified K-Medoids Partitioning (FasterPAM)")
print("=" * 80)
print(f"  Distance percentile: {DISTANCE_PERCENTILE}")
for meta in all_stratum_meta:
    print(f"  {meta['stratum']:>12}: k={meta['selected_k']}, "
          f"sil={meta['silhouette_after_filter']:.4f}, "
          f"{meta['n_assigned']:,} assigned, "
          f"{meta['unassigned_pct']:.1f}% unassigned")
print(f"  {'TOTAL':>12}: {n_clusters_total} clusters, "
      f"{n_assigned_total:,} assigned, "
      f"{n_unassigned_total:,} unassigned ({n_unassigned_total/len(df)*100:.1f}%)")
print("=" * 80)
