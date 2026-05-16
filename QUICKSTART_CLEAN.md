# Possession Value and Spatial Allocation in Women's Football

Open-source implementation of the Possession Value Added (PVA) model and the Zone Value Added (ZVA) and Routing Value Added (RVA) metrics for the FA Women's Super League (2018-19 and 2020-21).

Paper: `bfalzarano_practicum26.pdf`.

## Data

The scripts expect StatsBomb open event data at `open-data/data/`. Clone it from the StatsBomb repository and move the `data/` directory into the corresponding location in this project:

```bash
git clone https://github.com/statsbomb/open-data.git
```

## Project Structure

```
.
├── bfalzarano_practicum26.pdf
└── open-data/
    ├── notebooks/
    │   ├── pva_model/
    │   ├── zone_analysis/
    │   └── possession_patterning_analysis/
    ├── results/
    └── visualizations/
```

## Running the Pipeline

### PVA model

```bash
python open-data/notebooks/pva_model/pva_model_generic.py
python open-data/notebooks/pva_model/pva_weight_calibration.py
python open-data/notebooks/pva_model/metric_benchmark.py
python open-data/notebooks/pva_model/presentation_visuals.py
```

### Zone analysis (ZVA)

```bash
python open-data/notebooks/zone_analysis/statistical_testing/zone_statistical_testing.py
python open-data/notebooks/zone_analysis/visualization/zone_visualizations_grid.py
python open-data/notebooks/zone_analysis/visualization/zone_visualizations_kde.py
```

### Possession patterning (RVA)

```bash
python open-data/notebooks/possession_patterning_analysis/statistical_testing/01_extract_possessions.py
python open-data/notebooks/possession_patterning_analysis/statistical_testing/02_cluster_patterns.py
python open-data/notebooks/possession_patterning_analysis/statistical_testing/03_pattern_utilization.py
python open-data/notebooks/possession_patterning_analysis/visualization/04_visualize_clusters.py
python open-data/notebooks/possession_patterning_analysis/visualization/05_team_visualizations.py
```

## Visualizations

Generated figures are written to `open-data/visualizations/`:

- `zone_analysis/fa_wsl_2018-19/GRID/` and `.../KDE/` — zone-level figures for the 2018-19 season in grid and KDE form
- `zone_analysis/fa_wsl_2020-21/GRID/` and `.../KDE/` — same for 2020-21
- `possession_patterning_analysis/` — cluster centroid routes and per-team pattern grids

The xT grid and metric benchmark figures used in the paper are in `open-data/results/presentation/`.

## Citation

Falzarano, B. (2026). Using Possession Value to Evaluate Zone Utilization and Spatial Routing in Women's Football. Applied Data Science Capstone, Wesleyan University.

Event data: StatsBomb Open Data (https://github.com/statsbomb/open-data).
