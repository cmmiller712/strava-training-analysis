# Strava Training Analysis

Reproducible analysis of Strava training data: volume trends, consistency, and performance indicators (focused on sub-3 marathon).

**Why this project** — I'm a two-time Ironman and 100k ultramarathon finisher, currently training for a sub-3 marathon. I work in data engineering and am obsessed with my Strava stats. I built this to turn my export into something useful: a clean pipeline from raw activities + FIT files to training metrics, and notebooks that answer both serious questions (am I on track for sub-3? how's my ramp rate?) and fun ones (what's my rarest activity type? do harder workouts get more kudos?). Everything is reproducible, privacy-conscious (no GPS or personal data in outputs), and designed so the insights are viewable without running a single cell.

## View the insights (no setup required)

Rendered reports with all charts and tables—open in any browser:

| Report | What it shows |
|--------|----------------|
| **[Sub-3 performance modeling](reports/01_sub3_performance_modeling.html)** | Training volume, pace zones, consistency, marathon-pace miles from FIT streams. |
| **[Lifetime athlete intelligence](reports/02_lifetime_athlete_intelligence.html)** | Activity mix, most/rarest types, kudos by type, run volume over time. |

To regenerate reports after re-running the pipeline: `./scripts/export_reports.sh`

## Privacy

Raw Strava export data is not committed. GPS-related fields (coordinates, routes) are excluded from any published datasets.

## Project layout

| Path | Description |
|------|-------------|
| `data/raw/` | Private Strava export (git-ignored). Place unzipped export in `data/raw/strava_export/`. |
| `data/processed/` | Sanitized outputs (git-ignored): runs, weekly model, lifetime activities, summaries. |
| `reports/` | Rendered HTML of the notebooks (charts + tables); safe to share, no data. |
| `src/` | Data loading, feature engineering, and dataset build. |
| `notebooks/` | Analysis and charts (Jupyter). |
| `scripts/` | Helpers (e.g. export notebooks to `reports/`). |

**Pipeline:**

- **`src/build_datasets.py`** — Main pipeline. Reads Strava export + FIT files, builds run-level and weekly metrics (pace, MP miles from stream, AES, ramp rate, long run, etc.), writes `runs_enriched.csv` and `weekly_model.csv`. Used by the sub-3 notebook.
- **`src/build_lifetime_dataset.py`** — Lifetime pipeline. Builds `lifetime_activities.csv` from all activities ever (no date filter). Used by the lifetime “fun” notebook (activity mix, kudos, etc.).
- **`src/clean_data.py`** — Optional. Uses the same loader (`io_strava`) to produce `weekly_summary.csv` and `monthly_summary.csv` (all activities, no run filter).

Config (build start date, goal marathon pace, ramp threshold) lives in **`src/config.py`**.

## How to run

1. Place your Strava export in `data/raw/strava_export/` (include `activities.csv` and, for MP stream miles, the FIT files in the export).
2. **Build main datasets (required for notebooks):**
   ```bash
   python src/build_datasets.py
   ```
   Outputs: `data/processed/runs_enriched.csv`, `data/processed/weekly_model.csv`.
3. **Build lifetime dataset (for notebook 02):**
   ```bash
   python src/build_lifetime_dataset.py
   ```
   Outputs: `data/processed/lifetime_activities.csv`.
4. **Optional** — Weekly/monthly activity summaries (all sports):
   ```bash
   python src/clean_data.py
   ```
5. Open notebooks from `notebooks/` (paths assume you run from repo root or from `notebooks/`). Notebook `02_lifetime_athlete_intelligence.ipynb` uses `lifetime_activities.csv`.
6. **Optional** — Export notebooks to HTML for sharing (no code run required): `./scripts/export_reports.sh` → outputs in `reports/`.

## Requirements

- **Minimal (pipeline only):** `pandas`, `numpy`, `fitparse`
- **With notebooks:** install from `requirements.txt` (adds matplotlib, Jupyter, etc.)

```bash
pip install -r requirements.txt
```
