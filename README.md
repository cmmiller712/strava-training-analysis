# Strava Training Analysis

Reproducible analysis of my Strava training data focused on volume trends, consistency, and performance indicators.

## Privacy
Raw Strava export data is not committed to this repo. GPS-related fields (coordinates, routes) are excluded from any published datasets.

## Project layout
- data/raw: private Strava export (ignored by git)
- data/processed: sanitized datasets safe to share
- src: cleaning + feature engineering code
- notebooks: analysis and charts

## How to run
1. Place Strava export in `data/raw/strava_export/`
2. Run: `python3 src/clean_data.py`
3. Use outputs in `data/processed/`