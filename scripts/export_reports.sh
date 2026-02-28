#!/usr/bin/env bash
# Export notebooks to HTML with outputs (charts, tables) for viewing without running code.
# Run from repo root. Requires: pipeline has been run so data/processed/ exists.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p reports
# Keep Jupyter/IPython dirs inside repo so we don't need write access to home
export JUPYTER_CONFIG_DIR="$ROOT/.jupyter_config"
export IPYTHONDIR="$ROOT/.ipython_dir"
mkdir -p "$JUPYTER_CONFIG_DIR" "$IPYTHONDIR"
JUPYTER="${JUPYTER:-}"
if [[ -z "$JUPYTER" && -x "$ROOT/.venv/bin/jupyter" ]]; then
  JUPYTER="$ROOT/.venv/bin/jupyter"
elif [[ -z "$JUPYTER" ]]; then
  JUPYTER=jupyter
fi
cd notebooks
"$JUPYTER" nbconvert --to html --execute --ExecutePreprocessor.timeout=300 \
  --output-dir=../reports \
  01_sub3_performance_modeling.ipynb \
  02_lifetime_athlete_intelligence.ipynb
echo "Reports written to reports/"
