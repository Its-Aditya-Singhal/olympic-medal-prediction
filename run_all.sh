#!/usr/bin/env bash
# Rebuild everything from the raw Kaggle CSVs: data -> models -> figures.
set -euo pipefail
cd "$(dirname "$0")/src"
PY=${PYTHON:-../.venv/bin/python}

if [[ "${1:-}" == "--refresh" ]]; then
  $PY wdi.py              # World Bank indicators (network)
  $PY fetch_tokyo2020.py  # Tokyo 2020 medal table + team sizes (network)
fi
$PY build_dataset.py
$PY train.py
$PY experiments.py
$PY paper_audit.py
$PY plots.py
