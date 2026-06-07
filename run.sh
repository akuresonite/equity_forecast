#!/usr/bin/env bash
# Thin wrapper: activate the shared SSD venv and run a python script with a
# headless matplotlib backend. Usage:  ./run.sh eda.py   ./run.sh tier0_baselines/run_baselines.py
set -euo pipefail
VENV="${EQUITY_FORECAST_VENV:-/mnt/ssd/equity_forecast/venv}"
if [ ! -x "$VENV/bin/python" ]; then
  echo "venv not found at $VENV — run ./setup.sh first" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export MPLBACKEND=Agg
exec python "$@"
