#!/usr/bin/env bash
# One-time environment + data setup for equity_forecast.
#   - creates a single shared venv on the SSD (keeps the SD card / git tree clean)
#   - installs the pinned forecasting stack
#   - verifies the tricky numba/statsforecast imports up front (fail fast)
#   - downloads + unzips the Kaggle dataset
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSD="${EQUITY_FORECAST_HOME:-/mnt/ssd/equity_forecast}"
VENV="$SSD/venv"
DATA="$SSD/data"
RAW="$DATA/raw"
PY="${PYTHON:-python3}"

echo "[setup] root=$ROOT  ssd=$SSD  python=$($PY --version 2>&1)"
mkdir -p "$DATA" "$RAW"

# 1. venv ------------------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  echo "[setup] creating venv at $VENV"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null

# 2. deps (pinned) ---------------------------------------------------------
echo "[setup] installing pinned requirements (a few minutes on ARM the first time)…"
pip install -r "$ROOT/requirements.txt"

# 3. verify the load-bearing imports --------------------------------------
echo "[setup] verifying forecasting stack…"
python - <<'PY'
import numba, llvmlite, statsforecast, mlforecast, utilsforecast, lightgbm, polars, pyarrow, shap
print("  OK  numba", numba.__version__, "| statsforecast", statsforecast.__version__,
      "| mlforecast", mlforecast.__version__, "| lightgbm", lightgbm.__version__,
      "| polars", polars.__version__)
PY

# 4. dataset ---------------------------------------------------------------
SLUG="kalyan197/nifty50-stocks1999-2026-daily-ohlcv-and-fundamentals"
if [ ! -f "$RAW/nifty50_historical_data.csv" ]; then
  echo "[setup] downloading Kaggle dataset → $RAW"
  kaggle datasets download -d "$SLUG" -p "$RAW" --unzip
else
  echo "[setup] dataset already present, skipping download"
fi

echo "[setup] done. Files:"
ls -la "$RAW"
echo "[setup] next:  ./run.sh data.py materialise  &&  ./run.sh eda.py"
