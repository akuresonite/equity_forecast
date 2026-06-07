"""
Tier 1 — classical per-series statistical models via statsforecast:
AutoARIMA, AutoETS, AutoTheta, AutoCES (all season_length=5).

Feasible here: 50 series × ~25yr runs in minutes (the sibling nav_forecast
abandoned this tier only at 3,000+ series). Same protocol as tier0: both
targets, 6-fold walk-forward CV, horizons 5/20/60.

Output: <OUTPUT>/metrics_<target>.csv
Usage:  ./run.sh tier1_classical/run_classical.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import polars as pl

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, AutoCES, Naive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import splits          # noqa: E402
import metrics         # noqa: E402
import thermal         # noqa: E402

OUTPUT_DIR = Path(os.environ.get(
    'EQUITY_FORECAST_OUTPUT_T1', '/mnt/ssd/equity_forecast/tier1_classical/output'))
HORIZON   = splits.TEST_DAYS
PREDICT_H = HORIZON + 15
SEASON    = 5
TARGETS   = ['close', 'log_return']

MODELS = [
    AutoARIMA(season_length=SEASON),
    AutoETS(season_length=SEASON),
    AutoTheta(season_length=SEASON),
    AutoCES(season_length=SEASON),
]
MODEL_COLS = [m.alias for m in MODELS]


def _pd(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas().assign(ds=lambda d: pd.to_datetime(d['ds']))


def forecast_eval(train_pl: pl.DataFrame, test_pl: pl.DataFrame) -> pd.DataFrame:
    train = _pd(train_pl)[['unique_id', 'ds', 'y']]
    test  = _pd(test_pl)[['unique_id', 'ds', 'y']]
    # fallback_model: if a per-series fit fails (e.g. AutoCES "no model able to
    # be fitted" on noisy returns), degrade that cell to Naive instead of crashing.
    sf = StatsForecast(models=MODELS, freq='B', n_jobs=-1, fallback_model=Naive())
    fc = sf.forecast(df=train, h=PREDICT_H)        # one-shot fit+predict
    if 'unique_id' not in fc.columns:
        fc = fc.reset_index()
    merged = fc.merge(test, on=['unique_id', 'ds'], how='inner')
    if merged.empty:
        return pd.DataFrame()
    return metrics.evaluate_at_horizons(merged, train, MODEL_COLS)


def run_target(target: str) -> None:
    print(f'\n=== tier1  target={target} ===', flush=True)
    rows = []
    for fold, tr, te, end in splits.walk_forward_folds(n=6, target=target):
        thermal.cool_if_hot(tag=f'tier1-{target[:3]}-f{fold}')
        t0 = time.time()
        mm = forecast_eval(tr, te)
        if not mm.empty:
            mm['fold'] = fold; mm['test_end'] = str(end); rows.append(mm)
        print(f'  fold {fold}: test_end={end}  ids={te["unique_id"].n_unique()}  '
              f'{time.time() - t0:.1f}s  cpu={thermal.cpu_c():.1f}°C', flush=True)

    if not rows:
        print('  no metrics produced'); return
    out = pd.concat(rows, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_DIR / f'metrics_{target}.csv'
    out.to_csv(fp, index=False)
    print(f'  written → {fp}')

    agg = (out[out['horizon'] == 60].groupby('metric')[MODEL_COLS].mean())
    print(f'  cross-fold mean @h=60:\n{agg.round(4).to_string()}')


def main() -> None:
    # Optional CLI target(s) so a single target can be (re)run without redoing the
    # expensive other one — e.g. `run_classical.py log_return`.
    targets = [t for t in sys.argv[1:] if t in TARGETS] or TARGETS
    print(f'[tier1] start cpu={thermal.cpu_c():.1f}°C  targets={targets}', flush=True)
    for t in targets:
        run_target(t)
    print(f'[tier1] done cpu={thermal.cpu_c():.1f}°C', flush=True)


if __name__ == '__main__':
    main()
