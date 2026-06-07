"""
Tier 0 — trivial baselines: Naive, SeasonalNaive(5), RandomWalkWithDrift,
HistoricAverage, WindowAverage(20).

Run for BOTH targets (close, log_return), single split + 6-fold walk-forward CV,
evaluated at horizons 5/20/60 trading days (sliced from one 60-step forecast).

Output: <OUTPUT>/metrics_<target>.csv   (per-series × fold × horizon × model)
Usage:  ./run.sh tier0_baselines/run_baselines.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import polars as pl

from statsforecast import StatsForecast
from statsforecast.models import (
    Naive, SeasonalNaive, RandomWalkWithDrift, HistoricAverage, WindowAverage,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import splits          # noqa: E402
import metrics         # noqa: E402
import thermal         # noqa: E402

OUTPUT_DIR = Path(os.environ.get(
    'EQUITY_FORECAST_OUTPUT_T0', '/mnt/ssd/equity_forecast/tier0_baselines/output'))
HORIZON   = splits.TEST_DAYS
PREDICT_H = HORIZON + 15          # buffer so 60 trading days are covered on a B-day grid
TARGETS   = ['close', 'log_return']

MODELS = [
    Naive(),
    SeasonalNaive(season_length=5),
    RandomWalkWithDrift(),
    HistoricAverage(),
    WindowAverage(window_size=20),
]
MODEL_COLS = [m.alias for m in MODELS]


def _pd(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas().assign(ds=lambda d: pd.to_datetime(d['ds']))


def forecast_eval(train_pl: pl.DataFrame, test_pl: pl.DataFrame) -> pd.DataFrame:
    train = _pd(train_pl)[['unique_id', 'ds', 'y']]
    test  = _pd(test_pl)[['unique_id', 'ds', 'y']]
    sf = StatsForecast(models=MODELS, freq='B', n_jobs=-1)
    sf.fit(df=train)
    fc = sf.predict(h=PREDICT_H)
    if 'unique_id' not in fc.columns:
        fc = fc.reset_index()
    merged = fc.merge(test, on=['unique_id', 'ds'], how='inner')
    if merged.empty:
        return pd.DataFrame()
    return metrics.evaluate_at_horizons(merged, train, MODEL_COLS)


def run_target(target: str) -> None:
    print(f'\n=== tier0  target={target} ===', flush=True)
    rows = []

    s = splits.single_split(target=target)
    tv = pl.concat([s.train, s.val]).sort(['unique_id', 'ds'])
    m = forecast_eval(tv, s.test)
    if not m.empty:
        m['fold'] = 'single'; m['test_end'] = str(s.test_end_incl); rows.append(m)

    for fold, tr, te, end in splits.walk_forward_folds(n=6, target=target):
        thermal.cool_if_hot(tag=f'tier0-{target[:3]}-f{fold}')
        mm = forecast_eval(tr, te)
        if not mm.empty:
            mm['fold'] = fold; mm['test_end'] = str(end); rows.append(mm)
        print(f'  fold {fold}: test_end={end}  ids={te["unique_id"].n_unique()}', flush=True)

    if not rows:
        print('  no metrics produced'); return
    out = pd.concat(rows, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_DIR / f'metrics_{target}.csv'
    out.to_csv(fp, index=False)
    print(f'  written → {fp}')

    wf = out[out['fold'] != 'single']
    agg = (wf[wf['horizon'] == 60].groupby('metric')[MODEL_COLS].mean())
    print(f'  cross-fold mean @h=60:\n{agg.round(4).to_string()}')


def main() -> None:
    print(f'[tier0] start cpu={thermal.cpu_c():.1f}°C', flush=True)
    for t in TARGETS:
        run_target(t)
    print(f'[tier0] done cpu={thermal.cpu_c():.1f}°C', flush=True)


if __name__ == '__main__':
    main()
