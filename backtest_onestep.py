"""
1-STEP-AHEAD (rolling, re-anchored daily) backtest — the forecast that actually
TRACKS the actuals. For each day t in the last 60 held-out trading days, predict
y_t from the real data up to t-1, then re-anchor on the realized y_t and step
forward. This is the honest counterpart to the multi-step (h=60, blind) overlay in
backtest_plot.py — same data, different question:

    multi-step  : "given today, predict the next 60 days"      → flat, ~4% MAPE
    one-step     : "given yesterday, predict today (roll daily)" → tracks, ~1-2% MAPE

IMPORTANT honesty check, reported per model: 1-step "tracking" is mostly "predict ≈
yesterday's price". The real test of skill is **directional accuracy** — does the
predicted move have the right sign? For a random walk this is ≈50% (a coin flip),
which is what we find. Tracking ≠ predictive skill.

Writes assets/backtest_1step/<TICKER>.png + assets/backtest_1step_errors.csv
Usage: ./run.sh backtest_onestep.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import polars as pl

import lightgbm as lgb
from statsforecast import StatsForecast
from statsforecast.models import Naive, AutoETS
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import data       # noqa: E402
import splits     # noqa: E402
import mlfit       # noqa: E402

H = splits.TEST_DAYS
ASSETS = ROOT / 'assets' / 'backtest_1step'
STATIC = mlfit.STATIC


def _pd(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas().assign(ds=lambda d: pd.to_datetime(d['ds']))


def _expand_business_days(df_pd: pd.DataFrame) -> pd.DataFrame:
    df_pd = df_pd.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    bounds = df_pd.groupby('unique_id')['ds'].agg(['min', 'max']).reset_index()
    pieces = [pd.DataFrame({'unique_id': uid, 'ds': pd.bdate_range(mn, mx)})
              for uid, mn, mx in bounds.itertuples(index=False) if len(pd.bdate_range(mn, mx))]
    out = (pd.concat(pieces, ignore_index=True).merge(df_pd, on=['unique_id', 'ds'], how='left')
           .sort_values(['unique_id', 'ds']))
    out['y'] = out.groupby('unique_id')['y'].ffill()
    return out.reset_index(drop=True)


def mape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    m = (a != 0) & np.isfinite(f)
    return float(np.mean(np.abs((a[m] - f[m]) / a[m])) * 100) if m.any() else np.nan


def dir_acc(prev, act, pred):
    """Directional accuracy: did the predicted move match the realized move's sign?"""
    prev, act, pred = (np.asarray(x, float) for x in (prev, act, pred))
    a, p = np.sign(act - prev), np.sign(pred - prev)
    m = (a != 0) & np.isfinite(p) & (p != 0)
    return float((p[m] == a[m]).mean() * 100) if m.any() else np.nan


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    panel = data.load_panel()
    ids = splits._eligible_ids(panel)
    close = splits._slice(panel, ids, 'close')
    static_df = data.load_static().to_pandas()
    static_df['sector'] = static_df['sector'].fillna('unknown').astype(str)
    static_df['listing_age_years'] = static_df['listing_age_years'].fillna(0.0).astype(float)
    meta = static_df.set_index('unique_id').to_dict('index')

    full = _pd(close)[['unique_id', 'ds', 'y']]

    print('[1step] statsforecast rolling 1-step (Naive, RWD, AutoETS), refit=False…', flush=True)
    sf = StatsForecast(models=[Naive(), AutoETS(season_length=5)],
                       freq='B', n_jobs=-1, fallback_model=Naive())
    cv = sf.cross_validation(df=full, h=1, step_size=1, n_windows=H, refit=False)
    cv = cv.reset_index() if 'unique_id' not in cv.columns else cv

    print('[1step] mlforecast rolling 1-step (LightGBM levels), refit=False…', flush=True)
    fullm = (_expand_business_days(full).merge(static_df[['unique_id'] + STATIC],
             on='unique_id', how='left').dropna(subset=['y']))
    fullm['sector'] = fullm['sector'].astype('category')
    mlf = MLForecast(models={'LightGBM': lgb.LGBMRegressor(n_estimators=400, **mlfit.PARAMS)},
                     freq='B', lags=mlfit.LAGS, lag_transforms=mlfit.LAG_TF,
                     target_transforms=[Differences([1])])
    cvm = mlf.cross_validation(df=fullm, h=1, step_size=1, n_windows=H, refit=False,
                               static_features=STATIC)
    cvm = cvm.reset_index() if 'unique_id' not in cvm.columns else cvm

    models = ['Naive', 'AutoETS', 'LightGBM']
    cv = cv.merge(cvm[['unique_id', 'ds', 'LightGBM']], on=['unique_id', 'ds'], how='left')
    cv = cv.sort_values(['unique_id', 'ds'])
    cv['y_prev'] = cv.groupby('unique_id')['y'].shift(1)

    rows = []
    print('[1step] rendering charts…', flush=True)
    for uid in sorted(cv['unique_id'].unique()):
        g = cv[cv['unique_id'] == uid].sort_values('ds')
        if g['y'].notna().sum() < 5:
            continue
        rows.append({'unique_id': uid, **{f'{m}_mape': mape(g['y'], g[m]) for m in models},
                     'LightGBM_dir_acc': dir_acc(g['y_prev'], g['y'], g['LightGBM']),
                     'AutoETS_dir_acc': dir_acc(g['y_prev'], g['y'], g['AutoETS'])})
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(g['ds'], g['y'], color='#000', lw=2.2, marker='.', ms=4, label='ACTUAL', zorder=5)
        for m, c in zip(['Naive', 'AutoETS', 'LightGBM'], ['#1f77b4', '#d62728', '#2ca02c']):
            ax.plot(g['ds'], g[m], color=c, lw=1.2, alpha=0.9,
                    label=f"{m} 1-step ({mape(g['y'], g[m]):.1f}% MAPE)")
        mt = meta.get(uid, {})
        ax.set_title(f"{uid} · {mt.get('company_name','')} · {mt.get('sector','')}  — "
                     f"1-step-ahead (rolling) backtest", fontsize=10)
        ax.set_ylabel('price (INR, adj)'); ax.legend(fontsize=8, loc='best')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(ASSETS / f'{uid.replace("/", "_")}.png', dpi=120); plt.close(fig)

    err = pd.DataFrame(rows)
    err.to_csv(ASSETS.parent / 'backtest_1step_errors.csv', index=False)
    print(f'[1step] wrote {len(rows)} charts → {ASSETS}')
    print('\n[1step] mean 1-step MAPE over the held-out window (%):')
    print(err[[f'{m}_mape' for m in models]].mean().round(2).to_string())
    print('\n[1step] mean DIRECTIONAL accuracy (%, ~50 = coin-flip = no timing skill):')
    print(err[['LightGBM_dir_acc', 'AutoETS_dir_acc']].mean().round(1).to_string())


if __name__ == '__main__':
    main()
