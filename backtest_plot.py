"""
Backtest overlay: same chart style as live_forecast.py, but on the most recent
HELD-OUT 60-day window — so model forecasts can be compared against the realized
actuals. Trains on everything before the window, forecasts it, overlays truth.

Models (one representative per tier): RandomWalkWithDrift (t0) + AutoETS (t1, with
80% band) on the price level, and LightGBM on returns integrated to a price path (t2).

Writes:
  assets/backtest/<TICKER>.png       per-stock actual-vs-forecast chart
  assets/backtest_errors.csv          per-stock per-model MAPE over the window
Usage:  ./run.sh backtest_plot.py
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

from statsforecast import StatsForecast
from statsforecast.models import Naive, RandomWalkWithDrift, AutoETS

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import data       # noqa: E402
import splits     # noqa: E402
import mlfit      # noqa: E402

H = splits.TEST_DAYS          # 60-day held-out window
PREDICT_H = H + 15
HIST_TAIL = 150
LEVELS = [80]
ASSETS = ROOT / 'assets' / 'backtest'
LAGS = [1, 5, 10, 20, 60, 120, 252]
ROLL = [5, 20, 60]
STATIC = ['sector', 'listing_age_years']


def _pd(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas().assign(ds=lambda d: pd.to_datetime(d['ds']))


def _expand_business_days(df_pd: pd.DataFrame) -> pd.DataFrame:
    df_pd = df_pd.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    bounds = df_pd.groupby('unique_id')['ds'].agg(['min', 'max']).reset_index()
    pieces = [pd.DataFrame({'unique_id': uid, 'ds': pd.bdate_range(mn, mx)})
              for uid, mn, mx in bounds.itertuples(index=False) if len(pd.bdate_range(mn, mx))]
    full = pd.concat(pieces, ignore_index=True)
    out = full.merge(df_pd, on=['unique_id', 'ds'], how='left').sort_values(['unique_id', 'ds'])
    out['y'] = out.groupby('unique_id')['y'].ffill()
    return out.reset_index(drop=True)


def mape(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    m = a != 0
    return float(np.mean(np.abs((a[m] - f[m]) / a[m])) * 100) if m.any() else np.nan


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    panel = data.load_panel()
    ids = splits._eligible_ids(panel)
    close = splits._slice(panel, ids, 'close')           # unique_id, ds, y(=close)
    days = close.select(pl.col('ds').unique().sort()).to_series().to_list()
    test_start, test_end = days[-H], days[-1]
    print(f'[backtest] held-out window: {test_start} … {test_end} ({H} trading days)', flush=True)

    tr_close = close.filter(pl.col('ds') < test_start)
    te_close = _pd(close.filter((pl.col('ds') >= test_start) & (pl.col('ds') <= test_end)))

    static_df = data.load_static().to_pandas()
    static_df['sector'] = static_df['sector'].fillna('unknown').astype(str)
    static_df['listing_age_years'] = static_df['listing_age_years'].fillna(0.0).astype(float)
    meta = static_df.set_index('unique_id').to_dict('index')

    # tier0/tier1 level models with an 80% band from AutoETS
    print('[backtest] fitting level models (Naive, RWD, AutoETS)…', flush=True)
    sf = StatsForecast(models=[Naive(), RandomWalkWithDrift(), AutoETS(season_length=5)],
                       freq='B', n_jobs=-1, fallback_model=Naive())
    lvl = sf.forecast(df=_pd(tr_close)[['unique_id', 'ds', 'y']], h=PREDICT_H, level=LEVELS)
    if 'unique_id' not in lvl.columns:
        lvl = lvl.reset_index()

    # tier2 price model via mlfit: LightGBM on the price LEVEL (first-differenced) with
    # time-ordered validation + early stopping. Predicts increments (not integrated
    # returns) so it cannot compound a bias into a runaway path.
    print('[backtest] fitting LightGBM (levels) with early stopping…', flush=True)
    trp = (_expand_business_days(_pd(tr_close))
           .merge(static_df[['unique_id'] + STATIC], on='unique_id', how='left')
           .dropna(subset=['y']))
    fcst, info = mlfit.fit_lgb_es(trp, target='close')
    print(f"[backtest]   train-validation assessment: MAE={info['val_mae']:.2f} price-units · "
          f"directional acc={info['dir_acc'] * 100:.1f}% (~50% = no timing skill) · "
          f"trees={info['best_iter']} (early-stopped from 3000)", flush=True)
    rp = fcst.predict(h=PREDICT_H).rename(columns={'lgb': 'lgb_price'})

    rows = []
    hist_all = _pd(tr_close)
    print('[backtest] rendering charts…', flush=True)
    for uid in sorted(te_close['unique_id'].unique()):
        act = te_close[te_close['unique_id'] == uid].sort_values('ds')
        g = lvl[lvl['unique_id'] == uid].merge(act[['ds']], on='ds', how='inner').sort_values('ds')
        r = rp[rp['unique_id'] == uid].merge(act[['ds']], on='ds', how='inner').sort_values('ds')
        if act.empty or g.empty:
            continue
        e = {'unique_id': uid,
             'RWD': mape(act['y'], g['RWD']),
             'AutoETS': mape(act['y'], g['AutoETS']),
             'lgb': mape(act['y'].values[:len(r)], r['lgb_price']) if not r.empty else np.nan}
        rows.append(e)

        hist = hist_all[hist_all['unique_id'] == uid].sort_values('ds').tail(HIST_TAIL)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hist['ds'], hist['y'], color='#888', lw=1.0, label='history (train)')
        ax.plot(act['ds'], act['y'], color='#000', lw=2.0, marker='.', ms=3, label='ACTUAL (held-out)')
        ax.plot(g['ds'], g['RWD'], color='#1f77b4', lw=1.5, ls='--',
                label=f"RandomWalkWithDrift ({e['RWD']:.1f}% MAPE)")
        ax.plot(g['ds'], g['AutoETS'], color='#d62728', lw=1.5,
                label=f"AutoETS ({e['AutoETS']:.1f}%)")
        if 'AutoETS-lo-80' in g.columns:
            ax.fill_between(g['ds'], g['AutoETS-lo-80'], g['AutoETS-hi-80'],
                            color='#d62728', alpha=0.12, label='AutoETS 80%')
        if not r.empty:
            ax.plot(r['ds'], r['lgb_price'], color='#2ca02c', lw=1.5,
                    label=f"LightGBM levels ({e['lgb']:.1f}%)")
        ax.axvline(act['ds'].iloc[0], color='k', lw=0.8, ls=':', alpha=0.6)
        m = meta.get(uid, {})
        ax.set_title(f"{uid} · {m.get('company_name','')} · {m.get('sector','')}  — "
                     f"backtest on held-out {H}d", fontsize=10)
        ax.set_ylabel('price (INR, adj)'); ax.legend(fontsize=8, loc='best')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(ASSETS / f'{uid.replace("/", "_")}.png', dpi=120); plt.close(fig)

    err = pd.DataFrame(rows)
    err.to_csv(ASSETS.parent / 'backtest_errors.csv', index=False)
    print(f'[backtest] wrote {len(rows)} charts → {ASSETS}')
    print('[backtest] mean MAPE over the held-out window (lower = closer to actual):')
    print(err[['RWD', 'AutoETS', 'lgb']].mean().round(2).to_string())


if __name__ == '__main__':
    main()
