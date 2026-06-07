"""
Live forward forecast: refit on ALL available history and project the next
~60 business days per ticker.

  - Levels (close): RandomWalkWithDrift + AutoETS, with 80/95% prediction
    intervals (AutoETS).
  - Returns (log_return): the tier2 LightGBM, integrated to an implied price
    path  price_t = last_close · exp(cumsum(r̂)).

Writes:
  <DATA>/live_forecasts.parquet     long-format forecasts (all models)
  assets/forecasts/<TICKER>.png     per-stock chart (history tail + forecasts)
  assets/forecast_summary.csv       ranked next-h expected % move
Usage:  ./run.sh live_forecast.py
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
from statsforecast.models import RandomWalkWithDrift, AutoETS
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import data       # noqa: E402
import splits     # noqa: E402
import thermal    # noqa: E402

H = 60
LEVELS = [80, 95]
HIST_TAIL = 252
ASSETS = ROOT / 'assets' / 'forecasts'
LAGS = [1, 5, 10, 20, 60, 120, 252]
ROLL = [5, 20, 60]
STATIC = ['sector', 'listing_age_years']

RWD = RandomWalkWithDrift()
ETS = AutoETS(season_length=5)


def _pd(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas().assign(ds=lambda d: pd.to_datetime(d['ds']))


def _expand_business_days(df_pd: pd.DataFrame) -> pd.DataFrame:
    df_pd = df_pd.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    bounds = df_pd.groupby('unique_id')['ds'].agg(['min', 'max']).reset_index()
    pieces = [pd.DataFrame({'unique_id': uid, 'ds': pd.bdate_range(mn, mx)})
              for uid, mn, mx in bounds.itertuples(index=False)
              if len(pd.bdate_range(mn, mx))]
    full = pd.concat(pieces, ignore_index=True)
    out = full.merge(df_pd, on=['unique_id', 'ds'], how='left').sort_values(['unique_id', 'ds'])
    out['y'] = out.groupby('unique_id')['y'].ffill()
    return out.reset_index(drop=True)


def level_forecasts(close_pl: pl.DataFrame) -> pd.DataFrame:
    train = _pd(close_pl)[['unique_id', 'ds', 'y']]
    sf = StatsForecast(models=[RWD, ETS], freq='B', n_jobs=-1)
    fc = sf.forecast(df=train, h=H, level=LEVELS)
    return fc.reset_index() if 'unique_id' not in fc.columns else fc


def return_price_path(ret_pl: pl.DataFrame, last_close: pd.Series, static_df: pd.DataFrame) -> pd.DataFrame:
    train = _expand_business_days(_pd(ret_pl)).merge(static_df, on='unique_id', how='left')
    train = train.dropna(subset=['y'])
    train['sector'] = train['sector'].astype('category')
    fcst = MLForecast(
        models={'lgb_ret': __import__('lightgbm').LGBMRegressor(
            n_estimators=1500, learning_rate=0.05, max_depth=8, num_leaves=63,
            min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.9,
            bagging_freq=5, objective='regression', verbosity=-1, n_jobs=-1, random_state=42)},
        freq='B', lags=LAGS,
        lag_transforms={1: [RollingMean(w) for w in ROLL] + [RollingStd(w) for w in ROLL]},
        num_threads=4,
    )
    fcst.fit(train, static_features=STATIC, keep_last_n=max(LAGS) + 60)
    fc = fcst.predict(h=H)
    fc = fc.sort_values(['unique_id', 'ds'])
    fc['cum'] = fc.groupby('unique_id')['lgb_ret'].cumsum()
    fc = fc.merge(last_close.rename('last_close'), left_on='unique_id', right_index=True, how='left')
    fc['lgb_price'] = fc['last_close'] * np.exp(fc['cum'])
    return fc[['unique_id', 'ds', 'lgb_ret', 'lgb_price']]


def chart(uid, hist, lvl, ret, meta):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist['ds'], hist['y'], color='#222', lw=1.3, label='history (close)')
    g = lvl[lvl['unique_id'] == uid].sort_values('ds')
    if not g.empty:
        ax.plot(g['ds'], g['RWD'], color='#1f77b4', lw=1.6, ls='--', label='RandomWalkWithDrift')
        ax.plot(g['ds'], g['AutoETS'], color='#d62728', lw=1.6, label='AutoETS')
        if 'AutoETS-lo-80' in g.columns:
            ax.fill_between(g['ds'], g['AutoETS-lo-80'], g['AutoETS-hi-80'],
                            color='#d62728', alpha=0.15, label='AutoETS 80%')
    r = ret[ret['unique_id'] == uid].sort_values('ds')
    if not r.empty:
        ax.plot(r['ds'], r['lgb_price'], color='#2ca02c', lw=1.6, label='LightGBM (returns→price)')
    ttl = f"{uid}"
    if meta is not None:
        ttl += f"  ·  {meta.get('company_name','')}  ·  {meta.get('sector','')}"
    ax.set_title(ttl, fontsize=10)
    ax.set_ylabel('price (INR, adj)'); ax.legend(fontsize=8, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(ASSETS / f'{uid.replace("/", "_")}.png', dpi=120)
    plt.close(fig)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    panel = data.load_panel()
    ids = splits._eligible_ids(panel)
    close_pl = splits._slice(panel, ids, 'close')
    ret_pl = splits._slice(panel, ids, 'log_return')
    static_df = data.load_static().to_pandas()
    static_df['sector'] = static_df['sector'].fillna('unknown').astype(str)
    static_df['listing_age_years'] = static_df['listing_age_years'].fillna(0.0).astype(float)
    meta_by_id = static_df.set_index('unique_id').to_dict('index')

    print(f'[live] {len(ids)} tickers — fitting level models…', flush=True)
    lvl = level_forecasts(close_pl)
    thermal.cool_if_hot(tag='live-ret')
    print('[live] fitting LightGBM returns model…', flush=True)
    close_hist = _pd(close_pl)
    last_close = close_hist.sort_values('ds').groupby('unique_id')['y'].last()
    ret = return_price_path(ret_pl, last_close, static_df[['unique_id'] + STATIC])

    # persist combined long-format forecasts
    out_long = lvl.merge(ret, on=['unique_id', 'ds'], how='outer').sort_values(['unique_id', 'ds'])
    pl.from_pandas(out_long).write_parquet(data.DATA_DIR / 'live_forecasts.parquet')

    # summary: expected % move over the full horizon
    summ = []
    for uid in sorted(close_hist['unique_id'].unique()):
        lc = last_close.get(uid, np.nan)
        g = lvl[lvl['unique_id'] == uid].sort_values('ds')
        r = ret[ret['unique_id'] == uid].sort_values('ds')
        ets_end = g['AutoETS'].iloc[-1] if not g.empty else np.nan
        lgb_end = r['lgb_price'].iloc[-1] if not r.empty else np.nan
        summ.append({'unique_id': uid, 'last_close': round(lc, 2),
                     'ets_h60': round(ets_end, 2), 'ets_move_%': round((ets_end / lc - 1) * 100, 2),
                     'lgb_h60': round(lgb_end, 2), 'lgb_move_%': round((lgb_end / lc - 1) * 100, 2)})
    summary = pd.DataFrame(summ).sort_values('ets_move_%', ascending=False)
    summary.to_csv(ASSETS.parent / 'forecast_summary.csv', index=False)

    print('[live] rendering charts…', flush=True)
    for uid in sorted(close_hist['unique_id'].unique()):
        h = close_hist[close_hist['unique_id'] == uid].sort_values('ds').tail(HIST_TAIL)
        chart(uid, h, lvl, ret, meta_by_id.get(uid))
    print(f'[live] wrote {summary.shape[0]} charts → {ASSETS}')
    print('[live] top 5 expected movers (AutoETS, h=60):')
    print(summary.head(5).to_string(index=False))
    print('[live] bottom 5:')
    print(summary.tail(5).to_string(index=False))


if __name__ == '__main__':
    main()
