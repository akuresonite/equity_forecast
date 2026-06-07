"""
Tier 2 — global tree-based forecasting via mlforecast + LightGBM.

One model across all 50 tickers with lag + rolling features and leakage-safe
static covariates (sector, listing_age_years). Differences([1]) for `close`
only (returns are already stationary). SHAP on the final-fold refit.

Both targets, 6-fold walk-forward CV, horizons 5/20/60.
Output: <OUTPUT>/metrics_<target>.csv  +  shap_*_<target>.{png,csv}
Usage:  ./run.sh tier2_global_ml/run_global.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import shap

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from mlforecast.target_transforms import Differences

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data            # noqa: E402
import splits          # noqa: E402
import metrics         # noqa: E402
import thermal         # noqa: E402

OUTPUT_DIR = Path(os.environ.get(
    'EQUITY_FORECAST_OUTPUT_T2', '/mnt/ssd/equity_forecast/tier2_global_ml/output'))
HORIZON   = splits.TEST_DAYS
PREDICT_H = HORIZON + 15
TARGETS   = ['close', 'log_return']

LAGS = [1, 5, 10, 20, 60, 120, 252]
ROLL_WINDOWS = [5, 20, 60]
CAT_FEATURES = ['sector']
NUM_FEATURES = ['listing_age_years']
STATIC_FEATURES = CAT_FEATURES + NUM_FEATURES
MODEL = 'lgb'


def make_lgb():
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.05, max_depth=8, num_leaves=63,
        min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.9,
        bagging_freq=5, objective='regression', verbosity=-1, n_jobs=-1,
        random_state=42,
    )


def make_fcst(target: str) -> MLForecast:
    target_transforms = [Differences([1])] if target == 'close' else []
    lag_transforms = {1: [RollingMean(w) for w in ROLL_WINDOWS]
                       + [RollingStd(w) for w in ROLL_WINDOWS]}
    return MLForecast(models={MODEL: make_lgb()}, freq='B', lags=LAGS,
                      lag_transforms=lag_transforms,
                      target_transforms=target_transforms, num_threads=4)


def load_static() -> pd.DataFrame:
    s = data.load_static().to_pandas()
    s['sector'] = s['sector'].fillna('unknown').astype(str)
    s['listing_age_years'] = s['listing_age_years'].fillna(0.0).astype(float)
    return s[['unique_id'] + STATIC_FEATURES]


def _pd(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas().assign(ds=lambda d: pd.to_datetime(d['ds']))


def _expand_business_days(df_pd: pd.DataFrame) -> pd.DataFrame:
    """Reindex each series onto a regular B-day grid and ffill y (mlforecast
    needs a uniform freq; the NSE calendar has holiday gaps)."""
    df_pd = df_pd.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    bounds = df_pd.groupby('unique_id')['ds'].agg(['min', 'max']).reset_index()
    pieces = [pd.DataFrame({'unique_id': uid, 'ds': pd.bdate_range(mn, mx)})
              for uid, mn, mx in bounds.itertuples(index=False)
              if len(pd.bdate_range(mn, mx))]
    full = pd.concat(pieces, ignore_index=True)
    out = full.merge(df_pd, on=['unique_id', 'ds'], how='left').sort_values(['unique_id', 'ds'])
    out['y'] = out.groupby('unique_id')['y'].ffill()
    return out.reset_index(drop=True)


def fit_predict(target, train, test, static_df):
    train_pd = _expand_business_days(_pd(train)).merge(static_df, on='unique_id', how='left')
    train_pd = train_pd.dropna(subset=['y'])
    for c in CAT_FEATURES:
        train_pd[c] = train_pd[c].astype('category')
    fcst = make_fcst(target)
    fcst.fit(train_pd, static_features=STATIC_FEATURES, keep_last_n=max(LAGS) + 60)
    fc = fcst.predict(h=PREDICT_H)
    test_pd = _pd(test)[['unique_id', 'ds', 'y']]
    merged = fc.merge(test_pd, on=['unique_id', 'ds'], how='inner')
    return merged, fcst


def shap_analysis(fcst, train, static_df, target, n_sample=40000):
    train_pd = _expand_business_days(_pd(train)).merge(static_df, on='unique_id', how='left')
    train_pd = train_pd.dropna(subset=['y'])
    for c in CAT_FEATURES:
        train_pd[c] = train_pd[c].astype('category')
    prep = fcst.preprocess(train_pd, static_features=STATIC_FEATURES, dropna=True)
    cols = [c for c in prep.columns if c not in ('unique_id', 'ds', 'y')]
    X = prep[cols]
    if len(X) > n_sample:
        X = X.sample(n=n_sample, random_state=42)
    for c in CAT_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype('category')
    sv = shap.TreeExplainer(fcst.models_[MODEL]).shap_values(X)
    imp = (pd.DataFrame({'feature': X.columns, 'mean_abs_shap': np.abs(sv).mean(0)})
             .sort_values('mean_abs_shap', ascending=False).reset_index(drop=True))
    imp.to_csv(OUTPUT_DIR / f'shap_importance_{target}.csv', index=False)
    Xp = X.copy()
    for c in CAT_FEATURES:
        if c in Xp.columns:
            Xp[c] = Xp[c].cat.codes
    plt.figure(figsize=(9, 7))
    shap.summary_plot(sv, Xp, show=False, max_display=25, feature_names=list(X.columns))
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / f'shap_summary_{target}.png', dpi=130); plt.close()
    print(f'  [shap:{target}] top features:\n{imp.head(8).to_string(index=False)}')


def run_target(target, static_df):
    print(f'\n=== tier2  target={target}  model={MODEL} ===', flush=True)
    rows, last = [], None
    for fold, tr, te, end in splits.walk_forward_folds(n=6, target=target):
        thermal.cool_if_hot(tag=f'tier2-{target[:3]}-f{fold}')
        t0 = time.time()
        merged, fcst = fit_predict(target, tr, te, static_df)
        if not merged.empty:
            train_raw = _pd(tr)[['unique_id', 'ds', 'y']]   # raw train → comparable MASE scale
            mm = metrics.evaluate_at_horizons(merged, train_raw, [MODEL])
            mm['fold'] = fold; mm['test_end'] = str(end); rows.append(mm)
            last = (tr, fcst)
        print(f'  fold {fold}: test_end={end}  ids={te["unique_id"].n_unique()}  '
              f'{time.time() - t0:.1f}s  cpu={thermal.cpu_c():.1f}°C', flush=True)

    if not rows:
        print('  no metrics produced'); return
    out = pd.concat(rows, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = OUTPUT_DIR / f'metrics_{target}.csv'
    out.to_csv(fp, index=False)
    print(f'  written → {fp}')
    agg = out[out['horizon'] == 60].groupby('metric')[[MODEL]].mean()
    print(f'  cross-fold mean @h=60:\n{agg.round(4).to_string()}')
    if last is not None:
        try:
            shap_analysis(last[1], last[0], static_df, target)
        except Exception as exc:
            print(f'  [shap] failed: {exc}', flush=True)


def main():
    print(f'[tier2] start cpu={thermal.cpu_c():.1f}°C', flush=True)
    static_df = load_static()
    print(f'  static: {static_df.shape[0]} tickers, sectors={static_df["sector"].nunique()}', flush=True)
    for t in TARGETS:
        run_target(t, static_df)
    print(f'[tier2] done cpu={thermal.cpu_c():.1f}°C', flush=True)


if __name__ == '__main__':
    main()
