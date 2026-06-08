"""
GBDT bake-off + a small hyperparameter search.

Answers "why not XGBoost / CatBoost, and would tuning help?" by running them
head-to-head with LightGBM (default + an Optuna-tuned LightGBM) on the walk-forward,
for `close` (first-differenced) and `log_return`, reporting MASE at h=20/60 next to
the tier-0 RWD floor.

Expectation (literature + this data): gradient-boosted trees CLUSTER — they tie each
other and tie the random walk on daily equity point forecasts. This script is here to
*show* that rather than assert it.

3 walk-forward folds (xgb/cat are slow on ARM). Writes assets/bakeoff.csv.
Usage: ./run.sh bakeoff.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd
import polars as pl

from mlforecast import MLForecast
from mlforecast.target_transforms import Differences

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import data       # noqa: E402
import splits     # noqa: E402
import metrics    # noqa: E402
import mlfit      # noqa: E402
import thermal    # noqa: E402

FOLDS = 3
TARGETS = ['close', 'log_return']
STATIC = mlfit.STATIC
CAT = ['sector']


def _pd(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas().assign(ds=lambda d: pd.to_datetime(d['ds']))


def _expand(df_pd: pd.DataFrame) -> pd.DataFrame:
    df_pd = df_pd.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    b = df_pd.groupby('unique_id')['ds'].agg(['min', 'max']).reset_index()
    pieces = [pd.DataFrame({'unique_id': u, 'ds': pd.bdate_range(mn, mx)})
              for u, mn, mx in b.itertuples(index=False) if len(pd.bdate_range(mn, mx))]
    out = (pd.concat(pieces, ignore_index=True).merge(df_pd, on=['unique_id', 'ds'], how='left')
           .sort_values(['unique_id', 'ds']))
    out['y'] = out.groupby('unique_id')['y'].ffill()
    return out.reset_index(drop=True)


def make_model(name: str, params: dict | None = None):
    if name.startswith('lgb'):
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=1500, **(params or mlfit.PARAMS))
    if name == 'xgb':
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8,
                                objective='reg:squarederror', tree_method='hist',
                                enable_categorical=True, random_state=42, n_jobs=-1, verbosity=0)
    if name == 'cat':
        from catboost import CatBoostRegressor

        class CatAuto(CatBoostRegressor):           # keep cat_features out of __init__ so clone() works
            _CAT: list = []
            def fit(self, X, y=None, **kw):
                if 'cat_features' not in kw and self._CAT and hasattr(X, 'columns'):
                    kw['cat_features'] = [c for c in self._CAT if c in X.columns]
                return super().fit(X, y, **kw)
        CatAuto._CAT = list(CAT)
        return CatAuto(iterations=300, learning_rate=0.05, depth=8, l2_leaf_reg=3,
                       loss_function='RMSE', verbose=False, random_seed=42,
                       allow_writing_files=False, thread_count=-1)
    raise ValueError(name)


def fcst(name, target, params=None):
    return MLForecast(models={name: make_model(name, params)}, freq='B', lags=mlfit.LAGS,
                      lag_transforms=mlfit.LAG_TF,
                      target_transforms=[Differences([1])] if target == 'close' else [],
                      num_threads=4)


def evaluate(name, target, static_df, params=None) -> pd.DataFrame:
    rows = []
    for fold, tr, te, end in splits.walk_forward_folds(n=FOLDS, target=target):
        thermal.cool_if_hot(tag=f'bakeoff-{name}-{target[:3]}-f{fold}')
        trp = _expand(_pd(tr)).merge(static_df, on='unique_id', how='left').dropna(subset=['y'])
        trp['sector'] = trp['sector'].astype('category')
        f = fcst(name, target, params)
        f.fit(trp, static_features=STATIC, keep_last_n=max(mlfit.LAGS) + 60)
        fc = f.predict(h=splits.TEST_DAYS + 15)
        merged = fc.merge(_pd(te)[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='inner')
        if merged.empty:
            continue
        m = metrics.evaluate_at_horizons(merged, _pd(tr)[['unique_id', 'ds', 'y']], [name])
        m['fold'] = fold
        rows.append(m)
    out = pd.concat(rows, ignore_index=True)
    agg = (out[out['horizon'].isin([20, 60])].groupby(['metric', 'horizon'])[name].mean())
    return agg.rename(name)


def tune_lgb(static_df) -> dict:
    """Small Optuna search: minimise time-validation MAE on fold-0 `close` train."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fold0 = next(iter(splits.walk_forward_folds(n=FOLDS, target='close')))
    trp = _expand(_pd(fold0[1])).merge(static_df, on='unique_id', how='left').dropna(subset=['y'])

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int('num_leaves', 15, 127),
            max_depth=trial.suggest_int('max_depth', 4, 12),
            min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 50, 500),
            feature_fraction=trial.suggest_float('feature_fraction', 0.6, 1.0),
            bagging_fraction=trial.suggest_float('bagging_fraction', 0.6, 1.0),
            bagging_freq=5, objective='regression', verbosity=-1, n_jobs=-1, random_state=42)
        import lightgbm as lgb
        base = MLForecast(models={}, freq='B', lags=mlfit.LAGS, lag_transforms=mlfit.LAG_TF,
                          target_transforms=[Differences([1])])
        prep = base.preprocess(trp, static_features=STATIC, dropna=True)
        prep['sector'] = prep['sector'].astype('category')
        feats = [c for c in prep.columns if c not in ('unique_id', 'ds', 'y')]
        days = prep['ds'].drop_duplicates().sort_values()
        cut = days.iloc[-60]
        a, b = prep[prep['ds'] < cut], prep[prep['ds'] >= cut]
        model = lgb.LGBMRegressor(n_estimators=2000, **params)
        model.fit(a[feats], a['y'], eval_set=[(b[feats], b['y'])], eval_metric='l1',
                  callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        return float(model.best_score_['valid_0']['l1'])

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=25, show_progress_bar=False)
    print(f'[bakeoff] Optuna best val-MAE={study.best_value:.3f}  params={study.best_params}', flush=True)
    p = dict(study.best_params); p.update(bagging_freq=5, objective='regression',
                                          verbosity=-1, n_jobs=-1, random_state=42)
    return p


def main():
    static_df = data.load_static().to_pandas()
    static_df['sector'] = static_df['sector'].fillna('unknown').astype(str)
    static_df['listing_age_years'] = static_df['listing_age_years'].fillna(0.0).astype(float)
    static_df = static_df[['unique_id'] + STATIC]

    print('[bakeoff] tuning LightGBM (Optuna, 25 trials)…', flush=True)
    try:
        best = tune_lgb(static_df)
    except Exception as exc:
        print(f'[bakeoff] tuning failed ({exc}); skipping tuned model', flush=True); best = None

    runs = [('lgb', None), ('xgb', None), ('cat', None)] + ([('lgb_tuned', best)] if best else [])
    cols = {}
    for target in TARGETS:
        for name, params in runs:
            t0 = time.time()
            cols[(target, name)] = evaluate(name, target, static_df, params)
            print(f'[bakeoff] {target:11s} {name:10s} done in {time.time()-t0:.0f}s', flush=True)

    # assemble a tidy table: rows = (target, metric, horizon), cols = models
    frames = []
    for (target, name), s in cols.items():
        df = s.reset_index(); df['target'] = target; df['model'] = name
        frames.append(df.rename(columns={name: 'value'}))
    tidy = pd.concat(frames, ignore_index=True)
    wide = tidy.pivot_table(index=['target', 'metric', 'horizon'], columns='model', values='value')
    (ROOT / 'assets').mkdir(exist_ok=True)
    wide.to_csv(ROOT / 'assets' / 'bakeoff.csv')
    print('\n[bakeoff] MASE (and others) by model — GBDTs should cluster:')
    print(wide.loc[wide.index.get_level_values('metric') == 'mase'].round(4).to_string())


if __name__ == '__main__':
    main()
