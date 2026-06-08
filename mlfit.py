"""
Shared LightGBM trainer with TIME-ORDERED VALIDATION + EARLY STOPPING.

This is the "use a metric for model assessment during training" piece:
  1. preprocess the panel into the supervised (lags + rolling + static) matrix,
  2. hold out the last `val_days` business days as a time-ordered validation set,
  3. train with early stopping on validation **MAE** (how far the 1-step prediction
     lands from the actual increment) — stops before the model overfits,
  4. report val MAE, best iteration, and **directional accuracy** (does it get the
     sign of the move right? ≈0.5 means no timing skill — the honest random-walk result),
  5. refit at the best iteration for the recursive multi-step forecast.

Levels (`close`) are first-differenced so the model predicts increments and cannot
compound a bias into a runaway path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from mlforecast.target_transforms import Differences

LAGS = [1, 5, 10, 20, 60, 120, 252]
ROLL = [5, 20, 60]
STATIC = ['sector', 'listing_age_years']
LAG_TF = {1: [RollingMean(w) for w in ROLL] + [RollingStd(w) for w in ROLL]}
PARAMS = dict(learning_rate=0.05, max_depth=8, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
              objective='regression', verbosity=-1, n_jobs=-1, random_state=42)


def _mlf(target: str, n_estimators: int) -> MLForecast:
    return MLForecast(
        models={'lgb': lgb.LGBMRegressor(n_estimators=n_estimators, **PARAMS)},
        freq='B', lags=LAGS, lag_transforms=LAG_TF,
        target_transforms=[Differences([1])] if target == 'close' else [])


def fit_lgb_es(train_pd: pd.DataFrame, target: str = 'close', val_days: int = 60,
               max_trees: int = 3000, patience: int = 100) -> tuple[MLForecast, dict]:
    """train_pd: long df [unique_id, ds, y, sector, listing_age_years] (y already the
    target: close level or log_return). Returns (fitted MLForecast, assessment info)."""
    prep = _mlf(target, max_trees).preprocess(train_pd, static_features=STATIC, dropna=True)
    prep['sector'] = prep['sector'].astype('category')
    feats = [c for c in prep.columns if c not in ('unique_id', 'ds', 'y')]

    uniq_days = prep['ds'].drop_duplicates().sort_values()
    cut = uniq_days.iloc[-val_days] if len(uniq_days) > val_days else uniq_days.iloc[len(uniq_days) // 2]
    tr, va = prep[prep['ds'] < cut], prep[prep['ds'] >= cut]

    model = lgb.LGBMRegressor(n_estimators=max_trees, **PARAMS)
    model.fit(tr[feats], tr['y'], eval_set=[(va[feats], va['y'])], eval_metric='l1',
              callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)])
    best_iter = int(model.best_iteration_ or max_trees)
    val_mae = float(model.best_score_['valid_0']['l1'])
    pred = model.predict(va[feats], num_iteration=best_iter)
    # directional accuracy: sign of predicted increment/return vs actual (drop ~0 moves)
    a, p = va['y'].to_numpy(), np.asarray(pred)
    mask = np.abs(a) > 1e-9
    dir_acc = float((np.sign(p[mask]) == np.sign(a[mask])).mean()) if mask.any() else float('nan')

    fcst = _mlf(target, best_iter)
    tp = train_pd.copy(); tp['sector'] = tp['sector'].astype('category')
    fcst.fit(tp, static_features=STATIC, keep_last_n=max(LAGS) + 60)
    return fcst, {'val_mae': val_mae, 'dir_acc': dir_acc, 'best_iter': best_iter, 'n_val': int(len(va))}
