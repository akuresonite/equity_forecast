"""
Shared evaluation helpers for equity_forecast.

evaluate_at_horizons() takes a `merged` frame (forecasts joined to actuals, one
column per model) plus the training frame, and computes MAE/RMSE/sMAPE/MASE
restricted to the first H trading days per series, for each H in `horizons`.
That is the whole reason every tier forecasts once at the longest horizon and
reports 5/20/60 for free.

All frames are pandas (utilsforecast native). Long format:
  merged : unique_id, ds, y, <model_1>, <model_2>, ...
  train  : unique_id, ds, y
Returns long metrics: unique_id, metric, horizon, <model cols>.
"""
from __future__ import annotations

from functools import partial

import pandas as pd
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse, smape, mase

HORIZONS = (5, 20, 60)
SEASON = 5

mase5 = partial(mase, seasonality=SEASON)
mase5.__name__ = 'mase'          # utilsforecast labels rows by metric.__name__
_METRICS = [mae, rmse, smape, mase5]


def _first_h(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """First h rows per series, in ds order."""
    return (df.sort_values(['unique_id', 'ds'])
              .groupby('unique_id', group_keys=False).head(h))


def evaluate_at_horizons(merged: pd.DataFrame, train: pd.DataFrame,
                         model_cols: list[str], horizons=HORIZONS) -> pd.DataFrame:
    out = []
    for h in horizons:
        ev = evaluate(_first_h(merged, h), metrics=_METRICS,
                      train_df=train, models=list(model_cols))
        ev = ev.copy()
        ev.insert(2, 'horizon', h)
        out.append(ev)
    return pd.concat(out, ignore_index=True)
