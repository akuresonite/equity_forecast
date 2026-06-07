"""
Train / test splits for equity_forecast (adapted from nav_forecast/splits.py).

  single_split(target)       — one fixed train | val | test (60 trading days each)
  walk_forward_folds(target) — 6 monthly walk-forward folds, 60d test, expanding train

Eligibility (applied first): keep tickers with >= MIN_USABLE rows and a recent
last observation (drops any stale / delisted ticker). Returns polars DataFrames
in Nixtla long format:
  unique_id (str)  ds (date)  y (float)
y is `close` by default; pass target='log_return' for returns.

The Nifty50 panel shares one NSE calendar, so global trading-day indexing is
exact. TEST_DAYS = 60 is the longest evaluation horizon; metrics for the 5/20d
horizons are sliced from the same 60-step forecasts (see metrics.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterator

import polars as pl

import data

TEST_DAYS  = 60          # = the longest evaluation horizon
VAL_DAYS   = 60
MIN_TRAIN  = 252         # 1 trading year
MIN_USABLE = MIN_TRAIN + VAL_DAYS + TEST_DAYS    # 372
RECENT_CUTOFF = date(2026, 1, 1)                 # drop stale / delisted tickers


@dataclass
class Split:
    train: pl.DataFrame
    val:   pl.DataFrame
    test:  pl.DataFrame
    val_start:     date
    test_start:    date
    test_end_incl: date


def _eligible_ids(panel: pl.LazyFrame) -> set[str]:
    per = (
        panel.group_by('unique_id')
        .agg([pl.len().alias('rows'), pl.col('ds').max().alias('max_ds')])
        .filter((pl.col('rows') >= MIN_USABLE) & (pl.col('max_ds') >= RECENT_CUTOFF))
        .collect()
    )
    return set(per['unique_id'].to_list())


def _slice(panel: pl.LazyFrame, ids: set[str], target: str) -> pl.DataFrame:
    return (
        panel.filter(pl.col('unique_id').is_in(ids))
        .select(['unique_id', 'ds', pl.col(target).alias('y')])
        .filter(pl.col('y').is_not_null())
        .sort(['unique_id', 'ds'])
        .collect()
    )


def _trading_days(df: pl.DataFrame) -> list[date]:
    return df.select(pl.col('ds').unique().sort()).to_series().to_list()


def single_split(target: str = 'close') -> Split:
    panel = data.load_panel()
    df = _slice(panel, _eligible_ids(panel), target)
    days = _trading_days(df)
    test_end_incl = days[-1]
    test_start    = days[-TEST_DAYS]
    val_start     = days[-(TEST_DAYS + VAL_DAYS)]
    train = df.filter(pl.col('ds') < val_start)
    val   = df.filter((pl.col('ds') >= val_start) & (pl.col('ds') < test_start))
    test  = df.filter(pl.col('ds') >= test_start)
    return Split(train, val, test, val_start, test_start, test_end_incl)


def walk_forward_folds(n: int = 6, target: str = 'close'
                       ) -> Iterator[tuple[int, pl.DataFrame, pl.DataFrame, date]]:
    panel = data.load_panel()
    df = _slice(panel, _eligible_ids(panel), target)
    days = _trading_days(df)
    for fold in range(n):
        offset = (n - 1 - fold) * 21          # ~1 trading month per fold
        test_end_idx = len(days) - 1 - offset
        test_start_idx = test_end_idx - TEST_DAYS + 1
        if test_start_idx < MIN_TRAIN:
            continue
        test_start, test_end = days[test_start_idx], days[test_end_idx]
        train = df.filter(pl.col('ds') < test_start)
        test  = df.filter((pl.col('ds') >= test_start) & (pl.col('ds') <= test_end))
        yield fold, train, test, test_end


if __name__ == '__main__':
    s = single_split()
    print(f'single: train={s.train.height:,} val={s.val.height:,} '
          f'test={s.test.height:,} test_end={s.test_end_incl}')
    for fold, tr, te, end in walk_forward_folds():
        print(f'fold {fold}: train={tr.height:,} test={te.height:,} '
              f'test_end={end} ids={te["unique_id"].n_unique()}')
