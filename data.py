"""
Data layer for equity_forecast.

Materialises the Kaggle Nifty50 CSV into two parquet files in the Nixtla long
format, mirroring nav_forecast/.../data.py but reading a CSV instead of Postgres.

  equity_panel.parquet  : unique_id (ticker), ds (date), close, log_return
  equity_static.parquet : one row per ticker — sector + listing metadata

Only price/return columns enter the panel. The CSV's "fundamentals" (PE, EPS,
Beta, Market_Cap, ...) are point-in-time 2026 snapshots repeated across every
historical date, so using them as time-varying features would be lookahead
leakage — they are deliberately dropped here. See eda.py for the proof.

CLI:
  python data.py materialise   # build the parquet files from the raw CSV
  python data.py info          # summarise the materialised panel
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl

DATA_DIR = Path(os.environ.get('EQUITY_FORECAST_DATA', '/mnt/ssd/equity_forecast/data'))
RAW_CSV  = DATA_DIR / 'raw' / 'nifty50_historical_data.csv'
PANEL    = DATA_DIR / 'equity_panel.parquet'
STATIC   = DATA_DIR / 'equity_static.parquet'

PRICE_COL = 'Close'      # split/dividend-adjusted
ID_COL    = 'Ticker'
DATE_COL  = 'Date'


def materialise() -> None:
    if not RAW_CSV.exists():
        sys.exit(f'raw CSV not found at {RAW_CSV} — run ./setup.sh first')
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw = pl.read_csv(RAW_CSV, infer_schema_length=20000)

    # Normalise Date whether polars parsed it as str / date / datetime.
    ds = pl.col(DATE_COL).cast(pl.Utf8).str.slice(0, 10).str.to_date('%Y-%m-%d')

    panel = (
        raw
        .select([
            pl.col(ID_COL).cast(pl.Utf8).alias('unique_id'),
            ds.alias('ds'),
            pl.col(PRICE_COL).cast(pl.Float64).alias('close'),
        ])
        .filter(pl.col('close').is_not_null() & (pl.col('close') > 0))
        .unique(subset=['unique_id', 'ds'], keep='first')
        .sort(['unique_id', 'ds'])
        .with_columns(
            (pl.col('close').log() - pl.col('close').log().shift(1).over('unique_id'))
            .alias('log_return')
        )
    )
    panel.write_parquet(PANEL)

    static = (
        raw
        .group_by(ID_COL)
        .agg([
            pl.col('Company_Name').first().alias('company_name'),
            pl.col('Sector').first().alias('sector'),
        ])
        .rename({ID_COL: 'unique_id'})
        .with_columns(pl.col('unique_id').cast(pl.Utf8))
        .join(
            panel.group_by('unique_id').agg([
                pl.col('ds').min().alias('first_ds'),
                pl.col('ds').max().alias('last_ds'),
                pl.len().alias('n_obs'),
            ]),
            on='unique_id', how='inner',
        )
        .with_columns(
            ((pl.col('last_ds') - pl.col('first_ds')).dt.total_days() / 365.25)
            .round(2).alias('listing_age_years')
        )
        .sort('unique_id')
    )
    static.write_parquet(STATIC)

    print('materialised:')
    print(f'  panel  → {PANEL}  ({panel.height:,} rows, {panel["unique_id"].n_unique()} tickers)')
    print(f'  static → {STATIC}  ({static.height} tickers)')
    print(f'  date range: {panel["ds"].min()} … {panel["ds"].max()}')


def load_panel() -> pl.LazyFrame:
    return pl.scan_parquet(PANEL)


def load_static() -> pl.DataFrame:
    return pl.read_parquet(STATIC)


def info() -> None:
    p = load_panel().collect()
    s = load_static()
    print(f'panel : {p.height:,} rows | {p["unique_id"].n_unique()} tickers | '
          f'{p["ds"].min()} … {p["ds"].max()}')
    print(f'static: {s.height} tickers | sectors: {s["sector"].n_unique()}')
    rows = p.group_by('unique_id').agg(pl.len().alias('n')).sort('n')
    print(f'rows/ticker: min={rows["n"].min()} '
          f'median={int(rows["n"].median())} max={rows["n"].max()}')
    print(s.select(['unique_id', 'sector', 'first_ds', 'last_ds', 'n_obs',
                    'listing_age_years']).head(50))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'info'
    if cmd == 'materialise':
        materialise()
    elif cmd == 'info':
        info()
    else:
        sys.exit(f'unknown command: {cmd!r} (use materialise|info)')
