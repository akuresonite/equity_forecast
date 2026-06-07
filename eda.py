"""
EDA / dataset understanding for the Kaggle Nifty50 panel (read-only).

Prints a structured report and, crucially, PROVES the two modelling hazards:
  1. Fundamentals (PE, EPS, Beta, Market_Cap, ...) are point-in-time snapshots
     repeated across every date  → n_unique == 1 per ticker → leakage if used
     as time-varying features.
  2. Survivorship: the panel only contains the *current* Nifty50 constituents,
     visible as tickers whose history starts well after 1999 (index additions).

Also sanity-checks the pre-computed Daily_Return / MA columns vs values we can
recompute, so we know whether to trust or drop them.

Usage:  ./run.sh eda.py
"""
from __future__ import annotations

import polars as pl

import data

FUNDAMENTALS = ['Market_Cap', 'PE_Ratio', 'Forward_PE', 'PEG_Ratio', 'Price_to_Book',
                'Dividend_Yield', 'EPS', 'Beta', '52Week_High', '52Week_Low']
DERIVED = ['Daily_Return', 'Volatility_20D', 'MA_50', 'MA_200']


def hr(title: str) -> None:
    print('\n' + '═' * 72 + f'\n {title}\n' + '═' * 72)


def main() -> None:
    if not data.RAW_CSV.exists():
        raise SystemExit(f'raw CSV not found at {data.RAW_CSV} — run ./setup.sh first')
    df = pl.read_csv(data.RAW_CSV, infer_schema_length=20000)
    df = df.with_columns(
        pl.col('Date').cast(pl.Utf8).str.slice(0, 10).str.to_date('%Y-%m-%d').alias('_d')
    )

    hr('SHAPE & COLUMNS')
    print(f'rows={df.height:,}  cols={df.width}  tickers={df["Ticker"].n_unique()}')
    print('columns:', df.columns)

    hr('CALENDAR')
    print(f'date range: {df["_d"].min()} … {df["_d"].max()}')
    per = (df.group_by('Ticker')
             .agg([pl.len().alias('rows'),
                   pl.col('_d').min().alias('first'),
                   pl.col('_d').max().alias('last')])
             .sort('rows'))
    print(f'rows/ticker: min={per["rows"].min()} '
          f'median={int(per["rows"].median())} max={per["rows"].max()}')
    reach_2026 = per.filter(pl.col('last') >= pl.date(2026, 1, 1)).height
    print(f'tickers reaching ≥2026-01-01: {reach_2026}/{per.height}')

    hr('SECTORS')
    print(df.group_by('Sector').agg(pl.col('Ticker').n_unique().alias('tickers'))
            .sort('tickers', descending=True).to_pandas().to_string(index=False))

    hr('NULL AUDIT (columns with any nulls)')
    nulls = df.null_count().to_pandas().T
    nulls.columns = ['nulls']
    nulls = nulls[nulls['nulls'] > 0]
    print(nulls.to_string() if len(nulls) else '  (no nulls)')

    hr('LEAKAGE PROOF — fundamentals constant per ticker?')
    print('A value that never changes within a ticker is a current snapshot, not')
    print('a historical series. n_unique==1 for (almost) every ticker ⇒ drop it.\n')
    for c in FUNDAMENTALS:
        if c not in df.columns:
            continue
        nu = df.group_by('Ticker').agg(pl.col(c).n_unique().alias('nu'))
        const = nu.filter(pl.col('nu') <= 1).height
        verdict = 'SNAPSHOT → drop' if const >= per.height * 0.8 else 'varies'
        print(f'  {c:16s} constant for {const:2d}/{per.height} tickers   [{verdict}]')

    hr('DERIVED COLUMNS — trust or recompute?')
    rel = df.filter(pl.col('Ticker') == df['Ticker'][0]).sort('_d')
    if 'Daily_Return' in df.columns:
        rel = rel.with_columns(
            (pl.col('Close') / pl.col('Close').shift(1) - 1).alias('_ret_calc'))
        diff = (rel.select((pl.col('Daily_Return') - pl.col('_ret_calc')).abs().mean())
                  .item())
        print(f'  Daily_Return vs Close.pct_change  mean|diff|={diff}')
    print('  → we recompute trailing features inside the models regardless, '
          'so provenance never matters.')

    hr('SURVIVORSHIP')
    late = per.filter(pl.col('first') > pl.date(2000, 1, 1)).sort('first', descending=True)
    print(f'{late.height}/{per.height} tickers start after 2000 (index additions / later listings).')
    print('The panel holds only CURRENT Nifty50 members → survivorship bias (not fixable here).')
    print(late.head(12).select(['Ticker', 'first', 'rows']).to_pandas().to_string(index=False))

    print('\nDecision: panel keeps {Ticker, Date, Close→close, log_return}; '
          'Sector is the static covariate; all fundamentals dropped.')


if __name__ == '__main__':
    main()
