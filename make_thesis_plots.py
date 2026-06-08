"""
Generate figures for THESIS.md from the materialised panel + tier metrics.
Outputs → assets/thesis/*.png  (committed for GitHub rendering).
Usage:  ./run.sh make_thesis_plots.py
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import data            # noqa: E402
import build_scoreboard as bs   # noqa: E402

OUT = ROOT / 'assets' / 'thesis'
SSD = Path(os.environ.get('EQUITY_FORECAST_HOME', '/mnt/ssd/equity_forecast'))
TICKER = 'RELIANCE.NS'
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10, 'axes.grid': True,
                     'grid.alpha': 0.3})


def _series(uid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = (data.load_panel().filter(pl.col('unique_id') == uid)
          .select(['ds', 'close', 'log_return']).sort('ds').collect())
    return (df['ds'].to_numpy(), df['close'].to_numpy(),
            df['log_return'].drop_nulls().to_numpy())


def _acf(x: np.ndarray, n: int = 30) -> np.ndarray:
    x = x - x.mean()
    denom = np.sum(x * x)
    return np.array([1.0] + [np.sum(x[:-k] * x[k:]) / denom for k in range(1, n + 1)])


def fig_stationarity():
    ds, close, _ = _series(TICKER)
    rets = np.diff(np.log(close))
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(ds, close, lw=0.8, color='#1f3b73')
    ax[0].set_title(f'{TICKER} — adjusted close (level): trending, non-stationary')
    ax[0].set_ylabel('price (INR)')
    ax[1].plot(ds[1:], rets, lw=0.4, color='#a83232')
    ax[1].axhline(0, color='k', lw=0.6)
    ax[1].set_title('log-returns: mean-reverting around 0, ~stationary')
    ax[1].set_ylabel('log-return')
    fig.tight_layout(); fig.savefig(OUT / 'fig_stationarity.png'); plt.close(fig)


def fig_acf():
    _, close, rets = _series(TICKER)
    ap, ar = _acf(close), _acf(rets)
    lags = np.arange(len(ap))
    ci = 1.96 / np.sqrt(len(rets))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].stem(lags, ap); ax[0].set_title(f'ACF of price (level) — {TICKER}')
    ax[0].set_xlabel('lag'); ax[0].set_ylabel('autocorrelation')
    ax[1].stem(lags, ar); ax[1].axhline(ci, color='r', ls='--', lw=0.8)
    ax[1].axhline(-ci, color='r', ls='--', lw=0.8)
    ax[1].set_title('ACF of log-returns (≈ white noise)')
    ax[1].set_xlabel('lag')
    fig.tight_layout(); fig.savefig(OUT / 'fig_acf.png'); plt.close(fig)


def _board(long, target, h):
    b = bs.board(long, target, h)
    return b.set_index('model')['MASE'] if not b.empty else None


def fig_mase_bars(long):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # levels (log scale because HistoricAverage ~ 87)
    s = _board(long, 'close', 20).sort_values()
    ax[0].barh(range(len(s)), s.values, color='#1f3b73')
    ax[0].set_yticks(range(len(s))); ax[0].set_yticklabels(s.index, fontsize=8)
    ax[0].set_xscale('log'); ax[0].invert_yaxis()
    ax[0].set_title('MASE — target=close, h=20 (log scale)'); ax[0].set_xlabel('MASE')
    # returns (linear)
    s = _board(long, 'log_return', 20).sort_values()
    colors = ['#2e7d32' if v < 1 else '#a83232' for v in s.values]
    ax[1].barh(range(len(s)), s.values, color=colors)
    ax[1].set_yticks(range(len(s))); ax[1].set_yticklabels(s.index, fontsize=8)
    ax[1].axvline(1.0, color='k', ls='--', lw=0.8, label='MASE=1 (seasonal-naive)')
    ax[1].invert_yaxis(); ax[1].legend(fontsize=8)
    ax[1].set_title('MASE — target=log_return, h=20'); ax[1].set_xlabel('MASE')
    fig.tight_layout(); fig.savefig(OUT / 'fig_mase_bars.png'); plt.close(fig)


def fig_error_vs_horizon(long):
    horizons = [5, 20, 60]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, target in enumerate(['close', 'log_return']):
        boards = {h: bs.board(long, target, h).set_index('model')['MASE'] for h in horizons}
        models = boards[20].sort_values().index[:5]
        for m in models:
            ax[j].plot(horizons, [boards[h].get(m, np.nan) for h in horizons],
                       marker='o', label=m)
        ax[j].set_title(f'MASE vs horizon — target={target}')
        ax[j].set_xlabel('horizon (trading days)'); ax[j].set_ylabel('MASE')
        ax[j].set_xticks(horizons); ax[j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / 'fig_error_vs_horizon.png'); plt.close(fig)


def copy_shap():
    for t in ('close', 'log_return'):
        src = SSD / 'tier2_global_ml' / 'output' / f'shap_summary_{t}.png'
        if src.exists():
            shutil.copy(src, OUT / f'shap_summary_{t}.png')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_stationarity()
    fig_acf()
    long = bs.load_long()
    fig_mase_bars(long)
    fig_error_vs_horizon(long)
    copy_shap()
    print('thesis figures →', OUT)
    for p in sorted(OUT.glob('*.png')):
        print('  ', p.name, f'{p.stat().st_size // 1024} KB')


if __name__ == '__main__':
    main()
