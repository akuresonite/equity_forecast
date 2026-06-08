"""
Tier 3 — VOLATILITY: the one place a model genuinely beats naive on this data.

Daily returns are ~unpredictable in *direction* (the 1-step backtest showed directional
accuracy ≈50%), but their *variance clusters* — big moves follow big moves (the central
stylized fact of finance). So while no model beats the random walk on the price, a GARCH
model beats a constant-volatility assumption on the variance.

We fit GARCH(1,1) per stock and score its 1-step-ahead conditional variance with the
standard **QLIKE** loss against the realized proxy r² (robust to that proxy's noise):

    QLIKE = mean( log(σ̂²) + r² / σ̂² )      (lower is better)

Baselines: constant (unconditional) variance, and a trailing-20-day variance.

Writes assets/volatility/<TICKER>.png + assets/volatility_qlike.csv
Usage: ./run.sh tier3_volatility.py
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

from arch import arch_model

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import data       # noqa: E402
import splits     # noqa: E402
import thermal    # noqa: E402

TEST = 250        # held-out days for the volatility evaluation (~1 year)
SAMPLE = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS']
ASSETS = ROOT / 'assets' / 'volatility'


def qlike(r2: np.ndarray, var: np.ndarray) -> float:
    var = np.maximum(var, 1e-12)
    return float(np.mean(np.log(var) + r2 / var))


def garch_1step(ret_tr: np.ndarray, ret_te: np.ndarray):
    """Fit GARCH(1,1) on train returns; return 1-step-ahead conditional variance over the
    test window (return^2 units), computed from fitted params + realized returns."""
    res = arch_model(ret_tr * 100, mean='Constant', vol='Garch', p=1, q=1,
                     dist='normal').fit(disp='off')
    w, a, b = res.params['omega'], res.params['alpha[1]'], res.params['beta[1]']
    mu = res.params['mu']
    sig2 = float(res.conditional_volatility[-1] ** 2)
    prev_resid2 = float((ret_tr[-1] * 100 - mu) ** 2)
    out = []
    for r in ret_te * 100:
        sig2 = w + a * prev_resid2 + b * sig2          # 1-step forecast (uses info up to t-1)
        out.append(sig2)
        prev_resid2 = float((r - mu) ** 2)             # reveal realized return for next step
    return np.array(out) / (100.0 ** 2)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    panel = data.load_panel()
    ids = splits._eligible_ids(panel)
    ret = splits._slice(panel, ids, 'log_return').to_pandas()
    ret['ds'] = pd.to_datetime(ret['ds'])
    meta = data.load_static().to_pandas().set_index('unique_id').to_dict('index')

    rows, n_fail = [], 0
    print(f'[vol] fitting GARCH(1,1) on {len(ids)} tickers, {TEST}-day held-out eval…', flush=True)
    for i, uid in enumerate(sorted(ret['unique_id'].unique())):
        g = ret[ret['unique_id'] == uid].sort_values('ds')
        r = g['y'].to_numpy()
        if len(r) < TEST + 504:        # need >=2y train + the test year
            continue
        r_tr, r_te = r[:-TEST], r[-TEST:]
        ds_te = g['ds'].to_numpy()[-TEST:]
        try:
            var_garch = garch_1step(r_tr, r_te)
        except Exception:
            n_fail += 1; continue
        if i % 12 == 0:
            thermal.cool_if_hot(tag='garch')

        r2 = r_te ** 2
        var_const = np.full(TEST, float(np.var(r_tr)))             # constant (unconditional)
        # trailing 20-day variance, persisted 1 step (uses only past returns)
        s = pd.Series(r)
        trail = s.rolling(20).var().shift(1).to_numpy()[-TEST:]
        trail = np.where(np.isfinite(trail), trail, np.var(r_tr))

        rows.append({'unique_id': uid,
                     'qlike_garch': qlike(r2, var_garch),
                     'qlike_const': qlike(r2, var_const),
                     'qlike_trail20': qlike(r2, trail)})

        if uid in SAMPLE:
            real_vol = pd.Series(r_te).rolling(10).std().to_numpy() * 100 * np.sqrt(252)
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(ds_te, real_vol, color='#000', lw=1.6, label='realized vol (rolling 10d, ann.)')
            ax.plot(ds_te, np.sqrt(var_garch) * 100 * np.sqrt(252), color='#2ca02c', lw=1.6,
                    label='GARCH(1,1) 1-step forecast')
            ax.axhline(np.sqrt(np.var(r_tr)) * 100 * np.sqrt(252), color='#d62728', ls='--',
                       lw=1.2, label='constant-vol baseline')
            m = meta.get(uid, {})
            ax.set_title(f"{uid} · {m.get('company_name','')} — volatility is forecastable "
                         f"(unlike price)", fontsize=10)
            ax.set_ylabel('annualized volatility (%)'); ax.legend(fontsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')); fig.autofmt_xdate()
            fig.tight_layout(); fig.savefig(ASSETS / f'{uid.replace("/", "_")}.png', dpi=120)
            plt.close(fig)

    err = pd.DataFrame(rows)
    err.to_csv(ASSETS.parent / 'volatility_qlike.csv', index=False)
    win = (err['qlike_garch'] < err['qlike_const']).mean() * 100
    win_tr = (err['qlike_garch'] < err['qlike_trail20']).mean() * 100
    print(f'[vol] {len(err)} stocks fit ({n_fail} failed). Mean QLIKE (lower = better):')
    print(err[['qlike_garch', 'qlike_const', 'qlike_trail20']].mean().round(4).to_string())
    print(f'[vol] GARCH beats constant-vol on {win:.0f}% of stocks; '
          f'beats trailing-20d on {win_tr:.0f}%.')
    print('[vol] → volatility IS forecastable; this is where a model earns its keep.')


if __name__ == '__main__':
    main()
