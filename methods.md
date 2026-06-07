# Methods — tier roadmap

A disciplined, tier-based forecasting benchmark adapted from the sibling
`nav_forecast` project. Each tier must clear the floor set by the tier below it;
the point is to learn *what actually moves the needle*, not to assume complexity
wins. Across every tier we forecast **both** targets, evaluate at **3 horizons**,
and validate with **walk-forward CV**.

## Shared protocol

| Aspect | Choice | Rationale |
|---|---|---|
| Targets | `close` (level) **and** `log_return` | Levels are intuitive but non-stationary; log-returns are the finance-honest, stationary target. |
| Horizons | 5, 20, 60 trading days | Sliced from one 60-step forecast — 1 week / 1 month / 1 quarter. |
| Validation | Walk-forward CV, 6 folds, 60-day test, expanding train, ~1-month step | Time-ordered; no leakage from future folds. |
| Metrics | MAE, RMSE, sMAPE, **MASE** (seasonality 5) | MASE is scale-free and comparable across stocks; we report mean **and** median. |
| Frequency grid | business-day (`B`) + forward-fill | mlforecast needs a uniform grid; the NSE calendar has holiday gaps. |
| Static covariate | `sector` (+ `listing_age_years`) | The only leak-free cross-sectional signal in this dataset (see leakage note). |

## Tiers

- **Tier 0 — trivial baselines.** Naive, SeasonalNaive(5), RandomWalkWithDrift,
  HistoricAverage, WindowAverage(20). Establishes the floor. RandomWalkWithDrift
  is famously hard to beat on price *levels*.
- **Tier 1 — classical per-series.** AutoARIMA, AutoETS, AutoTheta, AutoCES
  (statsforecast, season 5). Fully feasible here — 50 series × ~25 yr runs in
  minutes (the sibling project abandoned this tier only at 3,000+ series).
- **Tier 2 — global tree ML.** One LightGBM across all tickers (mlforecast):
  lags `[1,5,10,20,60,120,252]`, rolling mean/std on lag-1 over `[5,20,60]`,
  `Differences([1])` for `close` only, static `sector` + `listing_age_years`.
  SHAP on the final-fold refit. Pooling lets short-history tickers borrow
  strength; returns is where global ML tends to win.

## Live forward forecast

After backtesting, models are refit on **all** history and projected ~60
business days forward: RandomWalkWithDrift + AutoETS (with 80/95% intervals) on
levels, and the tier2 LightGBM on returns integrated to an implied price path.
Per-ticker charts land in `assets/forecasts/`.

## Leakage & survivorship (critical)

- The CSV's **fundamentals** (PE, EPS, Beta, Market_Cap, Forward_PE, PEG,
  Price_to_Book, Dividend_Yield, 52-week H/L) are **point-in-time 2026 snapshots
  repeated across every historical date** (`eda.py` proves `n_unique==1` per
  ticker). Using them as time-varying features is lookahead leakage → **dropped**.
- The pre-computed `Daily_Return / Volatility_20D / MA_50 / MA_200` have unknown
  provenance; we **recompute trailing features inside the models** instead.
- The panel holds only the **current** Nifty50 constituents → **survivorship
  bias** (winners over-represented). Not fixable from this data; documented.

## Deferred (future tiers)

Tier 3 neural (NHITS/TFT/PatchTST), tier 4 hierarchical reconciliation
(sector → stock), tier 5 probabilistic/conformal, tier 6 foundation models
(Chronos/TimeGPT), tier 7 financial-ML discipline (purged CV + embargo, regime
features). The walk-forward CV here has no embargo/purging yet.
