# ELTIF Liquidity Buffer Simulation

Monte Carlo simulation framework for European Long-Term Investment Funds (ELTIFs) with a
revolving credit facility and varying haircut rates . It models how private-market (PM) fund performance, investor flows, and liquidity management interact across macroeconomic regimes, and lets a fund
manager look up an optimal cash buffer for a given credit/haircut/risk-aversion scenario.

All proprietary data (CSV, Excel, notebooks) is excluded from this repository. See [Data Requirements](#data-requirements) below for what you need to supply to reproduce the pipeline end to end.

## Pipeline architecture

```
Raw data (PM indices, LSEG fund panel, Bloomberg benchmarks, public FRED series)
   │
   ▼
[1] Return unsmoothing            Private_Markets_handling.py, cash_index_handling.py
   │
   ▼
[2] Regime-conditional scenarios  LSEG_merged/new_macro_indicators.py, Private_Markets_Simulation.py
   │
   ▼
[3] Fund flow regression          LSEG_merged/Fund_Flows_Regression.py (+ regression_input_analysis.py batch driver)
   │
   ▼
[4] ELTIF simulation engine       Core_fund_simulation.py, driven by run_simulation_pipeline.py /
                                   optimize_cash_buffer.py / optimization_grid_analysis.py / run_grid_batch.py
   │
   ▼
[5] Reporting.                    streamlit_dashboard.py, gamma_sensitivity_analysis.py,
                                   panel_f_crra_improvement.py, summary.py, pm_factor_regression.py
```

### Stage 1 — Private-market return unsmoothing

PM indices exhibit artificial autocorrelation from stale pricing and reporting lags.
`Private_Markets_handling.py` loads the Hamilton Lane index levels and fits a
Getmansky-Lo-Makarov MA(k) model (order chosen by BIC) to invert the smoothing filter,
producing `Hamilton_Lane_unsmoothed_returns.csv`. `cash_index_handling.py` converts the
EUR003 (3‑month Euribor) rate into quarterly compounded returns, producing
`Cash_quarterly_returns.csv`.

### Stage 2 — Regime-conditional scenario generation

`LSEG_merged/new_macro_indicators.py` builds a 4‑regime growth × inflation classification
(Goldilocks / Overheating / Downturn / Stagflation) following Ilmanen, Maloney & Ross
(2014), from public FRED and Philadelphia Fed SPF data. `Private_Markets_Simulation.py`
then merges unsmoothed PM returns, EUR cash returns, and EUR benchmark returns (Bloomberg
Euro Stoxx 600 / Euro Agg Bond), classifies each historicalquarter into a regime, and generates
10,000 Monte Carlo paths × 80 quarters via a within-regime block bootstrap that preserves cross-asset correlation. Output: `simulated_pm_cash_returns.csv`.

### Stage 3 — Fund flow regression (Goldstein model)

`LSEG_merged/Fund_Flows_Regression.py` estimates a Goldstein-style flow-performance
regression on ~65,500 quarterly fund-quarter LSEG/Lipper observations: rolling 8-quarter
fund alpha vs. S&P 500/bond benchmarks, regressed against flow rate with a kink for
negative alpha, regime interactions, lagged flow, log(TNA), and TER. It's heavily
parameterized via CLI flags so it can be re-run per asset-class subset (`--run_tag`,
`--keyword`, `--tna_min`, `--alpha_lag`, `--skip_model5`). Output: fitted coefficients as
`LSEG_merged/goldstein_model2_coefficients.{pkl,json}` (or under `LSEG_merged/runs/` when
tagged). `regression_input_analysis.py` drives this script across a fixed list of
asset-class subsets and builds a comparison table across all runs.

### Stage 4 — ELTIF simulation with revolving credit

`Core_fund_simulation.py` is the main engine. Each quarter, for every simulated path:

1. Apply the PM return and deduct TER.
2. Charge interest on any outstanding credit balance.
3. Compute rolling 8-quarter alpha and the Goldstein flow rate for the current regime.
4. Settle the net flow via the liquidity waterfall:
   - **Outflow:** Cash buffer → Credit line → Forced PM liquidation (with a haircut).
   - **Inflow:** Fill cash buffer → Repay outstanding credit → Rest into PM.
5. Record `Shortfall_Flag` (0 = none, 1 = credit drawn, 2 = forced liquidation) and all
   state variables.

`run_simulation_pipeline.py` is a CLI wrapper for a single scenario run. `optimize_cash_buffer.py`
grid-searches the cash weight across credit-capacity scenarios and derives the
CRRA-utility-optimal weight `w*` over a grid of risk-aversion values γ.
`optimization_grid_analysis.py` runs that over a haircut × spread grid, and
`run_grid_batch.py` runs the grid analysis over every `gate_mode × redemption_gate_pct`
combination — this is what actually populates `optimization_runs/`.

### Stage 5 — Reporting 

`streamlit_dashboard.py` ("ELTIF Cash Buffer Lookup Tool") is a **static lookup UI** over
the pre-computed CSVs in `optimization_runs/` — it does not re-run simulations live.
`gamma_sensitivity_analysis.py` and `panel_f_crra_improvement.py` are thesis-appendix
scripts that post-process `optimization_runs/grid_summary_*.csv` into figures/tables.
`summary.py` generates the thesis's descriptive-statistics chapter (fund universe,
cumulative returns, regime transition matrix, unsmoothing comparison). `pm_factor_regression.py`
is an independent robustness check regressing unsmoothed PM returns on market factors and
regime dummies, to justify the regime-conditional bootstrap used in Stage 2.

### Dependencies

Install with `pip install -r requirements.txt`.
One hard dependency is **not** in `requirements.txt`: `lseg.data` (LSEG/Refinitiv
Workspace API), needed only by `LSEG_merged/fund_flow_data.ipynb` to pull the raw
fund-flow panel from a licensed LSEG Workspace session. Everything downstream of the
already-exported CSVs runs without it.

## Data Requirements

All paths below are relative to the repository root. None of these files are committed
(see `.gitignore`); place them at these exact paths to reproduce the pipeline.

### Proprietary — must be obtained from a data provider

| Data | Source | Required file (path) | Used by |
|---|---|---|---|
| Hamilton Lane private-market index levels — quarterly USD levels for the Private Credit / Private Equity / (composite) Private Markets indices | Hamilton Lane | `PM_Indices.xlsx`, sheet `Hamilton_Lane` (`Date` + one column per index) | `Private_Markets_handling.py`, `summary.py` |
| 3-month Euribor rate (EUR003), annualized % | Bloomberg (ticker convention) | `Liquid_Assets_2.xlsx`, sheet with `Date`, `EUR003_Index` | `cash_index_handling.py` |
| EUR benchmark total-return levels: Euro Stoxx 600 and Bloomberg Euro Agg Bond index | Bloomberg (falls back to `yfinance` automatically if this file is absent) | `bloomberg_benchmarks.csv` (`Date`, `SXXR_Index`, `LBEATREU_Index`) | `Private_Markets_Simulation.py`, `pm_factor_regression.py`, `summary.py` |
| US open-end mutual fund register with Lipper Global asset-class classification | LSEG Data & Analytics (Lipper Global), via `lseg.data` in `LSEG_merged/fund_flow_data.ipynb` | `LSEG_merged/Clean_funds_with_asset_class.csv` (`DocumentTitle`, `RIC`, `FundClassLipperID`, `FundEntityLipperId`, `FundClassCurrency`, `IssueLipperGlobalSchemeName`, `FundClassStatus(Name)`, `RIC_clean`) | `Fund_Flows_Regression.py`, `summary.py` |
| Per-fund quarterly total-return series | LSEG (`TR.FundRollingPerformance`) | `LSEG_merged/returns_merged.csv` (`Instrument`, `Date`, `Rolling Performance`) | `Fund_Flows_Regression.py`, `summary.py` |
| Per-fund quarterly total net assets | LSEG (`TR.FundTotalNetAssets`) | `LSEG_merged/tna_merged.csv` (wide: one column per RIC) | `Fund_Flows_Regression.py` |
| Per-fund total expense ratio and inception date | LSEG (`TR.FundTER`) | `LSEG_merged/ter_merged.csv` (`Instrument`, `Date`, `Total Expense Ratio`, `Fund Inception Date`) | `Fund_Flows_Regression.py`, `summary.py` |

### Public / fully reproducible (no data-provider access needed)

| Data | Source | Required file (path) | Used by |
|---|---|---|---|
| Growth × inflation macro-regime classification (Ilmanen/Maloney/Ross) | FRED (CFNAI, INDPRO, CPIAUCSL) + Philadelphia Fed SPF, built by `LSEG_merged/new_macro_indicators.py` | `LSEG_merged/macro_indicators.csv` | `Private_Markets_Simulation.py`, `pm_factor_regression.py`, `summary.py`, `Fund_Flows_Regression.py` |
| USD benchmark returns: S&P 500, total bond market, 3M T-bill | yfinance (`^GSPC`, `VBMFX`, `^IRX`), built by `download_benchmarks()` in `Fund_Flows_Regression.py` | `LSEG_merged/benchmarks.csv` | `summary.py` |
| GDP/CPI/Fed-funds-based macro regime series | FRED GDP/CPI/FedFunds + NBER recession indicator, derived inside `Fund_Flows_Regression.py` | `LSEG_merged/macro_regimes.csv` | intermediate/output |
| Cached quarterly EUR benchmark returns | Derived from the Bloomberg-or-yfinance-fallback benchmark levels above | `eur_benchmarks.csv` | auto-regenerated by `Private_Markets_Simulation.py` if missing |


