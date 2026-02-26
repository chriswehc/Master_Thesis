# Model Workflow

## Overview

The codebase implements a **Monte Carlo simulation framework for European Long-Term Investment Funds (ELTIFs)** with a revolving credit facility. It models how PM fund performance, investor flows, and liquidity management interact across different macroeconomic regimes.

The pipeline has five sequential stages:

```
Raw Data (PM Indices, Fund Data, Macro)
    → [1] Return Unsmoothing
    → [2] Regime-Conditional Scenario Generation
    → [3] Fund Flow Regression (Goldstein Model)
    → [4] ELTIF Simulation with Revolving Credit
    → [5] Metrics & Visualization
```

---

## Stage 1 – Private Market Return Unsmoothing

**File:** `Private_Markets_handling.py`

PM indices exhibit artificial autocorrelation due to stale pricing and reporting lags. The **Getmansky-Lo-Makarov (GLM)** method recovers the true underlying returns:

1. Load Hamilton Lane PM indices from `PM_Indices.xlsx` (quarterly, 1997–2024)
2. For each series, fit an MA(k) model via MLE; select lag order k by BIC
3. Invert the smoothing filter to recover unsmoothed returns:

$$\eta_t = \frac{x_t - \sum_{j=1}^{k} \theta_j \cdot \eta_{t-j}}{\theta_0}$$

4. Output: `Hamilton_Lane_unsmoothed_returns.csv`

`cash_index_handling.py` converts EUR003 (3-month Euribor) from daily annualised to quarterly returns → `Cash_quarterly_returns.csv`

---

## Stage 2 – Regime-Conditional Scenario Generation

**File:** `Private_Markets_Simulation.py`

Four macro regimes are defined by GDP growth (high/low) × CPI inflation (high/low):

| Regime | Growth | Inflation |
|--------|--------|-----------|
| Goldilocks | High | Low |
| Overheating | High | High |
| Downturn | Low | Low |
| Stagflation | Low | High |

Historical quarters are bootstrapped **within regime** to preserve cross-asset correlations. For each of **10,000 Monte Carlo paths** over **80 quarters (20 years)**:

1. Merge unsmoothed PM returns, EUR cash returns, and benchmark returns (S&P 500, bonds, RF) by quarter
2. Classify each quarter into one of four macro regimes
3. Estimate regime transition probabilities → `macro_transition_matrix.csv`
4. For each path: simulate regime sequence using transition matrix, then sample a full historical row from the matching regime (block bootstrap)

**Output:** `simulated_pm_cash_returns.csv` (10,000 paths × 80 quarters)

---

## Stage 3 – Fund Flow Regression (Goldstein Model)

**File:** `LSEG_download/Fund_Flows_Regression.py`

Estimated on ~65,500 quarterly fund-quarter observations from LSEG/Lipper data.

**Alpha (rolling 8-quarter window per fund):**

```
Excess_Fund_Return = (Gross_Return − TER_quarterly) − RF
Alpha = intercept of: Excess_Fund_Return ~ Excess_SP500 + Excess_Bond
```

**Flow regression (Model 2):**

```
Flow_Rate ~ α_baseline · Alpha + α_neg · Alpha · I(Alpha < 0)
          + Σ_regime [regime interaction terms] · I(Regime)
          + β_lag · Lagged_Flow + β_TNA · log(TNA) + β_TER · TER
          + Regime fixed effects
```

**Fitted regime-specific flow sensitivities:**

| Regime | Sensitivity (α > 0) | Sensitivity (α < 0) |
|--------|--------------------|--------------------|
| Goldilocks | +1.60 | +0.08 |
| Overheating | +1.71 | −0.40 |
| Downturn | +0.00 | +1.95 |
| Stagflation | −0.50 | +0.18 |

**Output:** `LSEG_download/goldstein_model2_coefficients.pkl` / `.json`

---

## Stage 4 – ELTIF Simulation with Revolving Credit

**File:** `Core_fund_simulation.py`

### Initialisation

| Parameter | Value |
|-----------|-------|
| Initial TNA | €100M |
| Cash buffer | 15% of TNA |
| Credit capacity | 5% of TNA |
| Credit spread | 3% over cash rate |
| TER | 1.5% p.a. |
| Repayment trigger | Excess cash > 10% TNA (min. 2-quarter hold) |
| Alpha window | 8 quarters (rolling 2 years) |

### Quarterly Simulation Loop

For each quarter t = 1 … 80:

1. Apply PM return and deduct quarterly TER → provisional TNA
2. Charge interest on outstanding credit balance
3. Compute rolling alpha vs. bonds and S&P 500 (8-quarter window)
4. Compute flow rate via Goldstein model × current regime coefficients
5. Settle net flow:
   - **Inflow** → add to TNA; repay credit if excess cash > 10% TNA and credit held ≥ 2 quarters
   - **Outflow** → use cash buffer first → draw credit line → forced liquidation (5% haircut) if credit exhausted
6. Record all state variables

### Shortfall Flags

| Flag | Meaning |
|------|---------|
| 0 | No issue |
| 1 | Credit line drawn |
| 2 | Forced liquidation required |

**Full run:** `simulate_eltif_multipaths()` → 10,000 paths → `eltif_results_revolving_credit.csv`

---

## Stage 5 – Metrics & Visualization

**File:** `Core_fund_simulation.py` | **Dashboard:** `streamlit_dashboard.py`

### Metrics computed across all paths

- Credit usage probability (overall and by regime)
- Forced liquidation probability
- Average / maximum shortfall
- Credit interest cost (total and per quarter)
- Final TNA distribution (mean, median, p10, p90)
- Flow–alpha relationship by regime

### Outputs

| Output | Description |
|--------|-------------|
| `eltif_simulation_revolving_credit.png` | 9-panel static matplotlib figure |
| `streamlit_dashboard.py` | Interactive Plotly dashboard (run with `streamlit run streamlit_dashboard.py`) |

---

## Key Files Summary

| File | Stage | Role |
|------|-------|------|
| `Private_Markets_handling.py` | 1 | GLM return unsmoothing |
| `cash_index_handling.py` | 1 | Euribor → quarterly cash returns |
| `Private_Markets_Simulation.py` | 2 | Regime-conditional bootstrap scenario generation |
| `LSEG_download/Fund_Flows_Regression.py` | 3 | Goldstein regression & coefficient estimation |
| `LSEG_download/goldstein_model2_coefficients.pkl` | 3 | Fitted flow model coefficients |
| `Core_fund_simulation.py` | 4–5 | Main ELTIF simulation engine + metrics + matplotlib plots |
| `streamlit_dashboard.py` | 5 | Interactive Streamlit/Plotly dashboard |
| `simulated_pm_cash_returns.csv` | Input to Stage 4 | 10,000 paths × 80 quarters of simulated returns |
| `eltif_results_revolving_credit.csv` | Output of Stage 4 | Full simulation results (path, quarter, TNA, credit, flows, …) |
