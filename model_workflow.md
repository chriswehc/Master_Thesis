# ELTIF Cash Buffer Optimisation — Model Workflow

## Overview

Monte Carlo simulation framework for **European Long-Term Investment Funds (ELTIFs)**
with a revolving credit facility and CRRA-optimal cash buffer selection.

```
Raw Data (PM Indices, Fund Data, Macro)
    → [1] Return Unsmoothing
    → [2] Regime-Conditional Scenario Generation  (EUR benchmarks)
    → [3] Fund Flow Regression  (Goldstein et al. 2017, asset-class variants)
    → [4] ELTIF Simulation with Revolving Credit
    → [4.5] Cash Buffer Optimisation  (BSV 2009 CRRA)
    → [4.6] Sensitivity Grid  (haircut × credit spread)
    → [5] Metrics & Visualisation
```

---

## Stage 1 – Private Market Return Unsmoothing

**File:** `Private_Markets_handling.py`
**Input:** `PM_Indices.xlsx` (Hamilton Lane quarterly indices, USD, 1997–2024)
**Output:** `Hamilton_Lane_unsmoothed_returns.csv`

PM indices exhibit artificial autocorrelation from stale pricing and reporting lags.
The **Getmansky-Lo-Makarov (GLM)** method recovers true underlying returns:

$$\eta_t = \frac{x_t - \sum_{j=1}^{k} \theta_j \cdot \eta_{t-j}}{\theta_0}$$

where $x_t$ is the observed smoothed return, $\eta_t$ the unsmoothed return, and
$\theta_j$ MA weights estimated by MLE (lag order $k$ selected by BIC).

`cash_index_handling.py` converts EUR003 (3-month Euribor) from daily annualised
to quarterly compounded returns → `Cash_quarterly_returns.csv`.

---

## Stage 2 – Regime-Conditional Scenario Generation

**File:** `Private_Markets_Simulation.py`
**Inputs:** `Hamilton_Lane_unsmoothed_returns.csv`, `Cash_quarterly_returns.csv`,
`LSEG_merged/macro_regimes.csv`, `bloomberg_benchmarks.csv`
**Output:** `simulated_pm_cash_returns.csv` (10,000 paths × 80 quarters)

### EUR Conversion

Hamilton Lane indices are in USD. Converted to EUR using FRED DEXUSEU:

$$r^{EUR}_t = \frac{1 + r^{USD}_t}{1 + \Delta FX_t} - 1$$

### Macro Regimes

Four regimes defined by GDP YoY growth × CPI YoY inflation (median splits):

| Regime | Growth | Inflation |
|--------|--------|-----------|
| Goldilocks | High | Low |
| Overheating | High | High |
| Downturn | Low | Low |
| Stagflation | Low | High |

Regime transition probabilities estimated from FRED data; forward paths simulated
via first-order Markov chain. For each of 10,000 paths over 80 quarters:
simulate regime sequence → block-bootstrap one full historical quarter per regime.

### EUR Benchmarks

| Series | Source | Role |
|--------|--------|------|
| EURO STOXX 600 (`SXXR_Index`) | Bloomberg | Equity benchmark |
| Bloomberg EuroAgg Bond (`LBEATREU`) | Bloomberg | Bond benchmark |
| EUR003 Index | ECB / Refinitiv | Risk-free rate $r^f$ |

$$\text{Excess\_STOXX600}_t = r^{STOXX600}_t - \text{EUR003}_t, \qquad
  \text{Excess\_EUR\_Bond}_t = r^{EuroAgg}_t - \text{EUR003}_t$$

**Columns in `simulated_pm_cash_returns.csv`:**
`path, t, Date, Hamilton Lane Private Credit Index (EUR), Hamilton Lane Private Equity Index (EUR),
Hamilton Lane Private Markets Index (EUR), EUR003_Index, STOXX600_Return, EUR_Bond_Return,
EUR_RF, Excess_STOXX600, Excess_EUR_Bond, macro_regime, GDP_YoY, CPI_YoY, avg_pm_return`

---

## Stage 3 – Fund Flow Regression (Goldstein et al. 2017)

**File:** `LSEG_merged/Fund_Flows_Regression.py`
**Runner:** `regression_input_analysis.py`
**Data:** LSEG/Lipper mutual fund panel — ~7,400 funds, 2002–2026, merged from two downloads

### Alpha Calculation

Rolling 8-quarter OLS per fund (min. 6 valid obs out of 8):

$$\text{Excess\_Fund}_t = (r^{gross}_t - \text{TER}_{quarterly}) - \text{RF}_t$$

$$\alpha_t = \hat{\beta}_0 \text{ from: }
\text{Excess\_Fund} \sim \beta_0 + \beta_1 \cdot \text{Excess\_SP500} + \beta_2 \cdot \text{Excess\_Bond}$$

> The regression uses US benchmarks (S&P 500, Vanguard bond, T-bill) following
> Goldstein et al. (2017) exactly. The simulation uses EUR benchmarks (STOXX 600,
> Bloomberg EuroAgg, EUR003). This boundary is documented in the thesis.

### Model 2 — Macro Regime Flow Regression

**Dependent variable:** $\text{Flow\_Rate}_t = \Delta\text{TNA}_t / \text{TNA}_{t-1}$ (net of returns)

$$\text{Flow\_Rate} = \beta_0
  + \alpha^+ \cdot \text{Alpha}
  + \alpha^- \cdot \text{Alpha} \cdot \mathbb{1}[\alpha < 0]
  + \sum_{r \in \mathcal{R}} \!\left[
      \alpha^+_r \cdot \text{Alpha} + \alpha^-_r \cdot \text{Alpha} \cdot \mathbb{1}[\alpha < 0]
    \right] \mathbb{1}[\text{Regime}=r]
  + \beta_{\text{lag}} \cdot F_{t-1}
  + \beta_{\text{TNA}} \cdot \log\text{TNA}_{t-1}
  + \beta_{\text{age}} \cdot \log\text{Age}
  + \beta_{\text{TER}} \cdot \text{TER}
  + \gamma_q + \varepsilon$$

OLS with quarter fixed effects $\gamma_q$; SE clustered at fund level.

Regimes $\mathcal{R} = \{\text{Goldilocks, Overheating, Downturn, Stagflation}\}$
(Fed-regime models removed; macro regime model only).

### Simulation Intercept Adjustment

Raw $\beta_0$ absorbs secular trends; replaced by long-run mean (Ben-David et al. 2022, Zhu 2018):

$$\hat{\beta}^{sim}_0 = \overline{\text{Flow\_Rate}}_{train}$$

$\log(\text{TNA})$ term centred around training mean to avoid level drag:

$$\beta_{\text{TNA}} \cdot \!\left(\log\text{TNA}_t - \overline{\log\text{TNA}}_{train}\right)$$

### Multi-Asset-Class Pipeline

`regression_input_analysis.py` iterates over asset-class subsets:

| Keyword filter | Run tag | N obs (approx.) |
|----------------|---------|-----------------|
| *(all funds)*  | `_all`       | 66,500 |
| `bond`         | `_bond`      | 21,300 |
| `bond usd`     | `_bond_usd`  | 19,700 |
| `corporates`   | `_bond_corp` | 830    |
| `high yield`   | `_bond_hy`   | 2,600  |
| `equity`       | `_equity`    | varies |

Outputs per run → `LSEG_merged/runs/goldstein_model2_coefficients{tag}.pkl`
Comparison table → `Regression_Results/comparison_table.txt` / `.csv`

---

## Stage 4 – ELTIF Simulation with Revolving Credit

**File:** `Core_fund_simulation.py`
**Entry point:** `run_simulation_pipeline.py`

### Default Parameters

| Parameter | Default | Sensitivity range |
|-----------|---------|-------------------|
| Initial TNA | €100M | — |
| Cash weight $w$ | 15% | 5–50% (optimised, 1% steps) |
| Credit capacity | 5% TNA | 5% (tight) / 20% (ample) / 50% (reg_max) |
| Credit spread | 300bps p.a. | 100–500bps |
| TER | 1.5% p.a. | — |
| Haircut rate $h$ | 5% (standalone) / grid: 0–30% | 0, 5, 10, 20, 30% |
| Gate mode | `strict` | `strict` / `economic` |
| Buffer gate fraction | 50% | of target cash weight (strict) |
| Redemption gate | 20% TNA/qtr | economic mode only |
| Alpha window | 8 quarters (min. 6) | — |
| Horizon $T$ | 80 quarters | 20 years (scenario pool) |
| MC paths $N$ | 10,000 | — |

### Rolling Alpha (EUR, per simulation path)

At each quarter $t$, OLS on the last 8 quarters of the current path:

$$\alpha_t = \hat{\beta}_0 \text{ from: }
(r^{PM}_s - \text{EUR003}_s) \sim \beta_0 + \beta_1 \cdot \text{Excess\_STOXX600}_s
+ \beta_2 \cdot \text{Excess\_EUR\_Bond}_s$$

### Blended Return (gross investment income, EUR)

$$\text{BlendedReturn}_t = \underbrace{w \cdot \text{TNA}_{t-1}}_{\text{cash pool}} \cdot r^{cash}_t
  + \underbrace{(1-w) \cdot \text{TNA}_{t-1}}_{\text{PM pool}} \cdot r^{PM}_t$$

### Credit Interest

$$r^{credit}_t = \frac{r^{cash}_t \times 4 + \text{CreditSpread}}{4}, \qquad
\text{CreditInterest}_t = \text{CreditOutstanding}_{t-1} \cdot r^{credit}_t$$

### Fund Flow and Redemption Gate

The gate caps total quarterly outflow. Two modes:

**`strict` (ELTIF 2.0 regulatory):**

$$\text{Flow}_t = \text{clip}\!\left(F(\alpha_t, \text{Regime}_t;\,\hat{\theta}),\;
  -g_{\text{strict}},\; +0.5\right) \cdot \text{TNA}_{t-1}$$

$$g_{\text{strict}} = \text{BufferGateFraction} \times w \quad
  \text{(e.g. } 50\% \times 20\% = 10\%\text{ of TNA)}$$

Gate is anchored to the **declared target weight** $w$, not the actual pool level.

**`economic` (genuine optimisation landscape):**

$$g_{\text{economic}} = \text{RedemptionGatePct} \quad \text{(fixed, e.g. 20\% of TNA)}$$

Investors always redeem fully up to the gate; excess demand spills to credit → PM sale.

### Outflow Waterfall ($\text{Flow}_t < 0$, demand $D = -\text{Flow}_t$)

| Step | Source | Amount drawn |
|------|--------|--------------|
| 1 | Cash pool | $\min(D,\;\text{CashPool}_{t-1})$ |
| 2 | Revolving credit | $\min(\text{rem.},\;\text{Cap} - \text{Outstanding}_{t-1})$ |
| 3 | Forced PM liquidation | remaining demand + haircut friction |

$$\text{Haircut}_t = \text{ForcedSale}_t \times h \quad \text{(friction loss, borne by all investors)}$$

No mandatory same-quarter replenishment. The cash buffer is restored gradually
through the inflow waterfall as new subscriptions arrive.

### Inflow Waterfall ($\text{Flow}_t > 0$)

| Priority | Destination |
|----------|-------------|
| 1 | Restore cash buffer to $w \cdot \text{TNA}^{new}$ |
| 2 | Repay revolving credit |
| 3 | PM pool |

### Investor Return (per-unit, net of all costs)

$$r^{investor}_t = \frac{\text{BlendedReturn}_t - \text{CreditInterest}_t - \text{Haircut}_t}{\text{TNA}_{t-1}}$$

$$W_T = \prod_{t=1}^{T}(1 + r^{investor}_t)$$

---

## Stage 4.5 – Cash Buffer Optimisation (BSV 2009 CRRA)

**File:** `optimize_cash_buffer.py`
**Reference:** Berk, Stanton & Zechner (2009)
**Horizon:** 40 quarters (10 years) — ELTIF standard term

For each cash weight $w \in \mathcal{W} = \{5\%,6\%,\ldots,50\%\}$ (1% steps) and scenario $s$,
simulate $N$ paths and compute expected CRRA utility:

$$\hat{U}_\gamma(w,s) = \frac{1}{N}\sum_{n=1}^{N} \frac{W_T^{(n)\,1-\gamma}}{1-\gamma}
\quad(\gamma \neq 1), \qquad
\hat{U}_1(w,s) = \frac{1}{N}\sum_{n=1}^{N} \log W_T^{(n)}$$

**Optimal cash weight:**

$$w^*(\gamma, s) = \arg\max_{w \in \mathcal{W}}\; \hat{U}_\gamma(w, s)$$

The 138 grid points (46 weights × 3 scenarios) are parallelised across CPU cores
via `multiprocessing.Pool` with the initializer pattern (shared data sent once per
worker process).

### Scenarios

All three scenarios use the same gate mode; they differ only in **credit line capacity**:

| Scenario  | Credit capacity | Interpretation |
|-----------|----------------|----------------|
| `ample`   | 20% of TNA     | Easy replenishment; fund rarely needs to sell PM |
| `tight`   | 5% of TNA      | Constrained credit; fund must sell PM to cover shortfalls |
| `reg_max` | 50% of TNA     | Regulatory maximum credit line (ELTIF 2.0 upper bound) |

### Gate Modes

Run with `--gate_mode strict` or `--gate_mode economic`:

| Mode | Gate formula | Purpose |
|------|-------------|---------|
| `strict` | $g = 50\% \times w$ | ELTIF 2.0 mechanics; finding always-minimum $w^*$ is a valid regulatory result |
| `economic` | $g = \text{RedemptionGatePct}$ (fixed, e.g. 20%) | Genuine optimisation landscape; recovers interior $w^*$ |

### CRRA Grid

$\gamma \in \{1.0, 1.25, 1.5, \ldots, 5.0\}$; $\gamma=2$ is the benchmark (standard macro-finance).

### Output Columns

`optimization_runs/eltif_optimization_frontier{tag}.csv` — one row per (scenario, $w$):

| Column | Description |
|--------|-------------|
| `gate_mode` | `strict` or `economic` |
| `expected_log_growth_ann` | $\mathbb{E}[\log W_T]/T$ annualised |
| `p_fl_any_path` | $P(\geq 1$ forced liquidation over 40 quarters$)$ |
| `expected_shortfall_fl_cond` | $\mathbb{E}[\text{PM forced into liquidation} \mid \text{FL event}]$ in EUR |
| `crra_g{γ}` | Mean CRRA utility at cash weight $w$ |

`optimization_runs/eltif_optimization_optimal{tag}.csv` — one row per (scenario, $\gamma$):
$w^*$, $\mathbb{E}[r_{log}]$, $P(\text{FL})$, median terminal TNA, `gate_mode`.

### Auto-Tag System

Dashboard auto-generates: `{reg\_tag}\_h{haircut\%}\_s{spread\_bps}`
e.g. regression tag `_bond_usd` + haircut 10% + spread 300bps → `_bond_usd_h10_s300`.
Coefficients resolved: `LSEG_merged/runs/` first, then `LSEG_merged/`.

---

## Stage 4.6 – Sensitivity Grid (haircut × credit spread)

**File:** `optimization_grid_analysis.py`

Loops over a haircut × spread grid, calling Stage 4.5 for each cell.

**Recommended grid (5 × 4 = 20 cells):**

|  | 100bps | 200bps | 300bps | 500bps |
|--|--------|--------|--------|--------|
| **0%** | | | | |
| **5%** | | | | |
| **10%** | | | | |
| **20%** | | | | |
| **30%** | | | | |

```bash
# Regulatory result (ELTIF 2.0 mechanics):
python optimization_grid_analysis.py --reg_tag _bond_hy \
    --haircuts "0,5,10,20,30" --spreads "100,200,300,500" \
    --gate_mode strict

# Genuine optimisation landscape:
python optimization_grid_analysis.py --reg_tag _bond_hy \
    --haircuts "0,5,10,20,30" --spreads "100,200,300,500" \
    --gate_mode economic --redemption_gate_pct 0.10
```

**Summary** `optimization_runs/grid_summary{reg_tag}.txt` — four heatmaps at $\gamma=2$:

1. **Optimal $w^*$** — CRRA-optimal cash weight
2. **$\mathbb{E}[r_{log}]$ at $w^*$** — annualised log return at the optimum
3. **Return drag (bps/yr)** — $\left(\mathbb{E}[r_{log}^{w=5\%}] - \mathbb{E}[r_{log}^{w^*}]\right) \times 10{,}000$: bps/yr sacrificed for the buffer vs. minimum 5% allocation
4. **E[haircut cost | FL] % TNA** — $\mathbb{E}[\text{PM forced sold} \mid \text{FL}] \times h \;/\; \text{TNA}_0$: severity of each FL event as % of initial fund size

---

## Stage 5 – Illiquidity Premium & Visualisation

### Illiquidity Premium

Net-of-TER blended return compared to EUR benchmarks:

$$r^{ELTIF}_t(w) = (1-w)\cdot r^{PM}_t + w\cdot r^{cash}_t - \text{TER}_{quarterly}$$

$$\text{IlliqPremium}(w) = 4\cdot\mathbb{E}[\log(1+r^{ELTIF}(w))]
  - 4\cdot\mathbb{E}[\log(1+r^{STOXX600})]$$

As $w\to 1$: premium erodes to $4\cdot\mathbb{E}[\log(1+r^{cash}-\text{TER})]
- 4\cdot\mathbb{E}[\log(1+r^{STOXX600})] < 0$.

### Dashboard Tabs

| Tab | Content |
|-----|---------|
| **Simulation** | Fund flow traces, TNA distribution, credit usage by regime |
| **Optimisation** | 3D scatter ($\gamma$, $w$, $r_{log}$), efficient frontier vs ES, illiquidity premium chart, sensitivity heatmap (metric selector: $w^*$ / return drag / haircut cost %) |
| **Comparison** | Regime $\beta$ coefficients across asset classes; $w^*$ per class |

---

## Folder Structure

```
MT_Python/
├── Private_Markets_handling.py          Stage 1: GLM unsmoothing
├── cash_index_handling.py               Stage 1: EUR003 → quarterly
├── Private_Markets_Simulation.py        Stage 2: EUR scenario generation
├── Core_fund_simulation.py              Stage 4: simulation engine
├── run_simulation_pipeline.py           Stage 4: CLI entry point
├── optimize_cash_buffer.py              Stage 4.5: CRRA grid search (parallel)
├── optimization_grid_analysis.py        Stage 4.6: haircut × spread loop
├── streamlit_dashboard.py               Stage 5: interactive dashboard
├── regression_input_analysis.py         Stage 3 multi-asset runner
│
├── LSEG_merged/                         ~7,400 funds, 604k obs, 2002–2026
│   ├── Fund_Flows_Regression.py
│   ├── returns_merged.csv
│   ├── tna_merged.csv / ter_merged.csv
│   ├── Clean_funds_with_asset_class.csv
│   ├── macro_regimes.csv
│   └── runs/
│       ├── goldstein_model2_coefficients{tag}.pkl
│       ├── model_summary{tag}.csv
│       └── fund_flows_with_macro{tag}.csv
│
├── optimization_runs/
│   ├── eltif_optimization_frontier{tag}.csv
│   ├── eltif_optimization_optimal{tag}.csv
│   ├── grid_summary{reg_tag}.csv
│   └── grid_summary{reg_tag}.txt
│
├── Regression_Results/
│   ├── comparison_table.txt
│   └── comparison_table.csv
│
├── simulated_pm_cash_returns.csv        Input to Stage 4 (10k paths × 80 qtrs)
├── Hamilton_Lane_unsmoothed_returns.csv
├── Cash_quarterly_returns.csv
├── bloomberg_benchmarks.csv            EUR equity + bond (Bloomberg)
├── PM_Indices.xlsx                     Hamilton Lane raw (USD)
└── Liquid_Assets_2.xlsx                EUR003 daily
```

---

## Key Parameters Quick Reference

| Parameter | Symbol | Default | Sensitivity |
|-----------|--------|---------|-------------|
| Cash weight | $w$ | 15% | 5–50% (optimised, 1% steps) |
| Haircut rate | $h$ | 20% | 0–30% |
| Credit spread | — | 300bps | 100–500bps |
| Credit capacity | — | 5% TNA | 5% (tight) / 20% (ample) / 50% (reg_max) |
| TER | — | 1.5% p.a. | fixed |
| CRRA risk aversion | $\gamma$ | 2.0 | 1.0–5.0 |
| Gate mode | — | `strict` | `strict` / `economic` |
| Buffer gate fraction | — | 50% | strict: $g = 50\% \times w$ |
| Redemption gate | — | 20% TNA/qtr | economic mode |
| Optimisation horizon | $T$ | 40 quarters | 10 years |
| Scenario pool | — | 80 quarters | 20 years |
| MC paths (optimisation) | $N$ | 500–1,000 | — |
| MC paths (simulation) | $N$ | 10,000 | — |
