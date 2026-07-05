# Regression Specification: Interpretation Guide

## Model

$$
\text{Flow}_{i,t} = \sum_{r} \beta_r \cdot \alpha_{i,t} \cdot \mathbf{1}[\text{regime}_t = r]
+ \sum_{r} \gamma_r \cdot \alpha_{i,t} \cdot \mathbf{1}[\alpha_{i,t} < 0] \cdot \mathbf{1}[\text{regime}_t = r]
+ \delta' \mathbf{X}_{i,t} + \mu_t + \varepsilon_{i,t}
$$

where regimes $r \in \{\text{Goldilocks, Overheating, Downturn, Stagflation}\}$,
$\mu_t$ are quarter fixed effects, and $\mathbf{X}_{i,t}$ are fund-level controls.

---

## Regressors

| Statsmodels name | Mathematical object | Non-zero when |
|---|---|---|
| `Alpha_x_Goldilocks` | $\alpha_{i,t} \cdot \mathbf{1}[G_t]$ | Fund is in Goldilocks quarter |
| `Alpha_x_Overheating` | $\alpha_{i,t} \cdot \mathbf{1}[O_t]$ | Fund is in Overheating quarter |
| `Alpha_x_Downturn` | $\alpha_{i,t} \cdot \mathbf{1}[D_t]$ | Fund is in Downturn quarter |
| `Alpha_x_Stagflation` | $\alpha_{i,t} \cdot \mathbf{1}[S_t]$ | Fund is in Stagflation quarter |
| `AlphaNeg_x_Goldilocks` | $\alpha_{i,t} \cdot \mathbf{1}[\alpha < 0] \cdot \mathbf{1}[G_t]$ | Negative-alpha fund in Goldilocks |
| `AlphaNeg_x_Overheating` | $\alpha_{i,t} \cdot \mathbf{1}[\alpha < 0] \cdot \mathbf{1}[O_t]$ | Negative-alpha fund in Overheating |
| `AlphaNeg_x_Downturn` | $\alpha_{i,t} \cdot \mathbf{1}[\alpha < 0] \cdot \mathbf{1}[D_t]$ | Negative-alpha fund in Downturn |
| `AlphaNeg_x_Stagflation` | $\alpha_{i,t} \cdot \mathbf{1}[\alpha < 0] \cdot \mathbf{1}[S_t]$ | Negative-alpha fund in Stagflation |

---

## Design choices

### Why no baseline `Alpha` term?

The four `Alpha_x_{regime}` terms sum to $\alpha_{i,t}$ exactly (since the regimes are
mutually exclusive and exhaustive for every observed quarter):

$$
\alpha \cdot \mathbf{1}[G] + \alpha \cdot \mathbf{1}[O] + \alpha \cdot \mathbf{1}[D] + \alpha \cdot \mathbf{1}[S] = \alpha
$$

Including a standalone `Alpha` alongside all four interactions would create perfect
multicollinearity. Dropping the baseline gives each regime its own direct flow-performance
slope rather than a differential from an arbitrary reference regime.

### Why no `Alpha_Negative = I(α < 0)` level indicator?

By the same logic, the four `AlphaNeg_x_{regime}` terms sum to
$\alpha \cdot \mathbf{1}[\alpha < 0]$ (the piecewise slope), so including a separate
level indicator $\mathbf{1}[\alpha < 0]$ would be asymmetric with the treatment of
positive alpha and would introduce near-collinearity without a distinct economic
justification.

### Why no regime dummies?

Quarter fixed effects $\mu_t$ perfectly subsume regime main effects: each calendar
quarter belongs to exactly one regime, so $\mathbf{1}[\text{regime}_t = r]$ is a linear
combination of the quarter dummies. Including both would cause exact collinearity.
All regime-level intercept differences are absorbed by $\mu_t$.

---

## How to read a coefficient

### Positive-alpha fund in regime $r$

$$
\frac{\partial \text{Flow}_{i,t}}{\partial \alpha_{i,t}} \bigg|_{\alpha > 0,\, \text{regime}=r} = \beta_r
$$

A one-unit increase in $\alpha$ changes the quarterly flow rate by $\beta_r$ percentage
points of TNA. For example, $\beta_{\text{Downturn}} = 3.46$ means an additional 1 pp of
quarterly alpha raises the flow rate by 3.46 pp in a recession.

### Negative-alpha fund in regime $r$

$$
\frac{\partial \text{Flow}_{i,t}}{\partial \alpha_{i,t}} \bigg|_{\alpha < 0,\, \text{regime}=r} = \beta_r + \gamma_r
$$

The **total effective slope** is $\beta_r + \gamma_r$. The sign of the flow impact then
depends on the sign of $\alpha$:

$$
\text{Flow contribution} = (\beta_r + \gamma_r) \times \underbrace{\alpha}_{\displaystyle < 0}
$$

A **positive** total slope $(\beta_r + \gamma_r) > 0$ combined with negative alpha produces
**negative flows (outflows)**. The larger the total slope, the more sensitive
outflows are to the magnitude of underperformance.

### Worked example — Goldilocks, Bond HY

| | Value |
|---|---|
| $\beta_{\text{Goldilocks}}$ | −0.188 (n.s.) |
| $\gamma_{\text{Goldilocks}}$ | +2.640*** |
| Total slope for $\alpha < 0$ | −0.188 + 2.640 = **+2.452** |

For a fund with $\alpha = -0.02$ (−2 pp quarterly alpha) in Goldilocks:

$$
\text{Flow} = 2.452 \times (-0.02) = -0.049
$$

The fund experiences outflows of approximately **4.9% of TNA** in that quarter from
the alpha channel alone.

**Contrast with positive-alpha fund** ($\alpha = +0.02$):

$$
\text{Flow} = -0.188 \times 0.02 = -0.004 \approx 0 \text{ (n.s.)}
$$

Good performance generates essentially no inflows. This **asymmetry** — investors punish
underperformers but do not reward outperformers — is the bond fund concavity result
of Goldstein, Jiang & Ng (2017).

---

## Summary of effective slopes by regime (Bond HY, baseline specification)

| Regime | Slope ($\alpha > 0$) | Slope ($\alpha < 0$) | Interpretation |
|---|---|---|---|
| Goldilocks | −0.19 (n.s.) | **+2.45*** | Asymmetric: only underperformers punished |
| Overheating | −0.63 (n.s.) | −0.11 (n.s.) | Flows insensitive to alpha |
| Downturn | **+3.46** | **+3.29** | Both directions significant; outperformers attract inflows |
| Stagflation | +8.03 (n.s.) | +4.16 (n.s.) | Large but imprecise (few stagflation quarters) |

*Effective slope for $\alpha < 0$ = $\beta_r + \gamma_r$. Flow impact = slope × $\alpha$.*

---

## Controls

| Variable | Expected sign | Interpretation |
|---|---|---|
| `Lagged_Flow` | + | Flow momentum / persistence (~0.35) |
| `Log_TNA_lag` | − | Diseconomies of scale; large funds face flow headwinds |
| `Log_Age` | − | Lifecycle effect; older funds attract fewer flows |
| `TER` | − | Higher fees reduce net flows |
| `C(YearQuarter)` | — | Quarter FEs absorb aggregate industry trends, macro level effects, and regime main effects |

---

## Estimation details

- **Estimator**: OLS
- **Standard errors**: Clustered at fund level (following Goldstein et al. 2017).
  Quarter FEs absorb common time-period shocks, making the additional time dimension
  of two-way clustering largely redundant.
- **Sample**: US mutual funds, Bond High Yield category, $\geq$ \$1M TNA
- **Alpha**: 8-quarter rolling 2-factor OLS (excess return on excess equity + bond market),
  minimum 6 non-missing quarters. Contemporaneous specification (main); 1-quarter lag
  available as robustness.
- **Winsorisation**: Flow rate and alpha winsorised at 1% tails (Goldstein et al. 2017)

---

## Mapping to the simulation (`Core_fund_simulation.py`)

Each quarter the simulation calls `calculate_flow_macro_regime()`, which applies:

$$
\hat{\text{Flow}}_{t} = \bar{\mu} + \beta_r \cdot \alpha_t + \gamma_r \cdot \alpha_t \cdot \mathbf{1}[\alpha_t < 0]
+ \delta_{\text{flow}} \cdot \text{Flow}_{t-1} + \delta_{\text{tna}} \cdot \log(\text{TNA}_{t-1})
$$

where $\bar{\mu}$ is the empirical long-run mean flow rate (replacing the raw OLS intercept).

**Controls omitted from the dynamic equation:**
- `Log_Age`: lifecycle drag absorbed into $\bar{\mu}$ (estimated from training data including age effects)
- `TER`: already netted from PM returns in the simulation; including it in the flow equation would double-count the fee impact

**Gate application** (applied after the regression-predicted flow):
- *Strict mode*: flow clipped to $[-\delta \cdot w^*, +0.5]$ where $\delta$ = buffer gate fraction (ELTIF 2.0: 50%)
- *Economic mode*: flow clipped to $[-g, +0.5]$ where $g$ = fixed TNA-based redemption cap
