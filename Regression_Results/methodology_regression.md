# Regression Methodology — Extended Notes and Corrections

## Corrected Model Equation

The paragraph contains one specification error and one sign-interpretation error.
The corrected equation (dropping the level indicator) is:

$$
\begin{aligned}
F_{i,t} \;=\;&
  \sum_{m \in \mathcal{M}} \beta_{1,m} \cdot \alpha_{i,t} \cdot \mathbf{1}(s_t = m) \\
&+ \sum_{m \in \mathcal{M}} \beta_{2,m} \cdot \alpha_{i,t}
    \cdot \mathbf{1}(\alpha_{i,t} < 0) \cdot \mathbf{1}(s_t = m) \\
&+ \gamma_1 F_{i,t-1}
 + \gamma_2 \ln\mathrm{TNA}_{i,t-1}
 + \gamma_3 \ln\mathrm{Age}_{i,t}
 + \gamma_4 \mathrm{TER}_i
 + \lambda_t + \varepsilon_{i,t}
\end{aligned}
$$

**What was removed**: the term $\beta_0 \cdot \mathbf{1}(\alpha_{i,t} < 0)$ — a
regime-independent level shift for negative-alpha funds.

---

## Why the Level Indicator is Dropped

The original \citet{Goldstein2017} specification includes a binary indicator
$\mathbf{1}(\alpha < 0)$ to capture a discrete jump in the intercept when a fund
crosses into negative-alpha territory. Retaining it in the regime-interaction model,
however, is asymmetric and difficult to justify.

The argument for omitting the four regime main effects is that each calendar quarter
belongs to exactly one regime, making $\mathbf{1}(s_t = m)$ perfectly collinear with
the quarter fixed effects $\lambda_t$. The same logic applies symmetrically to both
sides of the alpha distribution:

- **Positive alpha**: no level indicator $\mathbf{1}(\alpha > 0)$ is included —
  the regime slopes $\beta_{1,m}$ fully capture the positive-alpha relationship.
- **Negative alpha**: by the same argument, the four piecewise slopes
  $\beta_{2,m}$ span the negative-alpha relationship. A separate
  $\mathbf{1}(\alpha < 0)$ level shift would require an independent economic
  justification — a discrete, regime-invariant jump in flows exactly at $\alpha = 0$
  — which is not supported by theory.

Dropping the level indicator also removes a near-collinearity concern: the sum of
the four $\beta_{2,m}$ interaction terms equals $\alpha \cdot \mathbf{1}(\alpha < 0)$
exactly (since the regimes are exhaustive), which is highly correlated with
$\mathbf{1}(\alpha < 0)$ when $\alpha$ is relatively stable in the negative range.

---

## Piecewise Structure and Coefficient Interpretation

The model is piecewise linear in $\alpha$, with the break point at zero. For a fund
$i$ in regime $m$ in quarter $t$:

$$
\frac{\partial F_{i,t}}{\partial \alpha_{i,t}} =
\begin{cases}
\beta_{1,m} & \text{if } \alpha_{i,t} \geq 0 \\
\beta_{1,m} + \beta_{2,m} & \text{if } \alpha_{i,t} < 0
\end{cases}
$$

- $\beta_{1,m}$: **flow-performance slope for positive-alpha funds in regime $m$**.
  A positive value means outperforming funds attract inflows.

- $\beta_{1,m} + \beta_{2,m}$: **total flow-performance slope for negative-alpha
  funds in regime $m$**. This is the economically relevant quantity for the fragility
  analysis.

- $\beta_{2,m}$: **incremental slope for negative-alpha funds**, i.e., how much the
  slope changes when $\alpha$ crosses zero. It measures the *degree of asymmetry*
  within regime $m$.

### Sign mechanics — why a positive $\beta_{2,m}$ produces outflows

This is the most common source of misinterpretation. The flow *contribution* from
the alpha channel for a negative-alpha fund is:

$$
(\beta_{1,m} + \beta_{2,m}) \times \underbrace{\alpha_{i,t}}_{\displaystyle < 0}
$$

A **positive** total slope $(\beta_{1,m} + \beta_{2,m}) > 0$ multiplied by a
**negative** alpha produces a **negative** flow — i.e., outflows. The larger the
positive total slope, the more strongly the fund is punished for underperforming.

Therefore: **the concavity result is captured by $\beta_{2,m} > 0$** (not negative,
as stated in the original paragraph). A significantly positive $\beta_{2,m}$ means
negative-alpha funds face a steeper outflow response per unit of underperformance
than positive-alpha funds face an inflow response per unit of outperformance.

### Corrected description of concavity

> *A significantly **positive** $\beta_{2,m}$ implies that the total flow-performance
> slope for underperforming funds, $\beta_{1,m} + \beta_{2,m}$, exceeds the slope for
> outperforming funds, $\beta_{1,m}$, generating larger outflows per unit of negative
> alpha than inflows per equivalent unit of positive alpha. This replicates, within a
> single macroeconomic regime, the concavity documented by \citet{Goldstein2017} for
> corporate bond funds.*

---

## Worked Example — Goldilocks Regime (Bond HY)

| Parameter | Estimate |
|---|---|
| $\beta_{1,\text{Goldilocks}}$ | −0.188 (n.s.) |
| $\beta_{2,\text{Goldilocks}}$ | +2.640*** |
| Total slope for $\alpha < 0$ | $-0.188 + 2.640 = +2.452$ |

For a fund with $\alpha = -0.02$ (−2 pp quarterly, i.e. underperforming):

$$
F = 2.452 \times (-0.02) = -0.049
$$

The fund experiences predicted outflows of approximately **4.9% of TNA** from the
alpha channel alone in a Goldilocks quarter.

For a fund with $\alpha = +0.02$ (outperforming):

$$
F = -0.188 \times 0.02 \approx 0
$$

Good performance generates essentially no incremental inflows (coefficient
insignificant, near zero). The asymmetry is stark: underperformance in calm
market conditions generates meaningful redemption pressure, while outperformance
goes largely unrewarded in terms of new subscriptions.

---

## Cross-Regime Variation in Asymmetry

The extension beyond \citet{Goldstein2017} tests whether $\beta_{2,m}$ varies
across regimes — i.e., whether the *degree* of flow-performance asymmetry depends
on macroeconomic conditions.

| Regime | $\beta_{1,m}$ | $\beta_{2,m}$ | Total slope ($\alpha < 0$) | Interpretation |
|---|---|---|---|---|
| Goldilocks | −0.19 (n.s.) | **+2.64*** | **+2.45** | Strong asymmetry in calm times |
| Overheating | −0.63 (n.s.) | +0.52 (n.s.) | −0.11 | Flows insensitive to alpha |
| Downturn | **+3.46** ** | −0.16 (n.s.) | +3.29 | Symmetric: positive alpha attracts inflows |
| Stagflation | +8.03 (n.s.) | −3.87 (n.s.) | +4.16 | Large but imprecise |

The Goldilocks result is the central finding for the simulation: underperforming
funds face significant outflow pressure even in benign macro conditions. Since
Goldilocks is the modal regime in the historical sample, this drives the dominant
source of liquidity stress in the ELTIF cash buffer optimisation.

---

## Identification

The interaction terms $\alpha_{i,t} \cdot \mathbf{1}(s_t = m)$ are identified by
**cross-sectional variation in alpha within each quarter-regime cell**. Since every
observation in a given quarter shares the same regime, the regime indicator itself
is collinear with $\lambda_t$. The slope interactions, however, vary across funds
within the same quarter because $\alpha_{i,t}$ differs by fund. This is the
identification strategy of \citet{Goldstein2017}: the time fixed effects absorb
aggregate shocks and regime-level intercepts, while the interaction terms capture
the within-period flow-performance sensitivity.

---

## Note on Simulation Mapping

In the simulation, the estimated coefficients enter through:

$$
\hat{F}_t = \bar{\mu} + \beta_{1,m} \cdot \alpha_t^+ + (\beta_{1,m} + \beta_{2,m}) \cdot \alpha_t^-
+ \gamma_1 F_{t-1} + \gamma_2 \ln\mathrm{TNA}_{t-1}
$$

where $\alpha_t^+ = \max(\alpha_t, 0)$, $\alpha_t^- = \min(\alpha_t, 0)$, and
$\bar{\mu}$ is the empirical long-run mean quarterly flow rate replacing the raw
OLS intercept (which absorbs secular industry growth trends not applicable
out-of-sample). Age and TER are excluded from the dynamic equation: age effects
are absorbed into $\bar{\mu}$, and TER is already netted from PM returns.
