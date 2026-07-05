import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from datetime import date
from pathlib import Path
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import pickle



### ELTIF SIMULATION WITH REVOLVING CREDIT LINE
### Key features:
### - Credit line stays drawn until repaid
### - Interest charged quarterly on outstanding balance
### - Automatic repayment when excess cash/inflows available
### - Credit spread: 3% 

## LOAD NECCESSARY DATA


# Load simulated PM returns and cash returns with macro regimes
pm_cash_sim_returns = pd.read_csv('simulated_pm_cash_returns.csv')

# Load regression coefficients
def load_goldstein_coefficients(filepath='LSEG_merged/goldstein_model2_coefficients.pkl',
                                verbose=True):
    """Load regression coefficients from fund_flow_analysis"""
    try:
        with open(filepath, 'rb') as f:
            coefficients = pickle.load(f)
        if verbose:
            print(f"✓ Loaded coefficients from: {filepath}")
        return coefficients
    except FileNotFoundError:
        if verbose:
            print(f"❌ File not found: {filepath}")
        raise

try:
    coefficients = load_goldstein_coefficients(verbose=False)
except FileNotFoundError:
    coefficients = None
    # Caller (optimize_cash_buffer.py / run_simulation_pipeline.py) handles warnings.
    # Suppressed here so worker processes don't flood stdout on re-import.


# =============================================================================
# ALPHA CALCULATION
# =============================================================================

def calculate_rolling_alpha(pm_returns_path, ter, window=8):
    """
    Calculate rolling 2-year (8-quarter) alpha
    Benchmarks are pre-aligned by regime in the simulated data
    
    NOTE: Returns are adjusted for TER (net of fees) to match Goldstein methodology
    """
    
    # Extract fund return (average of Hamilton Lane indices)
    pm_cols = [c for c in pm_returns_path.columns
               if 'Hamilton' in c or 'Private' in c]
    
    gross_fund_return = pm_returns_path[pm_cols].mean(axis=1).reset_index(drop=True)
    
    # Subtract TER to get net return (TER is annual, convert to quarterly)
    ter_quarterly = ter / 4
    fund_return = gross_fund_return - ter_quarterly
    
    # EUR benchmarks: EURO STOXX 50 + Euro Agg Bond + EUR003 risk-free
    excess_fund  = fund_return - pm_returns_path['EUR_RF']
    excess_bond  = pm_returns_path['Excess_EUR_Bond']
    excess_sp500 = pm_returns_path['Excess_STOXX600']
    
    # Initialize alpha series
    alphas = pd.Series(np.nan, index=range(len(fund_return)))
    
    # Calculate rolling alpha
    for i in range(window - 1, len(fund_return)):
        y = excess_fund.iloc[i - window + 1 : i + 1]
        X_bond = excess_bond.iloc[i - window + 1 : i + 1]
        X_sp500 = excess_sp500.iloc[i - window + 1 : i + 1]
        
        if y.isnull().any() or X_bond.isnull().any() or X_sp500.isnull().any():
            continue
        
        X = pd.DataFrame({
            'const': 1,
            'Excess_EUR_Bond': X_bond.values,
            'Excess_STOXX600':  X_sp500.values,
        })
        
        try:
            model = sm.OLS(y.values, X).fit()
            alpha = model.params['const']
            alpha = np.clip(alpha, -0.10, 0.10)
            alphas.iloc[i] = alpha
        except:
            continue
    
    alphas.index = pm_returns_path.index[:len(fund_return)]
    return alphas


# =============================================================================
# PM INDEX SELECTION
# =============================================================================

# Maps the user-facing pm_index string to a column-filter lambda.
# Column names in simulated_pm_cash_returns.csv:
#   "Hamilton Lane Private Credit Index"   — senior debt / direct lending
#   "Hamilton Lane Private Equity Index"   — buyout / growth equity
#   "Hamilton Lane Private Markets Index"  — broad composite (PE + PC blend)
#
# 'composite'   → composite index only       (recommended: avoids double-counting)
# 'credit'      → private credit only        (best match for bond-focused ELTIF)
# 'equity'      → private equity only
# 'equal_weight'→ equal average of all three (legacy behaviour, double-counts composite)

_PM_INDEX_MAP = {
    'composite':    lambda cols: [c for c in cols if 'Private Markets' in c],
    'credit':       lambda cols: [c for c in cols if 'Private Credit'  in c],
    'equity':       lambda cols: [c for c in cols if 'Private Equity'  in c],
    'equal_weight': lambda cols: [c for c in cols if 'Hamilton' in c and 'Private' in c],
}


def _select_pm_cols(columns, pm_index='composite'):
    """Return the list of PM return columns for the requested index choice."""
    if pm_index not in _PM_INDEX_MAP:
        raise ValueError(
            f"pm_index='{pm_index}' not recognised. "
            f"Choose from: {list(_PM_INDEX_MAP)}"
        )
    selected = _PM_INDEX_MAP[pm_index](list(columns))
    if not selected:
        raise ValueError(
            f"No columns found for pm_index='{pm_index}'. "
            f"Available columns: {list(columns)}"
        )
    return selected


# =============================================================================
# FLOW CALCULATION
# =============================================================================

def calculate_flow_macro_regime(
    alpha,
    regime,
    lagged_flow,
    tna_lag,
    coefficients,
    cash_weight_target,           # target cash fraction (e.g. 0.20) — used for strict gate
    max_inflow_rate=0.50,         # uncapped inflow ceiling
    buffer_gate_fraction=0.5,     # ELTIF 2.0: gate = this fraction of the TARGET cash weight
    gate_mode='strict',           # 'strict': fixed target-based gate | 'economic': fixed TNA gate
    redemption_gate_pct=0.20,     # economic mode: fixed redemption cap as fraction of TNA
):
    """
    Calculate fund flow using Goldstein model with macro regime interactions.

    gate_mode='strict'  (ELTIF 2.0 regulatory):
        Total outflow capped at buffer_gate_fraction × cash_weight_target.
        E.g. cash_weight=20%, buffer_gate_fraction=50% → gate = 10% of TNA, always.
        Gate is anchored to the declared target weight, not the current actual pool.

    gate_mode='economic' (genuine optimization landscape):
        Total outflow capped at a fixed redemption_gate_pct of TNA (e.g. 20%).
        Full cash pool services the redemption; excess spills to credit → PM sale.
        Investors always redeem fully up to the fixed gate.

    NOTE: TER is NOT included as a parameter because:
    - Alpha is already calculated on NET returns (after fees)
    - TER effect is embedded in the alpha calculation
    - Including TER here would double-count the fee impact
    """

    # Extract coefficients
    regime_coefs = coefficients['macro_regime']['regimes'].get(
        regime, {'alpha_pos': 0, 'alpha_neg': 0, 'flow_level': 0}
    )
    controls = coefficients['macro_regime']['controls']

    # Split alpha into positive and negative components (Goldstein piecewise)
    alpha_pos_val = alpha if alpha >= 0 else 0.0
    alpha_neg_val = alpha if alpha < 0  else 0.0

    # Flow = intercept + regime level shift + piecewise alpha slopes + controls
    flow  = controls['intercept']
    flow += regime_coefs.get('flow_level', 0.0)                              # δ_r: regime intercept shift
    flow += regime_coefs['alpha_pos'] * alpha_pos_val                        # β_α(r) × α⁺
    flow += regime_coefs['alpha_neg'] * alpha_neg_val                        # (β_α + β_αneg)(r) × α⁻
    flow += controls.get('alpha_negative_indicator', 0.0) * (1.0 if alpha < 0 else 0.0)  # β_neg indicator
    flow += controls['lagged_flow'] * lagged_flow
    flow += controls['log_tna'] * np.log(tna_lag) if tna_lag > 0 else 0

    if gate_mode == 'strict':
        # Gate = buffer_gate_fraction × target cash weight (fixed, not dependent on actual pool level)
        # e.g. cash_weight=20%, buffer_gate_fraction=50% → gate = 10% of TNA, always
        fixed_gate = buffer_gate_fraction * cash_weight_target
        flow = np.clip(flow, -fixed_gate, max_inflow_rate)
    else:
        # Economic mode: fixed TNA-based gate; full cash pool available in waterfall
        flow = np.clip(flow, -redemption_gate_pct, max_inflow_rate)

    return flow

# =============================================================================
# SINGLE PATH SIMULATION WITH REVOLVING CREDIT
# =============================================================================


def simulate_fund_with_buffer(
    pm_returns_path,
    coefficients,
    cash_returns=None,
    initial_tna=100_000_000,
    ter=0.015,
    cash_weight=0.15,
    credit_capacity=0.05,
    credit_spread=0.03,           # spread over base cash rate
    haircut_rate=0.05,            # loss on PM assets sold under forced liquidation
    buffer_gate_fraction=0.5,     # ELTIF 2.0: max outflow = this fraction of current cash buffer
    gate_mode='strict',           # 'strict': dynamic gate | 'economic': fixed TNA-based gate
    redemption_gate_pct=0.20,     # economic mode: fixed redemption cap as fraction of TNA
    pm_index='composite',         # which Hamilton Lane index to use as the PM return
):
    """
    Simulate ELTIF fund with explicit cash / PM pools and a revolving credit line.

    Cash and PM are tracked as separate pools with separate returns.

    Outflow waterfall (each quarter):
      1. Cash pool (full pool available)
      2. Revolving credit line (drawn at cash_rate + credit_spread)
      3. Forced PM liquidation (+ haircut on liquidated amount)

    Inflow allocation (each quarter):
      1. Top up cash pool to target buffer (cash_weight × new TNA)
      2. Repay outstanding credit from surplus inflow
      3. Remainder goes to PM pool
    """

    n_quarters = len(pm_returns_path)

    # ── Results DataFrame ─────────────────────────────────────────────────────
    results = pd.DataFrame({
        'Quarter':             range(n_quarters),
        'TNA':                 np.zeros(n_quarters),
        'Cash_Pool':           np.zeros(n_quarters),
        'PM_Pool':             np.zeros(n_quarters),
        'PM_Return':           np.zeros(n_quarters),
        'Alpha':               np.full(n_quarters, np.nan),
        'Flow_Rate':           np.zeros(n_quarters),
        'Flow':                np.zeros(n_quarters),
        'Shortfall':           np.zeros(n_quarters),
        'Shortfall_Flag':      np.zeros(n_quarters, dtype=int),
        'Cash_Used':           np.zeros(n_quarters),
        'Credit_Drawn':        np.zeros(n_quarters),
        'Credit_Repaid':       np.zeros(n_quarters),
        'Credit_Outstanding':  np.zeros(n_quarters),
        'Credit_Interest':     np.zeros(n_quarters),
        'Blended_Return_Amt':    np.zeros(n_quarters),  # gross investment gain (€) before flows/costs
        'Haircut':               np.zeros(n_quarters),  # forced-liquidation friction loss (€)
        'Regime':                [''] * n_quarters
    })

    # ── Pre-compute PM returns ─────────────────────────────────────────────────
    # Alpha is NOT pre-computed here; it is estimated dynamically inside the loop
    # from the blended fund return (cash pool + PM pool weighted by their sizes).
    pm_cols = _select_pm_cols(pm_returns_path.columns, pm_index)
    ter_quarterly           = ter / 4
    gross_pm_return         = pm_returns_path[pm_cols].mean(axis=1).values
    results['PM_Return']    = gross_pm_return - ter_quarterly
    results['Regime']       = pm_returns_path['macro_regime'].values

    # Benchmark series for dynamic rolling-alpha computation (EUR benchmarks)
    _rf_arr    = pm_returns_path['EUR_RF'].reset_index(drop=True).values
    _xbond_arr = pm_returns_path['Excess_EUR_Bond'].reset_index(drop=True).values
    _xsp5_arr  = pm_returns_path['Excess_STOXX600'].reset_index(drop=True).values

    # Rolling-window buffers (max length = ALPHA_WIN)
    _ALPHA_WIN  = 8        # 8 quarters = 2-year window, matching the training regression
    _buf_xfund  = []       # excess blended fund return history
    _buf_xbond  = []
    _buf_xsp5   = []

    # ── Initial state ─────────────────────────────────────────────────────────
    cash_pool = initial_tna * cash_weight
    pm_pool   = initial_tna * (1.0 - cash_weight)
    results.loc[0, 'TNA']      = initial_tna
    results.loc[0, 'Cash_Pool'] = cash_pool
    results.loc[0, 'PM_Pool']   = pm_pool

    lagged_flow        = 0.0
    credit_outstanding = 0.0

    for t in range(1, n_quarters):

        regime             = results.loc[t, 'Regime']
        alpha              = results.loc[t-1, 'Alpha']
        tna_lag            = results.loc[t-1, 'TNA']
        credit_outstanding = results.loc[t-1, 'Credit_Outstanding']

        # ── Cash return this quarter ──────────────────────────────────────────
        if cash_returns is not None and t < len(cash_returns):
            cash_return = cash_returns.iloc[t]
        else:
            cash_return = 0.03 / 4

        # ── Apply separate returns to each pool ───────────────────────────────
        pm_return       = results.loc[t, 'PM_Return']
        cash_pool_prev  = results.loc[t-1, 'Cash_Pool']
        pm_pool_prev_t  = results.loc[t-1, 'PM_Pool']
        cash_pool       = cash_pool_prev  * (1.0 + cash_return)
        pm_pool         = pm_pool_prev_t  * (1.0 + pm_return)

        # Gross investment return (€) for this quarter — before credit costs,
        # haircut, or flows.  Used later to compute per-unit investor return.
        blended_return_amt = (cash_pool_prev * cash_return
                              + pm_pool_prev_t * pm_return)
        results.loc[t, 'Blended_Return_Amt'] = blended_return_amt

        # ── Dynamic rolling alpha from blended fund return ─────────────────────
        # Blended quarterly return = weighted average of cash and PM returns,
        # where weights are the START-of-quarter pool sizes (before flows/rebalancing).
        # This is the total fund performance that investors observe each quarter.
        # We run rolling 8-quarter OLS of excess_blended ~ Excess_Bond + Excess_SP500
        # to obtain alpha — exactly mirroring the training regression but on
        # the simulated fund's actual (blended) return rather than pure PM returns.
        # reuse pool sizes already captured above
        if tna_lag > 0:
            blended_ret = blended_return_amt / tna_lag
        else:
            blended_ret = 0.0

        _buf_xfund.append(blended_ret - float(_rf_arr[t]))
        _buf_xbond.append(float(_xbond_arr[t]))
        _buf_xsp5.append(float(_xsp5_arr[t]))
        if len(_buf_xfund) > _ALPHA_WIN:
            _buf_xfund.pop(0)
            _buf_xbond.pop(0)
            _buf_xsp5.pop(0)

        if len(_buf_xfund) == _ALPHA_WIN:
            try:
                y_w = np.array(_buf_xfund)
                X_w = np.column_stack([np.ones(_ALPHA_WIN), _buf_xbond, _buf_xsp5])
                _a  = float(sm.OLS(y_w, X_w).fit().params[0])
                results.loc[t, 'Alpha'] = np.clip(_a, -0.10, 0.10)
            except Exception:
                pass  # remains np.nan — flow will be set to 0.0 below
        # else: fewer than 8 quarters of history → Alpha stays np.nan

        # ── Pay credit interest from cash pool ────────────────────────────────
        if credit_outstanding > 0:
            annual_credit_rate    = cash_return * 4.0 + credit_spread
            quarterly_credit_rate = annual_credit_rate / 4.0
            credit_interest       = credit_outstanding * quarterly_credit_rate
            cash_pool            -= credit_interest
            results.loc[t, 'Credit_Interest'] = credit_interest

        # ── Investor flow ─────────────────────────────────────────────────────
        if np.isnan(alpha):
            flow_rate = 0.0
        else:
            flow_rate = calculate_flow_macro_regime(
                alpha=alpha,
                regime=regime,
                lagged_flow=lagged_flow,
                tna_lag=tna_lag,
                coefficients=coefficients,
                cash_weight_target=cash_weight,
                buffer_gate_fraction=buffer_gate_fraction,
                gate_mode=gate_mode,
                redemption_gate_pct=redemption_gate_pct,
            )

        results.loc[t, 'Flow_Rate'] = flow_rate
        flow_amount = flow_rate * tna_lag
        results.loc[t, 'Flow'] = flow_amount

        # =================================================================
        # OUTFLOW — liquidity waterfall: Cash → Credit → Forced PM
        # =================================================================
        if flow_amount < 0:
            redemption = abs(flow_amount)

            # Step 1 — Cash pool (full pool available regardless of gate mode)
            cash_used = min(cash_pool, redemption)
            cash_pool -= cash_used
            remaining  = redemption - cash_used
            results.loc[t, 'Cash_Used'] = cash_used

            if remaining > 0:
                # Step 2 — Revolving credit line
                max_credit       = tna_lag * credit_capacity
                credit_available = max_credit - credit_outstanding
                credit_drawn     = min(remaining, max(0.0, credit_available))
                credit_outstanding += credit_drawn
                remaining          -= credit_drawn
                results.loc[t, 'Credit_Drawn'] = credit_drawn

                if credit_drawn > 0:
                    results.loc[t, 'Shortfall_Flag'] = 1

                if remaining > 0:
                    # Step 3 — Forced PM liquidation (+ haircut)
                    pm_pool -= remaining                     # PM sold to meet redemption
                    haircut  = remaining * haircut_rate
                    pm_pool -= haircut                       # haircut: friction loss borne by all investors
                    results.loc[t, 'Shortfall']      = remaining   # PM forced sale only (Credit_Drawn stored separately)
                    results.loc[t, 'Shortfall_Flag'] = 2
                    results.loc[t, 'Haircut']        = haircut


        # =================================================================
        # INFLOW — waterfall: Fill cash buffer → Repay credit → Rest to PM
        # =================================================================
        # Priority order rationale:
        #   1. Cash buffer is the primary liquidity defence — restore it first.
        #      Drawing credit again next quarter (because buffer wasn't replenished)
        #      would incur more interest than a one-quarter delay in repayment.
        #   2. Credit repayment comes from the SURPLUS after the buffer is full.
        #      The repayment flows directly from new subscribers to the bank
        #      (it does NOT pass through cash_pool, avoiding a double-subtraction
        #      artifact that would understate TNA while credit is outstanding).
        #   3. Any remaining surplus goes to the PM pool.
        # =================================================================
        else:
            remaining_inflow = flow_amount

            # Step 1 — Restore cash pool to target buffer weight
            if remaining_inflow > 0:
                new_tna          = cash_pool + pm_pool + remaining_inflow
                target_cash      = cash_weight * new_tna
                cash_addition    = min(remaining_inflow, max(0.0, target_cash - cash_pool))
                cash_pool       += cash_addition
                remaining_inflow -= cash_addition

            # Step 2 — Repay outstanding credit from surplus inflow
            #          Flows directly: new subscribers → bank (bypasses cash_pool)
            if credit_outstanding > 0 and remaining_inflow > 0:
                credit_repayment    = min(credit_outstanding, remaining_inflow)
                credit_outstanding -= credit_repayment
                remaining_inflow   -= credit_repayment
                results.loc[t, 'Credit_Repaid'] = credit_repayment

            # Step 3 — Remaining surplus to PM pool
            if remaining_inflow > 0:
                pm_pool += remaining_inflow

        # =================================================================
        # UPDATE STATE
        # =================================================================
        cash_pool = max(cash_pool, 0.0)
        pm_pool   = max(pm_pool,   0.0)

        results.loc[t, 'Cash_Pool']          = cash_pool
        results.loc[t, 'PM_Pool']            = pm_pool
        results.loc[t, 'TNA']               = cash_pool + pm_pool
        results.loc[t, 'Credit_Outstanding'] = credit_outstanding
        lagged_flow = flow_rate

    return results



# =============================================================================
# MULTI-PATH SIMULATION
# =============================================================================

def simulate_eltif_multipaths(
    sim_pm_returns,
    coefficients,
    cash_returns=None,
    initial_tna=100_000_000,
    ter=0.015,
    cash_weight=0.15,
    credit_capacity=0.05,
    credit_spread=0.03,
    verbose=True,
    haircut_rate=0.05,            # Loss applied to IL assets sold under forced liquidation
    buffer_gate_fraction=0.5,     # ELTIF 2.0: max outflow = this fraction of current cash buffer
    gate_mode='strict',           # 'strict': dynamic gate | 'economic': fixed TNA-based gate
    redemption_gate_pct=0.20,     # economic mode: fixed redemption cap as fraction of TNA
    pm_index='composite',         # which Hamilton Lane index to use as the PM return
):
    """Run ELTIF simulation across all paths"""

    if 'path' in sim_pm_returns.columns:
        path_ids = sim_pm_returns['path'].unique()
        n_paths = len(path_ids)
    else:
        n_paths = 1
        sim_pm_returns['path'] = 0
        path_ids = [0]

    if verbose:
        print("="*80)
        print(f"SIMULATING ELTIF ACROSS {n_paths} PATHS")
        print("="*80)
        print(f"Configuration:")
        print(f"  Initial TNA: €{initial_tna:,.0f}")
        print(f"  Cash Weight: {cash_weight:.1%}")
        print(f"  Credit Capacity: {credit_capacity:.1%}")
        print(f"  Credit Spread: {credit_spread:.2%} over base rate")
        print(f"  TER: {ter:.2%}")
        print(f"  Haircut Rate:       {haircut_rate:.1%}")
        if gate_mode == 'strict':
            print(f"  Gate mode:          strict — {buffer_gate_fraction:.0%} × target weight/qtr (ELTIF 2.0)")
        else:
            print(f"  Gate mode:          economic — fixed {redemption_gate_pct:.0%} of TNA/qtr")
        print(f"  PM Index:           {pm_index}")

    all_results = []

    for i, path_id in enumerate(path_ids):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Path {i+1}/{n_paths} ({100*(i+1)/n_paths:.0f}%)")

        path_data = sim_pm_returns[sim_pm_returns['path'] == path_id].copy()
        path_data = path_data.reset_index(drop=True)

        path_cash_returns = (
            path_data['EUR003_Index'] if 'EUR003_Index' in path_data.columns
            else cash_returns
        )
        path_result = simulate_fund_with_buffer(
            path_data,
            coefficients,
            cash_returns=path_cash_returns,
            initial_tna=initial_tna,
            ter=ter,
            cash_weight=cash_weight,
            credit_capacity=credit_capacity,
            credit_spread=credit_spread,
            haircut_rate=haircut_rate,
            buffer_gate_fraction=buffer_gate_fraction,
            gate_mode=gate_mode,
            redemption_gate_pct=redemption_gate_pct,
            pm_index=pm_index,
        )
        
        path_result['path'] = path_id
        path_result = path_result.set_index(['path', 'Quarter'])
        all_results.append(path_result)
    
    eltif_results = pd.concat(all_results)
    
    if verbose:
        print(f"\n✓ Simulation complete")
        print(f"  Total observations: {len(eltif_results):,}")
    
    return eltif_results


# =============================================================================
# ENHANCED METRICS
# =============================================================================

def calculate_eltif_metrics(eltif_results):
    """Calculate comprehensive metrics including revolving credit stats"""
    
    metrics = {}
    
    # Credit usage metrics
    metrics['credit_usage_probability'] = (eltif_results['Shortfall_Flag'] >= 1).mean()
    metrics['forced_liquidation_probability'] = (eltif_results['Shortfall_Flag'] == 2).mean()
    
    # Shortfall metrics
    shortfalls = eltif_results[eltif_results['Shortfall_Flag'] >= 1]['Shortfall']
    metrics['expected_shortfall'] = shortfalls.mean() if len(shortfalls) > 0 else 0
    metrics['max_shortfall'] = shortfalls.max() if len(shortfalls) > 0 else 0
    
    # Credit statistics
    credit_drawn = eltif_results[eltif_results['Credit_Drawn'] > 0]['Credit_Drawn']
    metrics['avg_credit_drawn'] = credit_drawn.mean() if len(credit_drawn) > 0 else 0
    metrics['max_credit_drawn'] = credit_drawn.max() if len(credit_drawn) > 0 else 0
    
    # Outstanding credit statistics
    metrics['avg_credit_outstanding'] = eltif_results['Credit_Outstanding'].mean()
    metrics['max_credit_outstanding'] = eltif_results['Credit_Outstanding'].max()
    
    # Total credit interest paid
    metrics['total_credit_interest'] = eltif_results['Credit_Interest'].sum()
    metrics['avg_credit_interest_per_quarter'] = eltif_results['Credit_Interest'].mean()
    
    # Final TNA distribution
    final_tna = eltif_results.groupby(level='path')['TNA'].last()
    metrics['final_tna_mean'] = final_tna.mean()
    metrics['final_tna_median'] = final_tna.median()
    metrics['final_tna_p10'] = final_tna.quantile(0.10)
    metrics['final_tna_p90'] = final_tna.quantile(0.90)
    
    # Performance metrics
    metrics['avg_pm_return'] = eltif_results['PM_Return'].mean() * 4
    metrics['avg_flow_rate'] = eltif_results['Flow_Rate'].mean()
    
    # Credit usage by regime
    regime_credit = eltif_results.groupby('Regime')['Shortfall_Flag'].apply(lambda x: (x >= 1).mean())
    metrics['credit_usage_by_regime'] = regime_credit.to_dict()
    
    return metrics


def calculate_optimization_metrics(eltif_results, initial_tna, horizon_years=10):
    """
    Compute metrics for the cash-buffer efficient frontier (BSV 2009 framework).

    Returns per-PATH probabilities (not per path-quarter) so that the X-axis
    of the frontier represents investor-relevant tail-risk: the fraction of
    fund lives that experience at least one forced liquidation event.

    Parameters
    ----------
    eltif_results : pd.DataFrame
        Output of simulate_eltif_multipaths() with MultiIndex (path, Quarter).
    initial_tna : float
        Starting TNA used in the simulation (€).
    horizon_years : int
        Horizon length in years (used only for annualising log-growth).

    Returns
    -------
    dict with keys:
        expected_log_growth       – E[log(W_T)] per-unit investor log return (annualised × horizon)
        expected_log_growth_ann   – annualised per-unit investor log return (÷ horizon_years)
        median_final_tna          – median terminal TNA (€)
        p10_final_tna             – 10th-percentile terminal TNA (€)
        p_fl_any_path             – P(≥1 forced liquidation per path)  ← X-axis
        p_credit_any_path         – P(≥1 credit-line draw per path)
        log_growth_std            – std of per-unit log return across paths
    """
    # Per-unit investor return: (Blended_Return_Amt - Credit_Interest - Haircut) / TNA_{t-1}
    # compounded over the horizon.  Flows (subscriptions/redemptions) do not affect per-unit value.
    # Falls back to log(TNA_T/TNA_0) if Blended_Return_Amt column is absent (legacy).
    if 'Blended_Return_Amt' in eltif_results.columns:
        def _unit_W(df):
            tna_lag    = df['TNA'].shift(1).fillna(initial_tna)
            net_return = (df['Blended_Return_Amt']
                          - df['Credit_Interest']
                          - df['Haircut'])
            r_t = np.where(tna_lag > 0, net_return / tna_lag, 0.0)
            return float(np.prod(1.0 + r_t))

        W_series   = eltif_results.groupby(level='path').apply(_unit_W).clip(lower=1e-6)
        log_growth = np.log(W_series)
    else:
        # Legacy fallback: AUM growth (includes inflow inflation)
        final_tna_leg = eltif_results.groupby(level='path')['TNA'].last().clip(lower=1_000)
        log_growth    = np.log(final_tna_leg / initial_tna)

    # Terminal TNA (for median / p10 reporting — keeps AUM perspective for size metrics)
    final_tna  = eltif_results.groupby(level='path')['TNA'].last().clip(lower=1_000)

    # Per-path flags (True if the path ever hit that threshold)
    fl_any = (eltif_results['Shortfall_Flag'] == 2).groupby(level='path').any()
    cr_any = (eltif_results['Shortfall_Flag'] >= 1).groupby(level='path').any()

    # Per-path total FL shortfall (€) — sum of all forced-liquidation shortfall
    # amounts within a path (0 if no FL event on that path)
    fl_rows = eltif_results[eltif_results['Shortfall_Flag'] == 2]['Shortfall']
    fl_per_path_total = fl_rows.groupby(level='path').sum()
    # Unconditional: average over ALL paths (zeros included)
    fl_shortfall_uncond = fl_per_path_total.reindex(
        eltif_results.index.get_level_values('path').unique(), fill_value=0.0
    ).mean()
    # Conditional: average over paths that had at least one FL event
    fl_shortfall_cond = fl_per_path_total.mean() if len(fl_per_path_total) > 0 else 0.0

    return {
        'expected_log_growth':        float(log_growth.mean()),
        'expected_log_growth_ann':    float(log_growth.mean() / horizon_years),   # per-unit
        'median_final_tna':           float(final_tna.median()),
        'p10_final_tna':              float(final_tna.quantile(0.10)),
        'p_fl_any_path':              float(fl_any.mean()),   # ← X-axis of frontier
        'p_credit_any_path':          float(cr_any.mean()),
        'log_growth_std':             float(log_growth.std()),
        # Shortfall severity (€) — requires re-run to appear in frontier CSV
        'expected_shortfall_fl_uncond': float(fl_shortfall_uncond),  # unconditional mean over all paths
        'expected_shortfall_fl_cond':   float(fl_shortfall_cond),    # conditional on FL occurring
    }


def calculate_crra_utility(eltif_results, initial_tna, gamma=5.0):
    """
    BSV (2009) CRRA utility of terminal wealth.

    Optimal cash weight: w* = argmax (1/N) Σ_paths U_γ(W_T)

    where
        U_γ(W) = W^(1-γ) / (1-γ)   for γ ≠ 1   (power / CRRA utility)
        U_1(W) = log(W)              for γ = 1   (log utility)
        W_T    = TNA_T / TNA_0       (terminal wealth ratio)

    Risk-aversion interpretation
    ----------------------------
        γ = 1  → log utility; equivalent to maximising E[log-growth]
        γ = 2  → standard macro-finance benchmark (Mehra-Prescott 1985)
        γ = 5  → highly risk-averse; ruin paths dominate the objective

    Near-zero wealth handling
    -------------------------
    For γ > 1, U_γ(W → 0) → −∞.  To avoid numerical overflow we clip
    W at 0.01 (i.e. TNA_T ≥ 1% of TNA_0 = €1 M from €100 M).
    At γ = 5:  U_5(0.01) = 0.01^(−4) / (−4) ≈ −2.5×10⁷  — large but finite.
    γ = 10 is excluded from the default grid because 0.01^(−9) ≈ 10¹⁸ risks float overflow.

    Parameters
    ----------
    eltif_results : pd.DataFrame
        MultiIndex (path, Quarter) output of simulate_eltif_multipaths().
    initial_tna : float
        Starting TNA (TNA_0) in EUR.
    gamma : float
        CRRA risk-aversion coefficient.  Must be > 0.

    Returns
    -------
    float
        Sample-average CRRA utility  (1/N) Σ U_γ(W_T)  across all paths.
    """
    # Compute per-unit investor return: tracks what €1 invested at t=0 grows to,
    # net of TER (already in PM_Return), credit costs, and forced-liquidation haircuts.
    # Flows (inflows/outflows by other investors) do NOT affect per-unit value —
    # they merely change the number of units outstanding.
    #
    # r_investor_t = (Blended_Return_Amt_t − Credit_Interest_t − Haircut_t) / TNA_{t-1}
    # W_T_investor = Π_{t=1}^{T} (1 + r_investor_t)
    #
    # Falls back to TNA_T / TNA_0 if the new columns are absent (backward compat).

    if 'Blended_Return_Amt' in eltif_results.columns:
        def _investor_W(df):
            tna_lag    = df['TNA'].shift(1).fillna(initial_tna)
            net_return = (df['Blended_Return_Amt']
                          - df['Credit_Interest']
                          - df['Haircut'])
            r_t = np.where(tna_lag > 0, net_return / tna_lag, 0.0)
            return float(np.prod(1.0 + r_t))

        W_series = eltif_results.groupby(level='path').apply(_investor_W)
    else:
        # Legacy fallback: TNA growth (includes inflow inflation)
        final_tna = eltif_results.groupby(level='path')['TNA'].last()
        W_series  = final_tna / initial_tna

    W = W_series.clip(lower=0.01)   # floor at 1% to keep utility finite for γ > 1

    if abs(gamma - 1.0) < 1e-9:
        return float(np.log(W).mean())
    else:
        return float(((W ** (1.0 - gamma)) / (1.0 - gamma)).mean())


def print_eltif_metrics(metrics):
    """Print comprehensive metrics"""
    
    print("\n" + "="*80)
    print("ELTIF SIMULATION RESULTS - REVOLVING CREDIT LINE")
    print("="*80)
    
    print("\nLIQUIDITY STRESS EVENTS:")
    print(f"  Credit Usage Probability: {metrics['credit_usage_probability']:.2%}")
    print(f"  Forced Liquidation Probability: {metrics['forced_liquidation_probability']:.2%}")
    
    print("\nCREDIT LINE STATISTICS:")
    print(f"  Avg Credit Drawn per Event: €{metrics['avg_credit_drawn']:,.0f}")
    print(f"  Max Credit Drawn: €{metrics['max_credit_drawn']:,.0f}")
    print(f"  Avg Outstanding Balance: €{metrics['avg_credit_outstanding']:,.0f}")
    print(f"  Max Outstanding Balance: €{metrics['max_credit_outstanding']:,.0f}")
    
    print("\nCREDIT COST IMPACT:")
    print(f"  Total Interest Paid (all paths): €{metrics['total_credit_interest']:,.0f}")
    print(f"  Avg Interest per Quarter: €{metrics['avg_credit_interest_per_quarter']:,.0f}")
    
    print("\nSHORTFALL METRICS:")
    print(f"  Expected Shortfall: €{metrics['expected_shortfall']:,.0f}")
    print(f"  Maximum Shortfall: €{metrics['max_shortfall']:,.0f}")
    
    print("\nFINAL TNA DISTRIBUTION:")
    print(f"  Mean:   €{metrics['final_tna_mean']:,.0f}")
    print(f"  Median: €{metrics['final_tna_median']:,.0f}")
    print(f"  10th %: €{metrics['final_tna_p10']:,.0f}")
    print(f"  90th %: €{metrics['final_tna_p90']:,.0f}")
    
    print("\nPERFORMANCE:")
    print(f"  Avg PM Return (ann.): {metrics['avg_pm_return']:.2%}")
    print(f"  Avg Flow Rate: {metrics['avg_flow_rate']:.2%}")
    
    print("\nCREDIT USAGE BY REGIME:")
    for regime, prob in metrics['credit_usage_by_regime'].items():
        print(f"  {regime:15s}: {prob:.2%}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_eltif_results(eltif_results, save_path='eltif_simulation_revolving_credit.png'):
    """Enhanced visualization with credit metrics"""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('ELTIF Simulation Results - Revolving Credit Line', fontsize=16, fontweight='bold')
    
    # 1. TNA Evolution
    ax = axes[0, 0]
    sample_paths = eltif_results.index.get_level_values('path').unique()
    for path in sample_paths:
        path_data = eltif_results.xs(path, level='path')
        ax.plot(path_data.index, path_data['TNA'] / 1e6, alpha=0.15, color='blue', linewidth=0.8)
    ax.set_title('TNA Evolution (all paths)')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('TNA (€M)')
    ax.grid(True, alpha=0.3)
    
    # 2. Credit Usage Probability by Regime
    ax = axes[0, 1]
    regime_credit = eltif_results.groupby('Regime')['Shortfall_Flag'].apply(lambda x: (x >= 1).mean())
    regime_credit.plot(kind='bar', ax=ax, color='crimson')
    ax.set_title('Credit Usage Probability by Regime')
    ax.set_ylabel('Probability')
    ax.set_xlabel('Macro Regime')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Flow Distribution
    ax = axes[0, 2]
    flows = eltif_results['Flow_Rate'].dropna()
    ax.hist(flows.clip(-0.5, 0.5), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Flow Rate Distribution')
    ax.set_xlabel('Flow Rate')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # 4. Credit Outstanding Over Time (all paths fan chart)
    ax = axes[1, 0]
    sample_paths = eltif_results.index.get_level_values('path').unique()
    for path in sample_paths:
        path_data = eltif_results.xs(path, level='path')
        ax.plot(path_data.index, path_data['Credit_Outstanding'] / 1e6,
                alpha=0.15, color='red', linewidth=0.8)
    all_credit = eltif_results.groupby(level='Quarter')['Credit_Outstanding']
    quarters = eltif_results.index.get_level_values('Quarter').unique()
    ax.plot(quarters, all_credit.median() / 1e6, color='darkred', linewidth=4, label='Median')
    ax.plot(quarters, all_credit.quantile(0.90) / 1e6, color='darkred', linewidth=4, linestyle='--', label='p90')
    ax.set_title('Credit Outstanding (all paths)')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Outstanding (€M)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5. Credit Interest Distribution
    ax = axes[1, 1]
    interest = eltif_results[eltif_results['Credit_Interest'] > 0]['Credit_Interest']
    if len(interest) > 0:
        ax.hist(interest, bins=30, color='orange', alpha=0.7, edgecolor='black')
        ax.set_title('Credit Interest Distribution')
        ax.set_xlabel('Interest Paid (€)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Interest Paid', ha='center', va='center', fontsize=14)
        ax.set_title('Credit Interest Distribution')
    
    # 6. Shortfall Size Distribution
    ax = axes[1, 2]
    shortfalls = eltif_results[eltif_results['Shortfall'] > 0]['Shortfall'] / 1e6
    if len(shortfalls) > 0:
        ax.hist(shortfalls, bins=30, color='crimson', alpha=0.7, edgecolor='black')
        ax.set_title('Shortfall Size Distribution')
        ax.set_xlabel('Shortfall (€M)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Shortfalls', ha='center', va='center', fontsize=14)
        ax.set_title('Shortfall Size Distribution')
    
    # 7. Final TNA Distribution
    ax = axes[2, 0]
    final_tna = eltif_results.groupby(level='path')['TNA'].last() / 1e6
    ax.hist(final_tna, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(final_tna.median(), color='red', linestyle='--', linewidth=2, 
               label=f'Median: €{final_tna.median():.0f}M')
    ax.set_title('Final TNA Distribution')
    ax.set_xlabel('Final TNA (€M)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. Flow-Performance Relationship
    ax = axes[2, 1]
    sample_data = eltif_results.sample(min(10000, len(eltif_results)))
    scatter = ax.scatter(sample_data['Alpha'] * 100, 
                        sample_data['Flow_Rate'] * 100,
                        c=sample_data['Regime'].map({'Goldilocks': 0, 'Overheating': 1, 'Downturn': 2, 'Stagflation': 3}),
                        cmap='viridis',
                        alpha=0.3,
                        s=10)
    ax.set_title('Flow-Performance Relationship')
    ax.set_xlabel('Alpha (%)')
    ax.set_ylabel('Flow Rate (%)')
    ax.grid(True, alpha=0.3)
    
    # 9. Cumulative Credit Cost (all paths fan chart)
    ax = axes[2, 2]
    sample_paths = eltif_results.index.get_level_values('path').unique()
    for path in sample_paths:
        path_data = eltif_results.xs(path, level='path')
        cumulative_interest = path_data['Credit_Interest'].cumsum()
        ax.plot(path_data.index, cumulative_interest / 1e6,
                alpha=0.15, color='darkred', linewidth=0.8)
    all_cumcost = eltif_results.groupby(level='path')['Credit_Interest'].cumsum()
    all_cumcost_df = all_cumcost.rename('CumCost').to_frame()
    all_cumcost_df.index = eltif_results.index
    quarters = eltif_results.index.get_level_values('Quarter').unique()
    p50_cost = all_cumcost_df.groupby(level='Quarter')['CumCost'].median() / 1e6
    p90_cost = all_cumcost_df.groupby(level='Quarter')['CumCost'].quantile(0.90) / 1e6
    ax.plot(quarters, p50_cost, color='darkred', linewidth=4, label='Median')
    ax.plot(quarters, p90_cost, color='darkred', linewidth=4, linestyle='--', label='p90')
    ax.set_title('Cumulative Credit Cost (all paths)')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Cumulative Interest (€M)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {save_path}")
    plt.show()


# =============================================================================
# RUN SIMULATION
# =============================================================================

if __name__ == "__main__":
    # Run simulation
    eltif_results = simulate_eltif_multipaths(
        pm_cash_sim_returns,
        coefficients,
        cash_returns=None,   # per-path EUR003_Index extracted inside simulate_eltif_multipaths
        initial_tna=100_000_000,
        cash_weight=0.15,
        credit_capacity=0.05,
        credit_spread=0.03  # 3% spread
    )

    # Calculate and print metrics
    metrics = calculate_eltif_metrics(eltif_results)
    print_eltif_metrics(metrics)

    # Visualize
    plot_eltif_results(eltif_results)

    # Save results
    eltif_results.to_csv('eltif_results_revolving_credit.csv')
    print("\n✓ Results saved to: eltif_results_revolving_credit.csv")