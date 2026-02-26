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


### TODO:
# - Add cash flow waterfall
# - Goldstein regression filter for bond asset classes
# - Download correct benchmark indices
# - Implement optimisation / simulation over multiple starting PM weights

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
def load_goldstein_coefficients(filepath='LSEG_download/goldstein_model2_coefficients.pkl'):
    """Load regression coefficients from fund_flow_analysis"""
    
    try:
        with open(filepath, 'rb') as f:
            coefficients = pickle.load(f)
        print(f"✓ Loaded coefficients from: {filepath}")
        return coefficients
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        raise

coefficients = load_goldstein_coefficients()


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
    
    # Benchmarks are already in the dataframe
    excess_fund = fund_return - pm_returns_path['RF']
    excess_bond = pm_returns_path['Excess_Bond']
    excess_sp500 = pm_returns_path['Excess_SP500']
    
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
            'Excess_Bond': X_bond.values,
            'Excess_SP500': X_sp500.values
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
# FLOW CALCULATION
# =============================================================================

def calculate_flow_macro_regime(
    alpha,
    regime,
    lagged_flow,
    tna_lag,
    coefficients,
    max_flow_rate=0.5,
):
    """
    Calculate fund flow using Goldstein model with macro regime interactions
    
    NOTE: TER is NOT included as a parameter because:
    - Alpha is already calculated on NET returns (after fees)
    - TER effect is embedded in the alpha calculation
    - Including TER here would double-count the fee impact
    """
    
    # Extract coefficients
    baseline = coefficients['macro_regime']['baseline']
    regime_coefs = coefficients['macro_regime']['regimes'].get(regime, {'alpha': 0, 'alpha_neg': 0})
    controls = coefficients['macro_regime']['controls']
    
    # Calculate components
    alpha_negative = 1 if alpha < 0 else 0
    alpha_x_negative = alpha * alpha_negative
    
    # BASELINE EFFECTS
    flow = controls['intercept']
    flow += baseline['alpha'] * alpha
    flow += baseline['alpha_negative'] * alpha_negative
    flow += baseline['alpha_x_negative'] * alpha_x_negative
    
    # REGIME-SPECIFIC INTERACTIONS
    flow += regime_coefs['alpha'] * alpha
    flow += regime_coefs['alpha_neg'] * alpha_x_negative
    
    # CONTROLS (without TER - already in net returns)
    flow += controls['lagged_flow'] * lagged_flow
    flow += controls['log_tna'] * np.log(tna_lag) if tna_lag > 0 else 0

    # Max redemption limit
    flow = np.clip(flow, -max_flow_rate, max_flow_rate)

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
    credit_spread=0.03,  # 3% spread over base rate
    repayment_threshold=0.10  # Only repay when excess cash > 10% of TNA
):
    """
    Simulate ELTIF fund with REVOLVING credit line
    
    Credit line:
    - Draws when cash insufficient
    - Stays outstanding until STRATEGIC repayment
    - Charges interest quarterly on outstanding balance
    - Repays only when significant excess cash available (>10% TNA)
    
    This prevents immediate repayment and shows realistic credit persistence
    """

    n_quarters = len(pm_returns_path)

    # Initialize results with credit tracking
    results = pd.DataFrame({
        'Quarter': range(n_quarters),
        'TNA': np.zeros(n_quarters),
        'PM_Return': np.zeros(n_quarters),
        'Alpha': np.zeros(n_quarters),
        'Flow_Rate': np.zeros(n_quarters),
        'Flow': np.zeros(n_quarters),
        'Shortfall': np.zeros(n_quarters),
        'Shortfall_Flag': np.zeros(n_quarters, dtype=int),
        'Cash_Used': np.zeros(n_quarters),
        'Credit_Drawn': np.zeros(n_quarters),
        'Credit_Repaid': np.zeros(n_quarters),
        'Credit_Outstanding': np.zeros(n_quarters),
        'Credit_Interest': np.zeros(n_quarters),
        'Regime': [''] * n_quarters
    })

    # Initial state
    results.loc[0, 'TNA'] = initial_tna

    # Select ONLY Hamilton Lane PM indices
    pm_cols = [c for c in pm_returns_path.columns 
               if 'Hamilton' in c and 'Private' in c]
    
    # Calculate net PM return (after TER)
    ter_quarterly = ter / 4
    gross_pm_return = pm_returns_path[pm_cols].mean(axis=1).values
    results['PM_Return'] = gross_pm_return - ter_quarterly
    results['Regime'] = pm_returns_path['macro_regime'].values

    # Rolling alpha
    results['Alpha'] = calculate_rolling_alpha(pm_returns_path, ter, window=8).values

    lagged_flow = 0
    credit_outstanding = 0

    for t in range(1, n_quarters):

        regime = results.loc[t, 'Regime']
        alpha = results.loc[t-1, 'Alpha']
        tna_lag = results.loc[t-1, 'TNA']
        credit_outstanding = results.loc[t-1, 'Credit_Outstanding']

        # Cash return
        if cash_returns is not None and t < len(cash_returns):
            cash_return = cash_returns.iloc[t]
        else:
            cash_return = 0.03 / 4

        # Calculate flow
        if np.isnan(alpha):
            flow_rate = 0
        else:
            flow_rate = calculate_flow_macro_regime(
                alpha=alpha,
                regime=regime,
                lagged_flow=lagged_flow,
                tna_lag=tna_lag,
                coefficients=coefficients
            )

        results.loc[t, 'Flow_Rate'] = flow_rate
        flow_amount = flow_rate * tna_lag
        results.loc[t, 'Flow'] = flow_amount

        # TNA after PM return
        pm_return = results.loc[t, 'PM_Return']
        tna_after_return = tna_lag * (1 + pm_return)

        # =================================================================
        # PAY INTEREST ON OUTSTANDING CREDIT FIRST
        # =================================================================
        
        # Interest is charged at the BEGINNING of the quarter on previous balance
        if credit_outstanding > 0:
            # Annualized rate = cash rate + spread
            annual_credit_rate = cash_return * 4 + credit_spread
            quarterly_credit_rate = annual_credit_rate / 4
            
            credit_interest = credit_outstanding * quarterly_credit_rate
            results.loc[t, 'Credit_Interest'] = credit_interest
            
            # Interest reduces TNA immediately
            tna_after_return -= credit_interest
        
        # =================================================================
        # REVOLVING CREDIT LINE LOGIC
        # =================================================================
        
        if flow_amount < 0:
            # ============================================================
            # OUTFLOW - Need to fund redemption
            # ============================================================
            redemption = abs(flow_amount)
            cash_available = tna_lag * cash_weight * (1 + cash_return)
            
            if redemption <= cash_available:
                # ✅ Cash sufficient - use cash only
                results.loc[t, 'Cash_Used'] = redemption
                excess_cash = cash_available - redemption
                
                # ⭐ STRATEGIC REPAYMENT: Only repay if SIGNIFICANT excess
                repayment_trigger = tna_lag * repayment_threshold
                
                if credit_outstanding > 0 and excess_cash > repayment_trigger:
                    # Repay 50% of excess cash above threshold
                    repayable_amount = (excess_cash - repayment_trigger) * 0.5
                    credit_repayment = min(credit_outstanding, repayable_amount)
                    credit_outstanding -= credit_repayment
                    results.loc[t, 'Credit_Repaid'] = credit_repayment
                
                tna_after_flow = tna_after_return - redemption
                
            else:
                # ⚠️ Cash insufficient - need credit line
                results.loc[t, 'Cash_Used'] = cash_available
                remaining = redemption - cash_available
                
                max_credit = tna_lag * credit_capacity
                credit_available = max_credit - credit_outstanding
                
                if remaining <= credit_available:
                    # ✅ Credit line sufficient
                    credit_outstanding += remaining
                    results.loc[t, 'Credit_Drawn'] = remaining
                    results.loc[t, 'Shortfall'] = remaining
                    results.loc[t, 'Shortfall_Flag'] = 1
                    
                    tna_after_flow = tna_after_return - redemption
                    
                else:
                    # 🚨 SEVERE SHORTFALL - Forced liquidation
                    credit_drawn = credit_available
                    credit_outstanding += credit_drawn
                    results.loc[t, 'Credit_Drawn'] = credit_drawn
                    
                    forced_liquidation = remaining - credit_available
                    results.loc[t, 'Shortfall'] = remaining
                    results.loc[t, 'Shortfall_Flag'] = 2
                    
                    haircut = forced_liquidation * 0.05
                    
                    tna_after_flow = tna_after_return - redemption - haircut
        
        else:
            # ============================================================
            # INFLOW - Add to TNA
            # ============================================================
            tna_after_flow = tna_after_return + flow_amount
            
            # Cash BEFORE inflow (grown from previous quarter)
            cash_before_inflow = tna_lag * cash_weight * (1 + cash_return)

            # Total cash after adding the inflow
            total_cash_after_inflow = cash_before_inflow + flow_amount

            # Safety buffer we always keep (10% of TNA)
            safety_buffer = tna_after_flow * 0.10
            
            # Only repay credit after a minimum 2-quarter holding period
            # (prevents immediate repayment in the very next quarter after drawing)
            quarters_with_credit = (results.loc[1:t-1, 'Credit_Outstanding'] > 0).sum()
            if credit_outstanding > 0 and quarters_with_credit >= 2:
                excess_cash = total_cash_after_inflow - safety_buffer
                if excess_cash > 0:
                    credit_repayment = min(credit_outstanding, excess_cash)
                    credit_outstanding -= credit_repayment
                    results.loc[t, 'Credit_Repaid'] = credit_repayment
                    tna_after_flow -= credit_repayment
        
        # =================================================================
        # UPDATE STATE
        # =================================================================
        
        results.loc[t, 'Credit_Outstanding'] = credit_outstanding
        results.loc[t, 'TNA'] = max(tna_after_flow, 0)
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
    verbose=True
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
    
    all_results = []
    
    for i, path_id in enumerate(path_ids):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Path {i+1}/{n_paths} ({100*(i+1)/n_paths:.0f}%)")
        
        path_data = sim_pm_returns[sim_pm_returns['path'] == path_id].copy()
        path_data = path_data.reset_index(drop=True)
        
        path_result = simulate_fund_with_buffer(
            path_data,
            coefficients,
            cash_returns=cash_returns,
            initial_tna=initial_tna,
            ter=ter,
            cash_weight=cash_weight,
            credit_capacity=credit_capacity,
            credit_spread=credit_spread
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
    ax.plot(quarters, all_credit.median() / 1e6, color='darkred', linewidth=2.5, label='Median')
    ax.plot(quarters, all_credit.quantile(0.90) / 1e6, color='darkred', linewidth=2.5, linestyle='--', label='p90')
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
    ax.plot(quarters, p50_cost, color='darkred', linewidth=2.5, label='Median')
    ax.plot(quarters, p90_cost, color='darkred', linewidth=2.5, linestyle='--', label='p90')
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
        cash_returns=pm_cash_sim_returns['EUR003_Index'],
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