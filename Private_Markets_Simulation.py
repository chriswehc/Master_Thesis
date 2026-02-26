# %%
"""
COMPLETE PM + BENCHMARK SIMULATION WITH MACRO REGIMES
======================================================

This script:
1. Loads PM returns, cash returns, and benchmark returns
2. Merges with macro regimes
3. Bootstraps ALL returns together by regime (maintains correlations)
4. Outputs simulated data ready for ELTIF simulation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# =============================================================================
# STEP 1: LOAD ALL DATA
# =============================================================================

print("="*80)
print("STEP 1: LOADING DATA")
print("="*80)

# PM returns (unsmoothed)
pm_returns = pd.read_csv('Hamilton_Lane_unsmoothed_returns.csv', index_col=0, parse_dates=True)
print(f"\n✓ Loaded PM returns: {pm_returns.shape}")

# Cash returns
cash_returns = pd.read_csv('Cash_quarterly_returns.csv', index_col=0, parse_dates=True)
print(f"✓ Loaded cash returns: {cash_returns.shape}")

# Benchmark returns (for alpha calculation)
benchmark_returns = pd.read_csv('benchmarks.csv', index_col=0, parse_dates=True)
print(f"✓ Loaded benchmark returns: {benchmark_returns.shape}")

# Macro regimes
macro_regimes = pd.read_csv('LSEG_download/macro_regimes.csv', parse_dates=['Date'])
print(f"✓ Loaded macro regimes: {macro_regimes.shape}")

# %%
# =============================================================================
# STEP 2: MERGE EVERYTHING TOGETHER
# =============================================================================

print("\n" + "="*80)
print("STEP 2: MERGING ALL DATA WITH MACRO REGIMES")
print("="*80)

# Reset all indices to have Date as column
pm_with_date = pm_returns.reset_index().rename(columns={'index': 'Date'})
pm_with_date['Date'] = pd.to_datetime(pm_with_date['Date'])

cash_with_date = cash_returns.reset_index().rename(columns={'index': 'Date'})
cash_with_date['Date'] = pd.to_datetime(cash_with_date['Date'])

bench_with_date = benchmark_returns.reset_index().rename(columns={'index': 'Date'})
bench_with_date['Date'] = pd.to_datetime(bench_with_date['Date'])

macro_regimes['Date'] = pd.to_datetime(macro_regimes['Date'])

# Convert to quarters for matching
pm_with_date['Quarter'] = pm_with_date['Date'].dt.to_period('Q')
cash_with_date['Quarter'] = cash_with_date['Date'].dt.to_period('Q')
bench_with_date['Quarter'] = bench_with_date['Date'].dt.to_period('Q')
macro_regimes['Quarter'] = macro_regimes['Date'].dt.to_period('Q')

# For each quarter, take the LAST macro regime observation
macro_quarterly = macro_regimes.sort_values('Date').groupby('Quarter').last().reset_index()

# Merge PM + Cash + Benchmarks + Macro
# Start with PM returns
combined_data = pm_with_date.copy()

# Add cash returns
combined_data = combined_data.merge(
    cash_with_date[['Quarter', 'EUR003_Index']],
    on='Quarter',
    how='left'
)

# Add benchmark returns
bench_cols = ['SP500_Return', 'Bond_Market_Return', 'RF', 'Excess_SP500', 'Excess_Bond']
combined_data = combined_data.merge(
    bench_with_date[['Quarter'] + bench_cols],
    on='Quarter',
    how='left'
)

# Add macro regimes
combined_data = combined_data.merge(
    macro_quarterly[['Quarter', 'macro_regime', 'GDP_YoY', 'CPI_YoY', 'FedFundsRate']],
    on='Quarter',
    how='left'
)

# Drop Quarter column and rows without regime
combined_data = combined_data.drop(columns=['Quarter'])
combined_data = combined_data.dropna(subset=['macro_regime'])

print(f"\n✓ Combined data shape: {combined_data.shape}")
print(f"  Date range: {combined_data['Date'].min()} to {combined_data['Date'].max()}")
print(f"\nRegime distribution:")
print(combined_data['macro_regime'].value_counts())

# Save combined data
combined_data.to_csv('combined_pm_cash_benchmark_macro.csv', index=False)
print("\n✓ Saved to: combined_pm_cash_benchmark_macro.csv")

# %%
# =============================================================================
# STEP 3: CALCULATE TRANSITION MATRIX
# =============================================================================

print("\n" + "="*80)
print("STEP 3: CALCULATING REGIME TRANSITION MATRIX")
print("="*80)

combined_sorted = combined_data.sort_values('Date').copy()

current_regime = combined_sorted['macro_regime']
next_regime = current_regime.shift(-1)

transition_matrix = pd.crosstab(
    current_regime,
    next_regime,
    normalize='index'
)

print("\n✓ Transition Matrix:")
print(transition_matrix.round(3))

print("\n" + "-"*80)
print("REGIME PERSISTENCE METRICS")
print("-"*80)

for regime in transition_matrix.index:
    stay_prob = transition_matrix.loc[regime, regime] if regime in transition_matrix.columns else 0
    avg_duration = 1 / (1 - stay_prob) if stay_prob < 1 else np.inf
    print(f"{regime:15s}: Stay prob = {stay_prob:.1%}, Avg duration = {avg_duration:.1f} quarters")

transition_matrix.to_csv('macro_transition_matrix.csv')
print("\n✓ Saved to: macro_transition_matrix.csv")

# %%
# =============================================================================
# STEP 4: SIMULATION FUNCTIONS
# =============================================================================

def simulate_regime_path(transition_matrix, n_quarters, start_regime='Goldilocks', seed=0):
    """
    Simulate a sequence of macro regimes using transition matrix
    """
    rng = np.random.default_rng(seed)
    regimes = transition_matrix.index.tolist()
    
    if start_regime not in regimes:
        start_regime = regimes[0]
    
    path = [start_regime]
    current_regime = start_regime
    
    for _ in range(n_quarters - 1):
        if current_regime in transition_matrix.index:
            probs = transition_matrix.loc[current_regime].values
            if not np.isclose(probs.sum(), 1.0):
                probs = probs / probs.sum()
            next_regime = rng.choice(transition_matrix.columns, p=probs)
        else:
            next_regime = rng.choice(regimes)
        
        path.append(next_regime)
        current_regime = next_regime
    
    return np.array(path)


def bootstrap_all_returns(combined_data, regime_path, seed=0):
    """
    Bootstrap ALL returns (PM + Cash + Benchmarks) conditional on regime
    
    ⭐ KEY DIFFERENCE: Samples entire ROWS from combined_data
    This preserves correlations between PM, cash, and benchmark returns!
    
    Parameters:
    -----------
    combined_data : DataFrame
        Combined PM, cash, benchmark returns with 'macro_regime' column
    regime_path : array
        Simulated regime sequence
    seed : int
        Random seed
    
    Returns:
    --------
    DataFrame : Simulated returns with all columns
    """
    rng = np.random.default_rng(seed)
    n_quarters = len(regime_path)
    
    # Identify columns to bootstrap
    exclude_cols = ['Date', 'macro_regime', 'GDP_YoY', 'CPI_YoY', 'FedFundsRate']
    return_cols = [c for c in combined_data.columns if c not in exclude_cols]
    
    simulated_returns = []
    
    for t, regime in enumerate(regime_path):
        # Get historical data for this regime
        regime_data = combined_data[combined_data['macro_regime'] == regime][return_cols]
        
        if len(regime_data) == 0:
            # Fallback: use overall mean
            print(f"Warning: No data for {regime}, using overall mean")
            sampled_returns = combined_data[return_cols].mean()
        else:
            # ⭐ Sample entire ROW to preserve correlations ⭐
            idx = rng.integers(0, len(regime_data))
            sampled_returns = regime_data.iloc[idx]
        
        simulated_returns.append(sampled_returns)
    
    # Create DataFrame
    sim_df = pd.DataFrame(simulated_returns).reset_index(drop=True)
    sim_df['macro_regime'] = regime_path
    sim_df.index.name = 't'
    
    return sim_df


def simulate_multipaths(
    combined_data,
    transition_matrix,
    n_paths=1000,
    n_quarters=80,
    start_regime='Downturn',
    seed=0,
    verbose=True
):
    """
    Generate multiple paths with PM + Cash + Benchmark returns
    
    ⭐ Returns simulated data ready for ELTIF simulation
    """
    if verbose:
        print("="*80)
        print(f"SIMULATING {n_paths} PATHS × {n_quarters} QUARTERS")
        print("="*80)
        print(f"Starting regime: {start_regime}")
    
    all_paths = []
    
    for path_id in range(n_paths):
        if verbose and (path_id + 1) % 200 == 0:
            print(f"  Path {path_id+1}/{n_paths} ({100*(path_id+1)/n_paths:.0f}%)")
        
        # Simulate regime path
        regime_path = simulate_regime_path(
            transition_matrix,
            n_quarters=n_quarters,
            start_regime=start_regime,
            seed=seed + path_id
        )
        
        # Bootstrap ALL returns for this regime path
        path_returns = bootstrap_all_returns(
            combined_data,
            regime_path,
            seed=seed + path_id + n_paths
        )
        
        # Add path identifier
        path_returns['path'] = path_id
        path_returns = path_returns.set_index(['path', path_returns.index])
        
        all_paths.append(path_returns)
    
    # Combine all paths
    boot_df = pd.concat(all_paths)
    boot_df.index = boot_df.index.set_names(['path', 't'])
    
    if verbose:
        print(f"\n✓ Simulation complete")
        print(f"  Shape: {boot_df.shape}")
        print(f"  Columns: {list(boot_df.columns)}")
        
        print("\n  Regime distribution in simulated data:")
        regime_dist = boot_df['macro_regime'].value_counts()
        for regime, count in regime_dist.items():
            pct = 100 * count / len(boot_df)
            print(f"    {regime:15s}: {count:6d} ({pct:5.1f}%)")
    
    return boot_df


# %%
# =============================================================================
# STEP 5: RUN SIMULATION
# =============================================================================

print("\n" + "="*80)
print("STEP 5: RUNNING SIMULATION")
print("="*80)

sim_df = simulate_multipaths(
    combined_data=combined_data,
    transition_matrix=transition_matrix,
    n_paths=10000,
    n_quarters=80,
    start_regime='Goldilocks',
    seed=42
)

# %%
# =============================================================================
# STEP 6: VALIDATION
# =============================================================================

print("\n" + "="*80)
print("STEP 6: VALIDATION - SIMULATED VS HISTORICAL")
print("="*80)

# Get PM columns
pm_cols = [c for c in combined_data.columns 
           if 'Hamilton' in c or 'Private' in c]

combined_data['avg_pm_return'] = combined_data[pm_cols].mean(axis=1)
sim_df['avg_pm_return'] = sim_df[pm_cols].mean(axis=1)

print("\nAverage PM Returns by Regime (Annualized):")
print("-"*80)

for regime in ['Goldilocks', 'Overheating', 'Downturn', 'Stagflation']:
    hist_data = combined_data[combined_data['macro_regime'] == regime]['avg_pm_return']
    sim_data = sim_df[sim_df['macro_regime'] == regime]['avg_pm_return']
    
    if len(hist_data) > 0 and len(sim_data) > 0:
        hist_mean = hist_data.mean() * 4
        sim_mean = sim_data.mean() * 4
        
        print(f"{regime:15s}:")
        print(f"  Historical: {hist_mean:6.2%}  (N={len(hist_data)})")
        print(f"  Simulated:  {sim_mean:6.2%}  (N={len(sim_data)})")

# %%
# =============================================================================
# STEP 7: SAVE FOR ELTIF SIMULATION
# =============================================================================

print("\n" + "="*80)
print("STEP 7: SAVING SIMULATED DATA")
print("="*80)

# Reset index to make it easier to work with
sim_df_export = sim_df.reset_index()

# Save
sim_df_export.to_csv('simulated_pm_cash_returns.csv', index=False)
print("\n✓ Saved to: simulated_pm_cash_returns.csv")

print(f"\nColumns in exported file:")
for col in sim_df_export.columns:
    print(f"  - {col}")

print("\n" + "="*80)
print("✅ SIMULATION COMPLETE - READY FOR ELTIF!")
print("="*80)
print("\nNext steps:")
print("1. Use 'simulated_pm_cash_returns.csv' as input to ELTIF simulation")
print("2. PM returns, cash returns, AND benchmarks are all included")
print("3. All returns are aligned by regime (correlations preserved)")
print("4. Use the 'path' column to separate different simulation paths")

# %%
# =============================================================================
# STEP 8: VISUALIZATION
# =============================================================================

def plot_validation(combined_data, sim_df):
    """Compare historical vs simulated by regime"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    pm_cols = [c for c in combined_data.columns if 'Hamilton' in c or 'Private' in c]
    combined_data['avg_return'] = combined_data[pm_cols].mean(axis=1)
    sim_df['avg_return'] = sim_df[pm_cols].mean(axis=1)
    
    regimes = ['Goldilocks', 'Overheating', 'Downturn', 'Stagflation']
    
    for ax, regime in zip(axes.flat, regimes):
        hist_data = combined_data[combined_data['macro_regime'] == regime]['avg_return'] * 4
        sim_data = sim_df[sim_df['macro_regime'] == regime]['avg_return'] * 4
        
        if len(hist_data) > 0 and len(sim_data) > 0:
            ax.hist(hist_data, bins=20, alpha=0.5, color='blue',
                   label=f'Historical (N={len(hist_data)})', density=True)
            ax.hist(sim_data, bins=50, alpha=0.5, color='red',
                   label=f'Simulated (N={len(sim_data)})', density=True)
            
            ax.axvline(hist_data.mean(), color='blue', linestyle='--', linewidth=2)
            ax.axvline(sim_data.mean(), color='red', linestyle='--', linewidth=2)
            
            ax.set_title(f'{regime}\n(μ_hist={hist_data.mean():.1%}, μ_sim={sim_data.mean():.1%})',
                        fontweight='bold')
            ax.set_xlabel('Annualized Return')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('historical_vs_simulated.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved validation plot to: historical_vs_simulated.png")
    plt.show()

plot_validation(combined_data, sim_df)