"""
Complete Pipeline for Goldstein-Style Fund Flow Regression
===========================================================

This script handles EVERYTHING from raw CSV files to final regression results:
1. Downloads benchmark data (S&P 500, VBMFX, 3-month T-bill)
2. Loads and processes your fund data (returns, TNA, TER)
3. Calculates fund alphas using two-factor model
4. Calculates fund flows
5. Runs Goldstein-style flow-performance regression

Based on: Goldstein, Jiang, and Ng (2017) JFE
"""

import pandas as pd
import numpy as np
import glob
import yfinance as yf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DOWNLOAD BENCHMARK DATA
# ============================================================================

def download_benchmarks(start_date='1991-01-01', end_date='2024-12-31'):
    """
    Download S&P 500, Vanguard Total Bond Market, and 3-month T-bill
    """
    print("="*80)
    print("STEP 1: DOWNLOADING BENCHMARK DATA")
    print("="*80)
    
    # Download S&P 500
    print("\n1. Downloading S&P 500...")
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    sp500_returns = sp500['Adj Close'].pct_change()
    
    # Download Vanguard Total Bond Market Index Fund
    print("2. Downloading Vanguard Total Bond Market (VBMFX)...")
    vbmfx = yf.download('VBMFX', start=start_date, end=end_date, progress=False)
    bond_returns = vbmfx['Adj Close'].pct_change()
    
    # Download 3-month T-bill rate
    print("3. Downloading 3-month T-bill rate...")
    try:
        tbill_data = yf.download('^IRX', start=start_date, end=end_date, progress=False)
        tbill_annual = tbill_data['Adj Close'] / 100  # Convert from percent
        tbill_monthly = tbill_annual / 12  # Convert annual to monthly
        print("   ✓ Downloaded from Yahoo Finance (^IRX)")
    except:
        print("   WARNING: Could not download T-bill. Using 0% risk-free rate.")
        tbill_monthly = pd.Series(0, index=sp500_returns.index)
    
    # Create DataFrame
    benchmarks = pd.DataFrame({
        'Date': sp500_returns.index,
        'SP500_Return': sp500_returns.values,
        'Bond_Market_Return': bond_returns.values,
        'RF': tbill_monthly.reindex(sp500_returns.index, method='ffill').fillna(0).values
    })
    
    # Remove missing values
    benchmarks = benchmarks.dropna(subset=['SP500_Return', 'Bond_Market_Return'])
    
    # Calculate excess returns
    benchmarks['Excess_SP500'] = benchmarks['SP500_Return'] - benchmarks['RF']
    benchmarks['Excess_Bond'] = benchmarks['Bond_Market_Return'] - benchmarks['RF']
    
    print(f"\n✓ Benchmark data downloaded:")
    print(f"  Date range: {benchmarks['Date'].min().date()} to {benchmarks['Date'].max().date()}")
    print(f"  Observations: {len(benchmarks)}")
    
    print("\n  Sample benchmark returns:")
    print(benchmarks.head())
    
    print("\n  Benchmark statistics (monthly):")
    print(benchmarks[['SP500_Return', 'Bond_Market_Return', 'RF']].describe())
    
    return benchmarks


# ============================================================================
# PART 2: LOAD YOUR CSV DATA
# ============================================================================

def load_return_data():
    """
    Load and combine all return_batch_*.csv files
    """
    print("\n" + "="*80)
    print("STEP 2: LOADING RETURN DATA")
    print("="*80)
    
    # Load all return batch files
    files = glob.glob('return_batch_*.csv')
    print(f"Found {len(files)} return batch files")
    
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    # Combine all batches
    returns = pd.concat(dfs, ignore_index=True)
    
    # Clean up
    returns['Date'] = pd.to_datetime(returns['Date'])
    returns = returns.rename(columns={'Rolling Performance': 'Return'})
    
    # CRITICAL: Convert from percent to decimal
    # Your data: 7.719205 → 0.07719205
    returns['Return'] = returns['Return'] / 100
    
    print(f"\n✓ Return data loaded:")
    print(f"  Total observations: {len(returns):,}")
    print(f"  Unique funds: {returns['Instrument'].nunique()}")
    print(f"  Date range: {returns['Date'].min().date()} to {returns['Date'].max().date()}")
    
    print("\n  Sample returns:")
    print(returns.head())
    
    print("\n  Return statistics:")
    print(returns['Return'].describe())
    
    return returns


def load_tna_data():
    """
    Load and combine all tna_checkpoint_*.csv files
    Convert from wide to long format
    """
    print("\n" + "="*80)
    print("STEP 3: LOADING TNA DATA")
    print("="*80)
    
    files = glob.glob('tna_checkpoint_*.csv')
    print(f"Found {len(files)} TNA checkpoint files")
    
    tna_list = []
    
    for file in files:
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Convert from wide to long format
        # Before: Date | LP40000007 | LP40000008 | ...
        # After:  Date | Instrument | TNA
        df_long = df.melt(
            id_vars=['Date'], 
            var_name='Instrument', 
            value_name='TNA'
        )
        
        tna_list.append(df_long)
    
    # Combine all batches
    tna = pd.concat(tna_list, ignore_index=True)
    
    # Remove missing TNA values
    tna = tna[tna['TNA'].notna()]
    
    # Remove zero or negative TNA (data errors)
    tna = tna[tna['TNA'] > 0]
    
    print(f"\n✓ TNA data loaded:")
    print(f"  Total observations: {len(tna):,}")
    print(f"  Unique funds: {tna['Instrument'].nunique()}")
    
    print("\n  Sample TNA:")
    print(tna.head())
    
    print("\n  TNA statistics:")
    print(tna['TNA'].describe())
    
    return tna


def load_fund_characteristics():
    """
    Load TER and inception date data
    """
    print("\n" + "="*80)
    print("STEP 4: LOADING FUND CHARACTERISTICS")
    print("="*80)
    
    # Load all TER batch files
    files = glob.glob('ter_batch_*.csv')
    print(f"Found {len(files)} TER batch files")
    
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    ter = pd.concat(dfs, ignore_index=True)
    
    # Clean dates
    ter['Date'] = pd.to_datetime(ter['Date'], errors='coerce')
    ter['Fund Inception Date'] = pd.to_datetime(ter['Fund Inception Date'], errors='coerce')
    
    # Rename columns
    ter = ter.rename(columns={
        'Total Expense Ratio': 'TER',
        'Fund Inception Date': 'Inception_Date'
    })
    
    # Get inception date (earliest date for each fund)
    inception = ter.groupby('Instrument')['Inception_Date'].min().reset_index()
    
    # Get most recent TER for each fund
    ter_latest = ter.sort_values(['Instrument', 'Date']).groupby('Instrument').last()[['TER']].reset_index()
    
    # Merge
    fund_chars = pd.merge(inception, ter_latest, on='Instrument', how='outer')
    
    print(f"\n✓ Fund characteristics loaded:")
    print(f"  Unique funds: {len(fund_chars):,}")
    
    print("\n  Sample characteristics:")
    print(fund_chars.head())
    
    print("\n  TER statistics:")
    print(fund_chars['TER'].describe())
    
    return fund_chars


# ============================================================================
# PART 3: CALCULATE ALPHAS
# ============================================================================

def calculate_alpha_rolling(fund_returns, benchmarks, window=12):
    """
    Calculate rolling 12-month alpha for each fund
    Alpha = intercept from: Excess_Fund_Return ~ Excess_Bond + Excess_Stock
    
    This follows Goldstein et al. (2017) exactly
    """
    print("\n" + "="*80)
    print("STEP 5: CALCULATING ROLLING ALPHAS")
    print("="*80)
    
    # Merge fund returns with benchmarks
    df = pd.merge(
        fund_returns,
        benchmarks[['Date', 'Excess_SP500', 'Excess_Bond', 'RF']],
        on='Date',
        how='left'
    )
    
    # Calculate excess fund return
    df['Excess_Fund_Return'] = df['Return'] - df['RF']
    
    # Sort by fund and date
    df = df.sort_values(['Instrument', 'Date']).reset_index(drop=True)
    
    # Initialize results list
    alphas = []
    
    # Group by fund
    funds = df['Instrument'].unique()
    print(f"Calculating alpha for {len(funds)} funds (this may take a while)...")
    
    for i, fund in enumerate(funds):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(funds)} funds ({100*(i+1)/len(funds):.1f}%)")
        
        fund_data = df[df['Instrument'] == fund].copy()
        fund_data = fund_data.reset_index(drop=True)
        
        # Calculate rolling alpha
        for idx in range(len(fund_data)):
            if idx < window - 1:
                # Not enough data for window
                alphas.append({
                    'Instrument': fund,
                    'Date': fund_data.loc[idx, 'Date'],
                    'Alpha': np.nan
                })
            else:
                # Get past 12 months of data
                window_data = fund_data.iloc[idx - window + 1 : idx + 1]
                
                # Check if we have enough non-missing data
                required_cols = ['Excess_Fund_Return', 'Excess_SP500', 'Excess_Bond']
                if window_data[required_cols].isnull().sum().sum() > 0:
                    alphas.append({
                        'Instrument': fund,
                        'Date': fund_data.loc[idx, 'Date'],
                        'Alpha': np.nan
                    })
                    continue
                
                # Run regression: Excess_Fund_Return ~ Excess_Bond + Excess_Stock
                X = window_data[['Excess_Bond', 'Excess_SP500']]
                X = sm.add_constant(X)
                y = window_data['Excess_Fund_Return']
                
                try:
                    model = sm.OLS(y, X).fit()
                    alpha = model.params['const']  # Intercept is alpha
                    
                    alphas.append({
                        'Instrument': fund,
                        'Date': fund_data.loc[idx, 'Date'],
                        'Alpha': alpha
                    })
                except:
                    alphas.append({
                        'Instrument': fund,
                        'Date': fund_data.loc[idx, 'Date'],
                        'Alpha': np.nan
                    })
    
    alpha_df = pd.DataFrame(alphas)
    
    print(f"\n✓ Alpha calculation complete:")
    print(f"  Total observations: {len(alpha_df):,}")
    print(f"  Non-missing alphas: {alpha_df['Alpha'].notna().sum():,}")
    
    print("\n  Alpha statistics (monthly):")
    print(alpha_df['Alpha'].describe())
    
    return alpha_df


# ============================================================================
# PART 4: CALCULATE FLOWS
# ============================================================================

def calculate_flows(returns, tna, fund_chars, alphas):
    """
    Merge all data and calculate flows
    Flow(t) = TNA(t) - TNA(t-1) * (1 + Return(t))
    """
    print("\n" + "="*80)
    print("STEP 6: CALCULATING FLOWS")
    print("="*80)
    
    # Merge returns and TNA
    df = pd.merge(
        tna,
        returns[['Instrument', 'Date', 'Return']],
        on=['Instrument', 'Date'],
        how='inner'
    )
    
    print(f"After merging TNA and returns: {len(df):,} observations")
    
    # Sort by fund and date
    df = df.sort_values(['Instrument', 'Date']).reset_index(drop=True)
    
    # Calculate lagged TNA
    df['TNA_lag'] = df.groupby('Instrument')['TNA'].shift(1)
    
    # Calculate flow
    # Flow(t) = TNA(t) - TNA(t-1) * (1 + Return(t))
    df['Flow'] = df['TNA'] - df['TNA_lag'] * (1 + df['Return'])
    
    # Calculate flow rate (percentage of lagged TNA)
    df['Flow_Rate'] = df['Flow'] / df['TNA_lag']
    
    # Drop first observation for each fund (no lagged TNA)
    df = df[df['Flow_Rate'].notna()]
    
    print(f"After calculating flows: {len(df):,} observations")
    
    # Add fund characteristics
    df = pd.merge(df, fund_chars, on='Instrument', how='left')
    
    # Calculate age in years
    df['Age'] = (df['Date'] - df['Inception_Date']).dt.days / 365.25
    
    # Log of lagged TNA (for size control)
    df['Log_TNA_lag'] = np.log(df['TNA_lag'])
    
    # Merge with alphas
    df = pd.merge(
        df,
        alphas[['Instrument', 'Date', 'Alpha']],
        on=['Instrument', 'Date'],
        how='left'
    )
    
    print(f"After merging alphas: {len(df):,} observations")
    
    print("\n  Flow statistics:")
    print(df['Flow_Rate'].describe())
    
    print("\n  Sample with flows:")
    print(df[['Instrument', 'Date', 'TNA', 'Return', 'Flow', 'Flow_Rate', 'Alpha']].head(10))
    
    return df


# ============================================================================
# PART 5: CLEAN DATA
# ============================================================================

def clean_data(df):
    """
    Remove outliers and missing data
    """
    print("\n" + "="*80)
    print("STEP 7: CLEANING DATA")
    print("="*80)
    
    print(f"Starting observations: {len(df):,}")
    
    # Remove extreme flow rates (likely corporate actions or errors)
    df = df[(df['Flow_Rate'] > -0.5) & (df['Flow_Rate'] < 2.0)]
    print(f"After flow rate filter [-50%, +200%]: {len(df):,}")
    
    # Remove extreme returns
    df = df[(df['Return'] > -0.9) & (df['Return'] < 3.0)]
    print(f"After return filter: {len(df):,}")
    
    # Remove missing alpha
    df = df[df['Alpha'].notna()]
    print(f"After alpha filter: {len(df):,}")
    
    # Remove missing TER
    df = df[df['TER'].notna()]
    print(f"After TER filter: {len(df):,}")
    
    # Remove negative age
    df = df[df['Age'] > 0]
    print(f"After age filter: {len(df):,}")
    
    # Add lagged flow
    df = df.sort_values(['Instrument', 'Date']).reset_index(drop=True)
    df['Lagged_Flow'] = df.groupby('Instrument')['Flow_Rate'].shift(1)
    
    # Remove first observation for each fund (no lagged flow)
    df = df[df['Lagged_Flow'].notna()]
    print(f"After lagged flow filter: {len(df):,}")
    
    print(f"\n✓ Final clean dataset: {len(df):,} observations")
    
    return df


# ============================================================================
# PART 6: RUN GOLDSTEIN REGRESSION
# ============================================================================

def goldstein_regression(df):
    """
    Run Goldstein-style flow-performance regression
    
    Flow(t) = α + β₁*Alpha(t-1) + β₂*Alpha(t-1)*I(Alpha<0) 
              + β₃*I(Alpha<0) + Controls + ε
    """
    print("\n" + "="*80)
    print("STEP 8: RUNNING GOLDSTEIN REGRESSION")
    print("="*80)
    
    # Create indicator for negative alpha
    df['Alpha_Negative'] = (df['Alpha'] < 0).astype(int)
    
    # Create interaction term
    df['Alpha_x_Negative'] = df['Alpha'] * df['Alpha_Negative']
    
    # Add year-month for time fixed effects
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    
    # Run regression
    print("\nRunning regression with controls and month fixed effects...")
    print("Dependent variable: Flow_Rate")
    print("Independent variables: Alpha, Alpha × I(Alpha<0), Controls, Month FE")
    
    model = smf.ols(
        '''Flow_Rate ~ Alpha + Alpha_x_Negative + Alpha_Negative 
           + Lagged_Flow + Log_TNA_lag + TER + C(YearMonth)''',
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['Instrument']})
    
    print("\n" + "="*80)
    print("REGRESSION RESULTS")
    print("="*80)
    print(model.summary())
    
    # Extract key coefficients
    beta1 = model.params['Alpha']
    beta2 = model.params['Alpha_x_Negative']
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"\nSensitivity of flows to POSITIVE alpha (β₁): {beta1:.4f}")
    print(f"  t-stat: {model.tvalues['Alpha']:.2f}")
    print(f"  Interpretation: 1% increase in alpha → {beta1:.4f}% increase in flows")
    
    print(f"\nAdditional sensitivity for NEGATIVE alpha (β₂): {beta2:.4f}")
    print(f"  t-stat: {model.tvalues['Alpha_x_Negative']:.2f}")
    
    total_negative = beta1 + beta2
    print(f"\nTotal sensitivity of flows to NEGATIVE alpha (β₁ + β₂): {total_negative:.4f}")
    print(f"  Interpretation: 1% decrease in alpha → {total_negative:.4f}% increase in outflows")
    
    ratio = total_negative / beta1 if beta1 != 0 else np.nan
    print(f"\nAsymmetry ratio: {ratio:.2f}")
    print(f"  Outflows are {ratio:.2f}x more sensitive than inflows")
    
    if beta2 > 0:
        print("\n✓ CONCAVE flow-performance relation (Goldstein finding)")
        print("  → Outflows MORE sensitive to bad performance than inflows to good performance")
    else:
        print("\n✗ CONVEX flow-performance relation (typical equity fund pattern)")
        print("  → Inflows MORE sensitive to good performance than outflows to bad performance")
    
    return model, df


# ============================================================================
# PART 7: SAVE RESULTS
# ============================================================================

def save_results(df, model):
    """
    Save cleaned data and regression results
    """
    print("\n" + "="*80)
    print("STEP 9: SAVING RESULTS")
    print("="*80)
    
    # Save cleaned data
    df.to_csv('fund_flows_analysis_data.csv', index=False)
    print("✓ Saved analysis data to: fund_flows_analysis_data.csv")
    
    # Save regression results
    with open('goldstein_regression_results.txt', 'w') as f:
        f.write("GOLDSTEIN-STYLE FUND FLOW REGRESSION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Number of observations: {len(df):,}\n")
        f.write(f"Number of unique funds: {df['Instrument'].nunique():,}\n")
        f.write(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n\n")
        f.write("="*80 + "\n\n")
        f.write(str(model.summary()))
        f.write("\n\n" + "="*80 + "\n")
        
        beta1 = model.params['Alpha']
        beta2 = model.params['Alpha_x_Negative']
        total_negative = beta1 + beta2
        
        f.write("\nKEY COEFFICIENTS:\n")
        f.write(f"  β₁ (Alpha): {beta1:.6f}\n")
        f.write(f"  β₂ (Alpha × Negative): {beta2:.6f}\n")
        f.write(f"  Total negative sensitivity: {total_negative:.6f}\n")
        f.write(f"  Asymmetry ratio: {total_negative/beta1 if beta1 != 0 else np.nan:.2f}\n")
    
    print("✓ Saved regression results to: goldstein_regression_results.txt")
    
    # Save summary statistics
    summary_stats = df[['Flow_Rate', 'Alpha', 'Return', 'TNA', 'TER', 'Age']].describe()
    summary_stats.to_csv('summary_statistics.csv')
    print("✓ Saved summary statistics to: summary_statistics.csv")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run the complete pipeline
    """
    print("\n" + "="*80)
    print("GOLDSTEIN FUND FLOW ANALYSIS - COMPLETE PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Step 1: Download benchmarks
        benchmarks = download_benchmarks()
        benchmarks.to_csv('benchmarks.csv', index=False)
        print("\n✓ Benchmarks saved to: benchmarks.csv")
        
        # Step 2-4: Load your data
        returns = load_return_data()
        tna = load_tna_data()
        fund_chars = load_fund_characteristics()
        
        # Step 5: Calculate alphas
        alphas = calculate_alpha_rolling(returns, benchmarks, window=12)
        alphas.to_csv('fund_alphas.csv', index=False)
        print("\n✓ Alphas saved to: fund_alphas.csv")
        
        # Step 6: Calculate flows
        df = calculate_flows(returns, tna, fund_chars, alphas)
        
        # Step 7: Clean data
        df = clean_data(df)
        
        # Step 8: Run regression
        model, df = goldstein_regression(df)
        
        # Step 9: Save results
        save_results(df, model)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nOutput files created:")
        print("  1. benchmarks.csv - Benchmark returns")
        print("  2. fund_alphas.csv - Fund alphas")
        print("  3. fund_flows_analysis_data.csv - Complete analysis dataset")
        print("  4. goldstein_regression_results.txt - Regression output")
        print("  5. summary_statistics.csv - Descriptive statistics")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()