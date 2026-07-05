# %%
import pandas as pd
import numpy as np
import glob
import yfinance as yf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats.mstats import winsorize
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
# from sklearn.linear_model import RidgeCV
# from sklearn.preprocessing import StandardScaler
import pickle
import argparse
import sys

# ── Setting Arguments ───────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--run_tag', type=str, default='',
                     help='Suffix appended to output filenames, e.g. _bond_hy')
_parser.add_argument('--keyword', type=str, default='',
                     help='Case-insensitive substring filter for IssueLipperGlobalSchemeName, '
                          'e.g. "bond", "equity us". Comma-separate for multiple. '
                          'Blank = all funds.')
_parser.add_argument('--tna_min', type=float, default=10_000,
                     help='Minimum TNA_lag in native currency units (default 10000).')
_parser.add_argument('--alpha_lag', type=int, default=0,
                     help='Quarters to lag alpha before regression (default 0 = contemporaneous).')
_parser.add_argument('--skip_model5', action='store_true',
                     help='Skip Model 5 (fund FEs) — fast mode for large asset classes '
                          'where only Models 1-4 are needed for the comparison table.')
_args, _ = _parser.parse_known_args()
RUN_TAG      = _args.run_tag
TNA_MIN      = _args.tna_min
ALPHA_LAG    = _args.alpha_lag
SKIP_MODEL5  = _args.skip_model5

import os as _os
if RUN_TAG:
    _OUT_DIR = 'runs'
    _os.makedirs(_OUT_DIR, exist_ok=True)
    print(f"[run_tag={RUN_TAG!r}] Tagged outputs will be written to: LSEG_merged/runs/")
else:
    _OUT_DIR = '.'


# %%
def download_benchmarks(start_date='1991-01-01', end_date='2024-12-31'):
    """
    Download S&P 500, Vanguard Total Bond Market, and 3-month T-bill
    RETURNS QUARTERLY DATA (end of quarter)
    """
    print("="*80)
    print("STEP 1: DOWNLOADING BENCHMARK DATA")
    print("="*80)
    
    # Download S&P 500
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500_prices = sp500['Close']['^GSPC']
    else:
        sp500_prices = sp500['Close']
    
    # Download Vanguard Total Bond Market Index Fund
    vbmfx = yf.download('VBMFX', start=start_date, end=end_date, progress=False)
    if isinstance(vbmfx.columns, pd.MultiIndex):
        bond_prices = vbmfx['Close']['VBMFX']
    else:
        bond_prices = vbmfx['Close']
    
    # Download 3-month T-bill rate
    try:
        tbill_data = yf.download('^IRX', start=start_date, end=end_date, progress=False)
        if isinstance(tbill_data.columns, pd.MultiIndex):
            tbill_annual = tbill_data['Close']['^IRX'] / 100
        else:
            tbill_annual = tbill_data['Close'] / 100
        tbill_daily = tbill_annual / 252  # Convert annual to daily
    except Exception as e:
        print(f"   WARNING: Could not download T-bill ({str(e)}). Using 0% risk-free rate.")
        tbill_daily = pd.Series(0, index=sp500_prices.index)
    
    # Create daily DataFrame
    daily_data = pd.DataFrame({
        'Date': sp500_prices.index,
        'SP500_Price': sp500_prices.values,
        'Bond_Price': bond_prices.values,
        'RF_Daily': tbill_daily.reindex(sp500_prices.index, method='ffill').fillna(0).values
    })
    
    # CRITICAL: Resample to QUARTERLY (end of quarter)
    daily_data = daily_data.set_index('Date')
    
    # Resample to quarter-end
    quarterly = daily_data.resample('QE').last()
    
    # Calculate quarterly returns
    quarterly['SP500_Return'] = quarterly['SP500_Price'].pct_change()
    quarterly['Bond_Market_Return'] = quarterly['Bond_Price'].pct_change()
    
    # Calculate quarterly risk-free rate (compound daily rates)
    quarterly['RF'] = daily_data['RF_Daily'].resample('QE').apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Calculate excess returns
    quarterly['Excess_SP500'] = quarterly['SP500_Return'] - quarterly['RF']
    quarterly['Excess_Bond'] = quarterly['Bond_Market_Return'] - quarterly['RF']
    
    # Clean up
    benchmarks = quarterly[['SP500_Return', 'Bond_Market_Return', 'RF', 'Excess_SP500', 'Excess_Bond']].copy()
    benchmarks = benchmarks.reset_index()
    benchmarks = benchmarks.dropna(subset=['SP500_Return', 'Bond_Market_Return'])
    
    return benchmarks



def load_return_data():
    """
    Load and combine all return_batch_*.csv files
    """
    print("\n" + "="*80)
    print("STEP 2: LOADING RETURN DATA")
    print("="*80)
    
    # Load merged returns file (union of LSEG_download + LSEG_download_new)
    files = glob.glob('returns_merged.csv')
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
    
    # Check if returns are already in decimal or percent
    median_return = returns['Return'].median()
    if abs(median_return) > 1:
        print(f"  Returns appear to be in percent (median: {median_return:.2f}), converting to decimal")
        returns['Return'] = returns['Return'] / 100
    else:
        print(f"  Returns appear to be in decimal (median: {median_return:.4f})")
    
    # Validate returns are in reasonable range (-100% to +100% per quarter)
    before = len(returns)
    returns = returns[(returns['Return'] >= -1) & (returns['Return'] <= 1)]
    after = len(returns)
    if before > after:
        print(f"  Removed {before - after:,} rows with extreme returns")
    
    print(f"  Return statistics: min={returns['Return'].min():.2%}, max={returns['Return'].max():.2%}, median={returns['Return'].median():.4f}")
    
    return returns


def load_tna_data():
    """
    Load and combine all tna_checkpoint_*.csv files
    Convert from wide to long format
    """
    print("\n" + "="*80)
    print("STEP 3: LOADING TNA DATA")
    print("="*80)
    
    # Load merged TNA file (wide format, already combined)
    files = glob.glob('tna_merged.csv')
    print(f"Found {len(files)} TNA merged file(s)")

    tna_list = []
    for file in files:
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df_long = df.melt(
            id_vars=['Date'],
            var_name='Instrument',
            value_name='TNA'
        )
        tna_list.append(df_long)

    # Combine
    tna = pd.concat(tna_list, ignore_index=True)
    
    # Remove missing TNA values
    tna = tna[tna['TNA'].notna()]
    
    # Remove zero or negative TNA (data errors)
    tna = tna[tna['TNA'] > 0]
    
    return tna



def load_fund_characteristics(returns):
    """
    Load TER and inception dates
    """
    print("\n" + "="*80)
    print("STEP 4: LOADING FUND CHARACTERISTICS")
    print("="*80)
    
    # Load merged TER file (union of both downloads, NEW preferred on overlap)
    files = glob.glob('ter_merged.csv')
    print(f"Found {len(files)} TER merged file(s)")

    dfs = []
    for file in files:
        dfs.append(pd.read_csv(file))

    ter = pd.concat(dfs, ignore_index=True)
    
    # Clean dates
    ter['Date'] = pd.to_datetime(ter['Date'], errors='coerce')
    ter['Fund Inception Date'] = pd.to_datetime(ter['Fund Inception Date'], errors='coerce')
    
    # Rename columns
    ter = ter.rename(columns={
        'Total Expense Ratio': 'TER',
        'Fund Inception Date': 'Inception_Date'
    })
    
    # Get inception date from TER files (earliest date for each fund)
    inception_from_ter = ter.groupby('Instrument')['Inception_Date'].min().reset_index()
    
    # Get most recent TER for each fund
    ter_latest = ter.sort_values(['Instrument', 'Date']).groupby('Instrument').last()[['TER']].reset_index()
    
    # Merge TER with inception dates from TER files
    fund_chars = pd.merge(ter_latest, inception_from_ter, on='Instrument', how='outer')

    fund_chars['TER'] = pd.to_numeric(fund_chars['TER'], errors='coerce')

    # Convert percent to decimal 
    median_ter = fund_chars['TER'].median()
    if median_ter > 0.1:
        print(f"\n  Converting TER from percent to decimal (median: {median_ter:.2f}%)")
        fund_chars['TER'] = fund_chars['TER'] / 100
        print(f"  After conversion: {fund_chars['TER'].median():.6f}")
    
    # Calculate first return date for each fund (as fallback)
    print("\nCalculating first return dates as fallback for missing inception dates...")
    first_return_dates = returns.groupby('Instrument')['Date'].min().reset_index()
    first_return_dates.columns = ['Instrument', 'First_Return_Date']
    
    # Merge with first return dates
    fund_chars = pd.merge(fund_chars, first_return_dates, on='Instrument', how='outer')
    
    # Fill missing inception dates with first return date
    missing_before = fund_chars['Inception_Date'].isna().sum()
    fund_chars['Inception_Date'] = fund_chars['Inception_Date'].fillna(fund_chars['First_Return_Date'])
    missing_after = fund_chars['Inception_Date'].isna().sum()
    filled = missing_before - missing_after
    
    print(f"  Inception dates from TER files: {len(fund_chars) - missing_before:,}")
    print(f"  Inception dates filled from first return date: {filled:,}")
    print(f"  Still missing inception dates: {missing_after:,}")
    
    # Drop the temporary First_Return_Date column
    fund_chars = fund_chars.drop(columns=['First_Return_Date'])
    
    print(f"\n✓ Fund characteristics loaded:")
    print(f"  Unique funds: {len(fund_chars):,}")
    print(f"  Funds with TER: {fund_chars['TER'].notna().sum():,}")
    print(f"  Funds with Inception Date: {fund_chars['Inception_Date'].notna().sum():,}")
    
    print("\n  Sample characteristics:")
    print(fund_chars.head(10))
    
    print("\n  TER statistics:")
    print(fund_chars['TER'].describe())
    
    print("\n  Inception date range:")
    print(f"    Earliest: {fund_chars['Inception_Date'].min()}")
    print(f"    Latest: {fund_chars['Inception_Date'].max()}")
    
    return fund_chars



def load_fund_types(filepath='Clean_funds_with_asset_class.csv'):
    """
    Load fund type / asset class information from Clean_funds_with_asset_class.csv.)
    """
    print("\n" + "="*80)
    print("LOADING FUND TYPE DATA")
    print("="*80)

    fund_types = pd.read_csv(filepath)
    fund_types = fund_types.dropna(subset=['RIC_clean'])
    fund_types = fund_types.rename(columns={'RIC_clean': 'Instrument'})
    fund_types = fund_types[['Instrument', 'IssueLipperGlobalSchemeName']]

    return fund_types


def filter_by_asset_class(df, fund_types, asset_classes=None, keywords=None):
    """
    Filter fund data to selected asset classes from IssueLipperGlobalSchemeName.
    """
    print("\n" + "="*80)
    print("ASSET CLASS FILTER")
    print("="*80)

    # Coerce Instrument to str on both sides to prevent int/str type mismatch
    df = df.copy()
    df['Instrument'] = df['Instrument'].astype(str)
    fund_types = fund_types.copy()
    fund_types['Instrument'] = fund_types['Instrument'].astype(str)

    # Merge asset class labels onto main dataset
    df = df.merge(fund_types, on='Instrument', how='left')

    n_before = len(df)
    funds_before = df['Instrument'].nunique()
    matched = df['IssueLipperGlobalSchemeName'].notna().sum()
    print(f"  Rows in fund data:      {n_before:,}")
    print(f"  Unique funds:           {funds_before:,}")
    print(f"  Rows with asset class:  {matched:,} ({100*matched/n_before:.1f}%)")

    if asset_classes is not None:
        mask = df['IssueLipperGlobalSchemeName'].isin(asset_classes)
    elif keywords is not None:
        pattern = '|'.join(keywords)
        mask = df['IssueLipperGlobalSchemeName'].str.contains(
            pattern, case=False, na=False, regex=True
        )
    else:
        print("No asset class filter applied — using all funds.")
        return df

    df_filtered = df[mask].copy()
    print(f"  Rows after filter:      {len(df_filtered):,} ({100*len(df_filtered)/n_before:.1f}%)")
    print(f"  Unique funds after:     {df_filtered['Instrument'].nunique():,}")

    return df_filtered


def calculate_alpha_rolling(fund_returns, benchmarks, fund_chars, window=8):
    """
    Calculate rolling two year alpha for each fund
    Alpha = intercept from: Excess_Fund_Return ~ Excess_Bond + Excess_Stock
    
    This follows Goldstein et al. (2017)
    
    NOTE: Returns are adjusted for TER (net of fees) to match Goldstein methodology
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
    
    # Merge TER from fund characteristics
    df = pd.merge(df, fund_chars[['Instrument', 'TER']], on='Instrument', how='left')
    
    # Fill missing TER with median TER (better than 0 which would overstate returns)
    ter_median = df['TER'].median()
    df['TER'] = df['TER'].fillna(ter_median)
    
    print(f"  TER statistics after conversion:")
    print(f"    Min: {df['TER'].min():.4f}, Max: {df['TER'].max():.4f}, Median: {df['TER'].median():.4f}")
    
    # Calculate NET return (after TER) - TER is annual, convert to quarterly
    df['TER_Quarterly'] = df['TER'] / 4
    df['Net_Return'] = df['Return'] - df['TER_Quarterly']
    
    # Calculate excess fund return using NET return (after fees)
    df['Excess_Fund_Return'] = df['Net_Return'] - df['RF']
    
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
                
                # Require at least 6 out of 8 quarters to be non-missing.
                # Requiring all 8 excludes too many short-lived and dead funds
                # (disproportionately the failed ones), introducing survivorship bias.
                required_cols = ['Excess_Fund_Return', 'Excess_SP500', 'Excess_Bond']
                valid_obs = window_data[required_cols].dropna()
                if len(valid_obs) < 6:
                    alphas.append({
                        'Instrument': fund,
                        'Date': fund_data.loc[idx, 'Date'],
                        'Alpha': np.nan
                    })
                    continue
                # Use only the valid rows for the regression
                window_data = valid_obs
                
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
                except Exception:
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
    
    # Calculate lagged TNA and lagged date
    df['TNA_lag']  = df.groupby('Instrument')['TNA'].shift(1)
    df['Date_lag'] = df.groupby('Instrument')['Date'].shift(1)

    # Drop observations where the gap to the previous record is not ~1 quarter.
    # Non-consecutive TNA records (e.g. missing quarter) would distort the flow
    # formula because TNA(t-1) would reflect a different horizon than 1 quarter.
    df['months_gap'] = (df['Date'] - df['Date_lag']).dt.days / 30.44
    before_gap = len(df)
    df = df[df['months_gap'].between(2, 4)]   # 2–4 months tolerance around 1 quarter
    after_gap = len(df)
    print(f"  Removed {before_gap - after_gap:,} rows with non-consecutive TNA dates")
    df = df.drop(columns=['Date_lag', 'months_gap'])

    # Calculate flow
    # Flow(t) = TNA(t) - TNA(t-1) * (1 + Return(t))
    df['Flow'] = df['TNA'] - df['TNA_lag'] * (1 + df['Return'])

    # Calculate flow rate (percentage of lagged TNA)
    df['Flow_Rate'] = df['Flow'] / df['TNA_lag']

    # Drop first observation for each fund (no lagged TNA)
    df = df[df['Flow_Rate'].notna()]
    
    # Data validation - check for extreme values
    print(f"\nData validation before cleaning:")
    print(f"  Flow_Rate: min={df['Flow_Rate'].min():.2f}, max={df['Flow_Rate'].max():.2f}, median={df['Flow_Rate'].median():.4f}")
    print(f"  TNA_lag: min={df['TNA_lag'].min():.2f}, max={df['TNA_lag'].max():.2e}")
    
    # Remove extreme Flow_Rate values (beyond +/- 200% - likely data errors)
    # Also remove rows where TNA_lag is below the minimum threshold
    before     = len(df)
    df         = df[(df['Flow_Rate'] >= -2) & (df['Flow_Rate'] <= 2)]
    after_flow = len(df)
    df         = df[df['TNA_lag'] >= TNA_MIN]
    after_tna  = len(df)
    print(f"\n  Removed {before - after_flow:,} rows: extreme Flow_Rate (|rate| > 2)")
    print(f"  Removed {after_flow - after_tna:,} rows: TNA_lag < {TNA_MIN:,.0f}")
    print(f"  Flow_Rate after cleaning: min={df['Flow_Rate'].min():.2f}, max={df['Flow_Rate'].max():.2f}, median={df['Flow_Rate'].median():.4f}")
    
    # Add TER from fund characteristics
    df = pd.merge(df, fund_chars[['Instrument', 'TER']], on='Instrument', how='left')
    
    # Calculate inception date as first observed return date for each fund
    # Since returns are "since inception", the first date is the inception date
    print("\nCalculating fund age using first observed return date as inception...")
    first_dates = df.groupby('Instrument')['Date'].min().reset_index()
    first_dates.columns = ['Instrument', 'Inception_Date']
    
    df = pd.merge(df, first_dates, on='Instrument', how='left')
    
    print(f"  Funds with inception date: {df['Inception_Date'].notna().sum():,} ({100*df['Inception_Date'].notna().sum()/len(df):.1f}%)")
    
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
    
    print("\n  Age statistics (years since first return):")
    print(df['Age'].describe())
    
    print("\n  Sample with flows:")
    print(df[['Instrument', 'Date', 'TNA', 'Return', 'Flow', 'Flow_Rate', 'Alpha', 'Age']].head(10))
    
    return df


def clean_data(df):
    """
    Refined cleaning for Simulation:
    1. Removes missing values
    2. Winsorizes Flow_Rate and Alpha to handle extreme skew (136.0)
    3. Creates necessary interaction terms
    """
    print("\n" + "="*80)
    print("STEP 7: CLEANING & WINSORIZING DATA FOR SIMULATION")
    print("="*80)
    
    # Fill missing TER with cross-sectional median before dropping anything.
    # Dead/small funds disproportionately lack TER data; dropping them would
    # reintroduce survivorship bias even though the raw data includes them.
    ter_median = df['TER'].median()
    n_ter_missing = df['TER'].isna().sum()
    df['TER'] = df['TER'].fillna(ter_median)
    if n_ter_missing > 0:
        print(f"  Filled {n_ter_missing:,} missing TER values with median ({ter_median:.4f})")

    # Drop rows missing essential simulation variables (TER no longer causes drops)
    cols_to_check = ['Alpha', 'Log_TNA_lag', 'Flow_Rate', 'macro_regime']
    df = df.dropna(subset=cols_to_check)
    
    # More aggressive cleaning before winsorization
    print(f"\nBefore aggressive cleaning: {len(df):,} rows")
    print(f"  Flow_Rate: min={df['Flow_Rate'].min():.2f}, max={df['Flow_Rate'].max():.2f}")
    print(f"  Alpha: min={df['Alpha'].min():.2f}, max={df['Alpha'].max():.2f}")
    print(f"  Log_TNA_lag: min={df['Log_TNA_lag'].min():.2f}, max={df['Log_TNA_lag'].max():.2f}")
    
    # Sanity bounds on Log_TNA_lag (Flow_Rate already bounded in calculate_flows)
    df = df[df['Log_TNA_lag'] > 0]  # Log(TNA) must be positive
    df = df[df['Log_TNA_lag'] < 30] # Reasonable max for log(TNA)
    
    print(f"After aggressive cleaning: {len(df):,} rows")

    if len(df) == 0:
        raise ValueError(
            "Dataset is empty after cleaning. Likely causes:\n"
            "  1. Asset class filter matched 0 funds — check 'Rows with asset class' "
            "in the ASSET CLASS FILTER output above. If 0%, Instrument types don't match.\n"
            "  2. Macro regime merge matched 0 quarters — check 'After macro join' above."
        )

    # WINSORIZATION: Cap the top and bottom 1% to remove outliers while
    # preserving the tail behaviour that drives the concavity result.
    # Standard in the fund flow literature (Goldstein et al. 2017 use 1%).
    df['Flow_Rate'] = winsorize(df['Flow_Rate'], limits=[0.01, 0.01])
    df['Alpha'] = winsorize(df['Alpha'], limits=[0.01, 0.01])

    print(f"After winsorization (1% tails):")
    print(f"  Flow_Rate: min={df['Flow_Rate'].min():.4f}, max={df['Flow_Rate'].max():.4f}")
    print(f"  Alpha: min={df['Alpha'].min():.4f}, max={df['Alpha'].max():.4f}")
    
    # Create the interaction terms needed for the simulation
    df['Alpha_Negative'] = (df['Alpha'] < 0).astype(int)
    df['Alpha_x_Negative'] = df['Alpha'] * df['Alpha_Negative']
    
    for regime in ['Goldilocks', 'Overheating', 'Downturn', 'Stagflation']:
        mask = (df['macro_regime'] == regime).astype(int)
        df[f'Alpha_x_{regime}'] = df['Alpha'] * mask
        df[f'AlphaNeg_x_{regime}'] = df['Alpha_x_Negative'] * mask
    
    # Lagged flow for momentum in simulation
    # Log(Age) — standard control in Goldstein et al. (2017).
    # Older funds tend to have more stable investor bases and lower flow volatility.
    # Clip Age to at least 0.25 years (1 quarter) to avoid log(0) or negative values.
    df['Age'] = df['Age'].clip(lower=0.25)
    df['Log_Age'] = np.log(df['Age'])

    df = df.sort_values(['Instrument', 'Date'])
    df['Lagged_Flow'] = df.groupby('Instrument')['Flow_Rate'].shift(1)
    df = df.dropna(subset=['Lagged_Flow'])

    print(f"✓ Cleaned dataset: {len(df):,} observations")
    print(f"  New Skewness (Flow_Rate): {df['Flow_Rate'].skew():.2f}")
    return df

# %%
# Data loading cell — run once. Filter + regression are in the execution cell below.

try:
    # Step 1: Download benchmarks
    benchmarks = download_benchmarks()
    benchmarks.to_csv('benchmarks.csv', index=False)
    print("\n✓ Benchmarks saved to: benchmarks.csv")

    # Step 2-4: Load your data
    returns = load_return_data()
    tna = load_tna_data()
    fund_chars = load_fund_characteristics(returns)
    alphas = calculate_alpha_rolling(returns, benchmarks, fund_chars)

    # Step 6: Calculate flows (unfiltered — filter applied in execution cell below)
    df_raw = calculate_flows(returns, tna, fund_chars, alphas)

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()


# %%

def download_macro_data_direct(start_date='1988-01-01', end_date='2024-12-31'):
    """
    Download macroeconomic data directly from FRED as CSV
    """
    
    print("\n" + "="*80)
    print("STEP 9: DOWNLOADING MACRO DATA FROM FRED")
    print("="*80)
    
    # FRED provides direct CSV download via these URLs
    fred_urls = {
        'GDP': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP',
        'CPI': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL',
        'FEDFUNDS': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS',
        'USREC': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC'
    }
    
    macro_series = {}
    
    try:
        for name, url in fred_urls.items():
            print(f"\n{len(macro_series)+1}. Downloading {name}...")
            
            # Read CSV without parse_dates (FRED format changed)
            df = pd.read_csv(url)
            
            # The first column is the date (name varies)
            date_col = df.columns[0]  # Usually 'DATE' or 'observation_date'
            value_col = df.columns[1]
            
            # Rename columns
            df = df.rename(columns={date_col: 'Date', value_col: name})
            df = df[['Date', name]]
            
            # Convert date to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Filter by date range
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            
            macro_series[name] = df
            print(f"   ✓ Downloaded {len(df)} observations")
        
        # Merge all series
        print("\n5. Merging all series...")
        macro_data = macro_series['GDP']
        for name in ['CPI', 'FEDFUNDS', 'USREC']:
            macro_data = pd.merge(macro_data, macro_series[name], on='Date', how='outer')
        
        macro_data = macro_data.rename(columns={
            'FEDFUNDS': 'FedFundsRate',
            'USREC': 'NBER',
            'CPI': 'CPI'
        })
        
        macro_data = macro_data.sort_values('Date').reset_index(drop=True)
        
        print(f"\n✓ Macro data downloaded:")
        print(f"  Date range: {macro_data['Date'].min().date()} to {macro_data['Date'].max().date()}")
        print(f"  Observations: {len(macro_data)}")
        print("\n  Sample macro data:")
        print(macro_data.head())
        
        return macro_data
        
    except Exception as e:
        print(f"\n❌ ERROR downloading from FRED: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. FRED website may be temporarily down")
        raise


def load_ilmanen_macro_regimes(csv_path=None):
    """
    Load balanced macro regimes from the CSV produced by new_macro_indicators.py.
    Maps the 4-cell growth×inflation labels to economic names so all downstream
    code (regression, simulation, dashboard) remains unchanged.
    """
    print("\n" + "="*80)
    print("STEP 10: LOADING ILMANEN MACRO REGIMES FROM CSV")
    print("="*80)

    if csv_path is None:
        csv_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                 'macro_indicators.csv')

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index.name = 'Date'
    df = df.reset_index()

    # Drop early rows where z-scores hadn't yet accumulated enough history
    # (MIN_OBS=30 quarters, set in new_macro_indicators.py): NaN growth/inflation
    # means the composite is invalid, but pandas would have silently assigned
    # growth_up=0 → fake GrowthDown regime.
    n_before = len(df)
    df = df.dropna(subset=['growth', 'inflation'])
    print(f"  Dropped {n_before - len(df)} rows with NaN z-score composites "
          f"(expanding window warm-up); {len(df)} quarters remain.")

    regime_map = {
        'GrowthUp/InflationDown':   'Goldilocks',
        'GrowthUp/InflationUp':     'Overheating',
        'GrowthDown/InflationDown': 'Downturn',
        'GrowthDown/InflationUp':   'Stagflation',
    }
    df['macro_regime'] = df['regime'].map(regime_map)
    df = df.dropna(subset=['macro_regime'])

    print("\n  Macro Regime Distribution (Ilmanen balanced):")
    print(df['macro_regime'].value_counts())

    return df[['Date', 'macro_regime', 'growth', 'inflation']]


def create_macro_regimes(macro_data):
    """
    Create macro regime variables (Goldilocks, Overheating, Downturn, Stagflation)
    Returns QUARTERLY data to match fund data frequency
    """
    print("\n" + "="*80)
    print("STEP 10: CREATING MACRO REGIMES")
    print("="*80)
    
    df = macro_data.copy()
    
    # Apply information lags
    df['GDP'] = df['GDP'].shift(2)
    df['CPI'] = df['CPI'].shift(1)
    
    # Forward-fill
    df[['GDP', 'CPI', 'FedFundsRate', 'NBER']] = df[['GDP', 'CPI', 'FedFundsRate', 'NBER']].ffill()
    
    # YoY growth
    df['GDP_YoY'] = df['GDP'].pct_change(periods=12)
    df['CPI_YoY'] = df['CPI'].pct_change(periods=12)
    
    # Growth regime
    df['growth_regime'] = df['NBER'].apply(lambda x: 'Low' if x == 1 else 'High')
    
    # Inflation regime
    df['CPI_YoY_lag'] = df['CPI_YoY'].shift(12)
    df['inflation_surprise'] = df['CPI_YoY'] - df['CPI_YoY_lag']
    df['inflation_regime'] = 'Low'
    df.loc[(df['CPI_YoY'] > 0.02) & (df['inflation_surprise'] > 0), 'inflation_regime'] = 'High'
    
    # Macro regimes
    conditions = [
        (df['growth_regime'] == 'High') & (df['inflation_regime'] == 'Low'),
        (df['growth_regime'] == 'High') & (df['inflation_regime'] == 'High'),
        (df['growth_regime'] == 'Low') & (df['inflation_regime'] == 'Low'),
        (df['growth_regime'] == 'Low') & (df['inflation_regime'] == 'High'),
    ]
    choices = ['Goldilocks', 'Overheating', 'Downturn', 'Stagflation']
    df['macro_regime'] = np.select(conditions, choices, default='Unknown')
    
    df = df.dropna()
    
    # Resample to QUARTERLY to match fund data!
    df = df.set_index('Date')
    
    # Resample to quarter-end, taking the last value of each quarter
    df_quarterly = df.resample('QE').last()
    
    # For categorical variables, use the mode (most common value) in the quarter
    # For numeric variables, use the last value
    df_quarterly['macro_regime'] = df.resample('QE')['macro_regime'].apply(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[-1]
    )
    df_quarterly = df_quarterly.reset_index()


    print("\n  Macro Regime Distribution:")
    print(df_quarterly['macro_regime'].value_counts())

    return df_quarterly


def merge_macro_with_funds(fund_data, macro_data):
    """
    Merge macro regimes with fund data
    """
    print("\n" + "="*80)
    print("STEP 11: MERGING MACRO WITH FUND DATA")
    print("="*80)
    
    fund_data = fund_data.copy()
    macro_data = macro_data.copy()
    
    fund_data['Date'] = pd.to_datetime(fund_data['Date'])
    macro_data['Date'] = pd.to_datetime(macro_data['Date'])

    # Match on year-quarter — robust regardless of whether fund dates are
    # quarter-start, quarter-end, or mid-quarter.
    fund_data = fund_data.copy()
    macro_data = macro_data.copy()
    fund_data['_yq'] = fund_data['Date'].dt.to_period('Q').astype(str)
    macro_data['_yq'] = macro_data['Date'].dt.to_period('Q').astype(str)

    macro_cols = ['_yq', 'macro_regime', 'growth', 'inflation']
    df_merged = fund_data.merge(macro_data[macro_cols], on='_yq', how='left')
    df_merged = df_merged.drop(columns=['_yq'])

    n_before = len(df_merged)
    df_merged = df_merged.dropna(subset=['macro_regime'])
    print(f"  Fund rows:        {n_before:,}")
    print(f"  After macro join: {len(df_merged):,} ({100*len(df_merged)/max(n_before,1):.1f}% matched)")

    return df_merged


def goldstein_macro_regressions(df):
    """
    Run 4 regression models with macro interactions
    """
    print("\n" + "="*80)
    print("STEP 12: GOLDSTEIN REGRESSIONS WITH MACRO INTERACTIONS")
    print("="*80)
    
    df = df.copy()
    df['Alpha_Negative'] = (df['Alpha'] < 0).astype(int)
    df['Alpha_x_Negative'] = df['Alpha'] * df['Alpha_Negative']
    df['YearQuarter'] = df['Date'].dt.to_period('Q').astype(str)   # quarterly FEs (data is quarterly)
    
    results = {}
    
    # ========================================================================
    # MODEL 1: Baseline (replicate earlier regression)
    # ========================================================================
    print("\n" + "-"*80)
    print("MODEL 1: BASELINE (No Macro)")
    print("-"*80)
    
    try:
        model1 = smf.ols(
            '''Flow_Rate ~ Alpha + Alpha_x_Negative + Alpha_Negative
               + Lagged_Flow + Log_TNA_lag + Log_Age + TER + C(YearQuarter)''',
            data=df
        ).fit(cov_type='cluster', cov_kwds={'groups': df['Instrument']})
        
        results['baseline'] = model1
        
        beta1 = model1.params['Alpha']
        beta2 = model1.params['Alpha_x_Negative']
        
        print(f"\n✓ Baseline Results:")
        print(f"  Alpha (β₁): {beta1:.4f} (t={model1.tvalues['Alpha']:.2f})")
        print(f"  Alpha × Negative (β₂): {beta2:.4f} (t={model1.tvalues['Alpha_x_Negative']:.2f})")
        print(f"  R²: {model1.rsquared:.4f}")
        print(f"  N: {int(model1.nobs)}")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        results['baseline'] = None
    
    # ========================================================================
    # MODEL 2: Macro Regime Interactions: MAIN REGRESSION CONFIGURATION
    # ========================================================================
    print("\n" + "-"*80)
    print("MODEL 2: MACRO REGIME INTERACTIONS CONFIGURATION")
    print("-"*80)
    
    try:
        # Create interactions for each regime
        for regime in df['macro_regime'].unique():
            if regime != 'Unknown':
                df[f'Alpha_x_{regime}'] = df['Alpha'] * (df['macro_regime'] == regime).astype(int)
                df[f'AlphaNeg_x_{regime}'] = df['Alpha_x_Negative'] * (df['macro_regime'] == regime).astype(int)

        # Optionally lag all alpha columns by ALPHA_LAG quarters
        if ALPHA_LAG > 0:
            print(f"\n  Applying alpha lag: α_{{t-{ALPHA_LAG}}} used instead of α_t")
            df = df.sort_values(['Instrument', 'Date'])
            _regimes = [r for r in df['macro_regime'].unique() if r != 'Unknown']
            _alpha_cols = (
                ['Alpha', 'Alpha_x_Negative']
                + [f'Alpha_x_{r}' for r in _regimes]
                + [f'AlphaNeg_x_{r}' for r in _regimes]
            )
            for col in _alpha_cols:
                if col in df.columns:
                    df[col] = df.groupby('Instrument')[col].shift(ALPHA_LAG)
            df = df.dropna(subset=['Alpha'])
            print(f"  N after alpha lag drop: {len(df):,}")

        # Full model specification:
        # Flow_Rate = β₀ + β₄*Alpha×Goldilocks + β₅*Alpha×Overheating + β₆*Alpha×Downturn + β₇*Alpha×Stagflation
        #           + β₈*AlphaNeg×Goldilocks + β₉*AlphaNeg×Overheating + β₁₀*AlphaNeg×Downturn + β₁₁*AlphaNeg×Stagflation
        #           + γ₁*Lagged_Flow + γ₂*Log_TNA_lag + γ₃*TER
        #           + C(YearQuarter)   [quarter FEs absorb regime main effects + secular trends]
        #           + ε
        #
        # Design notes:
        # (1) C(macro_regime) is NOT included — it is perfectly collinear with C(YearQuarter)
        #     because each calendar quarter belongs to exactly one macro regime.  Regime main
        #     effects are absorbed by the quarter FEs (Goldstein, Jiang & Ng 2017).
        # (2) Baseline 'Alpha' and 'Alpha_x_Negative' are NOT included.
        #     For every observation with a known regime:
        #       Alpha_x_G + Alpha_x_O + Alpha_x_D + Alpha_x_S = Alpha  (exactly)
        #     Including both the baseline and all four interactions causes perfect
        #     multicollinearity — statsmodels would silently drop one regime, making
        #     the extraction ambiguous.  Dropping the baselines gives each regime its own
        #     direct alpha sensitivity coefficient (not a differential from a reference).
        # (3) Alpha_Negative = I(Alpha<0) is NOT included: the piecewise slope interactions
        #     AlphaNeg_x_{regime} already span negative-alpha observations, symmetric to how
        #     Alpha_x_{regime} spans positive-alpha observations without a separate level indicator.

        model2 = smf.ols(
            '''Flow_Rate ~ Alpha_x_Goldilocks + Alpha_x_Overheating + Alpha_x_Downturn + Alpha_x_Stagflation
            + AlphaNeg_x_Goldilocks + AlphaNeg_x_Overheating + AlphaNeg_x_Downturn + AlphaNeg_x_Stagflation
            + Lagged_Flow + Log_TNA_lag + Log_Age + TER
            + C(YearQuarter)''',
            data=df
        ).fit(cov_type='cluster', cov_kwds={'groups': df['Instrument']})
        
        results['macro_regime'] = model2
        
        print(f"\n✓ Macro Regime Results:")
        print(f"  R²: {model2.rsquared:.4f}")
        print(f"  N: {int(model2.nobs)}")
        
        # Print baseline coefficients
        print("\n  Baseline Coefficients (pooled across all regimes):")
        print("  " + "-"*60)
        if 'Alpha' in model2.params:
            beta = model2.params['Alpha']
            tval = model2.tvalues['Alpha']
            pval = model2.pvalues['Alpha']
            sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
            print(f"  Alpha:          β = {beta:7.3f} (t={tval:6.2f}){sig}")
        
        if 'Alpha_x_Negative' in model2.params:
            beta = model2.params['Alpha_x_Negative']
            tval = model2.tvalues['Alpha_x_Negative']
            pval = model2.pvalues['Alpha_x_Negative']
            sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
            print(f"  Alpha×Negative: β = {beta:7.3f} (t={tval:6.2f}){sig}")
        
        print("\n  Flow Sensitivity to POSITIVE Alpha by Regime:")
        print("  " + "-"*60)
        for regime in ['Goldilocks', 'Overheating', 'Downturn', 'Stagflation']:
            var_name = f'Alpha_x_{regime}'
            if var_name in model2.params:
                beta = model2.params[var_name]
                tval = model2.tvalues[var_name]
                pval = model2.pvalues[var_name]
                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                print(f"  {regime:15s}: β = {beta:7.3f} (t={tval:6.2f}){sig}")
        
        print("\n  Additional Sensitivity for NEGATIVE Alpha by Regime:")
        print("  " + "-"*60)
        for regime in ['Goldilocks', 'Overheating', 'Downturn', 'Stagflation']:
            var_name = f'AlphaNeg_x_{regime}'
            if var_name in model2.params:
                beta = model2.params[var_name]
                tval = model2.tvalues[var_name]
                pval = model2.pvalues[var_name]
                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                print(f"  {regime:15s}: β = {beta:7.3f} (t={tval:6.2f}){sig}")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        results['macro_regime'] = None

    # ========================================================================
    # MODEL 3: Continuous macro interactions
    # ========================================================================
    print("\n" + "-"*80)
    print("MODEL 3: CONTINUOUS GROWTH × INFLATION INTERACTIONS")
    print("-"*80)

    try:
        # growth and inflation are z-scored composites already in df (from merge).
        # Hierarchy principle: include Alpha, Alpha_Negative, Alpha_x_Negative as
        # main effects — unlike Model 2 they are NOT spanned by the interactions.
        # Standalone growth/inflation are omitted: perfectly absorbed by C(YearQuarter).
        df['Alpha_x_growth']      = df['Alpha'] * df['growth']
        df['Alpha_x_inflation']   = df['Alpha'] * df['inflation']
        df['AlphaNeg_x_growth']   = df['Alpha_x_Negative'] * df['growth']
        df['AlphaNeg_x_inflation'] = df['Alpha_x_Negative'] * df['inflation']

        model3 = smf.ols(
            '''Flow_Rate ~ Alpha + Alpha_Negative + Alpha_x_Negative
            + Alpha_x_growth + Alpha_x_inflation
            + AlphaNeg_x_growth + AlphaNeg_x_inflation
            + Lagged_Flow + Log_TNA_lag + Log_Age + TER
            + C(YearQuarter)''',
            data=df
        ).fit(cov_type='cluster', cov_kwds={'groups': df['Instrument']})

        results['continuous'] = model3

        print(f"\n✓ Continuous Macro Results:")
        print(f"  R²: {model3.rsquared:.4f}")
        print(f"  N: {int(model3.nobs)}")

        def _fmt3(param):
            if param not in model3.params:
                return ''
            b = model3.params[param]
            t = model3.tvalues[param]
            p = model3.pvalues[param]
            sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
            return f"β = {b:7.3f} (t={t:6.2f}){sig}"

        print("\n  Baseline alpha sensitivity (at growth=0, inflation=0):")
        print("  " + "-"*60)
        print(f"  Alpha (pos):        {_fmt3('Alpha')}")
        print(f"  Alpha_x_Negative:   {_fmt3('Alpha_x_Negative')}")
        print(f"  Alpha_Negative:     {_fmt3('Alpha_Negative')}")

        print("\n  Macro modulation of alpha sensitivity:")
        print("  " + "-"*60)
        for param, label in [
            ('Alpha_x_growth',      'α(+) × Growth     '),
            ('Alpha_x_inflation',   'α(+) × Inflation  '),
            ('AlphaNeg_x_growth',   'α(−) × Growth     '),
            ('AlphaNeg_x_inflation','α(−) × Inflation  '),
        ]:
            print(f"  {label}: {_fmt3(param)}")

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        results['continuous'] = None

    # ========================================================================
    # MODEL 4: Piecewise alpha + direct macro indicators (year FEs)
    # ========================================================================
    print("\n" + "-"*80)
    print("MODEL 4: PIECEWISE ALPHA + DIRECT MACRO INDICATORS (Fund FEs)")
    print("-"*80)
    print("  Note: growth/inflation absorbed by C(YearQuarter) → fund FEs used instead.")
    print("  Within-fund identification: how does a fund's flows respond when macro changes?")

    try:
        model4 = smf.ols(
            '''Flow_Rate ~ Alpha + Alpha_x_Negative + Alpha_Negative
            + growth + inflation
            + Lagged_Flow + Log_TNA_lag + Log_Age + TER
            + C(Instrument)''',
            data=df
        ).fit(cov_type='cluster', cov_kwds={'groups': df['Instrument']})

        results['direct_macro'] = model4

        print(f"\n✓ Direct Macro Results:")
        print(f"  R²: {model4.rsquared:.4f}")
        print(f"  N: {int(model4.nobs)}")

        def _fmt4(param):
            if param not in model4.params:
                return ''
            b = model4.params[param]
            t = model4.tvalues[param]
            p = model4.pvalues[param]
            sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
            return f"β = {b:7.3f} (t={t:6.2f}){sig}"

        print("\n  Alpha sensitivity (piecewise):")
        print("  " + "-"*60)
        for param, label in [
            ('Alpha',            'Alpha (pos)       '),
            ('Alpha_x_Negative', 'Alpha_x_Negative  '),
            ('Alpha_Negative',   'Alpha_Negative    '),
        ]:
            print(f"  {label}: {_fmt4(param)}")

        print("\n  Direct macro effects on flows:")
        print("  " + "-"*60)
        for param, label in [
            ('growth',    'Growth    '),
            ('inflation', 'Inflation '),
        ]:
            print(f"  {label}: {_fmt4(param)}")

        print("\n  Controls:")
        print("  " + "-"*60)
        for param, label in [
            ('Lagged_Flow', 'Lagged_Flow'),
            ('Log_TNA_lag', 'Log_TNA_lag'),
            ('Log_Age',     'Log_Age    '),
            ('TER',         'TER        '),
        ]:
            print(f"  {label}: {_fmt4(param)}")

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        results['direct_macro'] = None

    # ========================================================================
    # MODEL 5: Goldstein differential parameterisation + Fund FEs
    # ========================================================================
    # Skipped when --skip_model5 is passed (large asset classes, comparison table only).
    if SKIP_MODEL5:
        print("\n" + "-"*80)
        print("MODEL 5: SKIPPED (--skip_model5)")
        results['regime_fund_fe'] = None
        return results, df

    # Specification (Goldstein-consistent):
    #   Flow = α_fund + δ_regime + β_α·Alpha + β_αneg·Alpha_x_Negative
    #          + Δβ_α(regime)·Alpha + Δβ_αneg(regime)·Alpha_x_Negative
    #          + β_neg·Alpha_Negative + controls
    #
    # Patsy encodes C(macro_regime) with Downturn as reference (alphabetically
    # first), so the uninteracted Alpha / Alpha_x_Negative coefficients are the
    # Downturn baseline slopes and the interaction terms are regime *deviations*.
    # ========================================================================
    print("\n" + "-"*80)
    print("MODEL 5: GOLDSTEIN DIFFERENTIAL PARAMETERISATION (Fund FEs)")
    print("-"*80)
    print("  Baseline regime = Downturn (alphabetically first → Patsy reference)")
    print("  Interaction terms = deviation from Downturn baseline")
    print("  C(Instrument) absorbs time-invariant fund characteristics")

    try:
        model5 = smf.ols(
            '''Flow_Rate ~ Alpha + Alpha_x_Negative + Alpha_Negative
            + C(macro_regime)
            + Alpha:C(macro_regime)
            + Alpha_x_Negative:C(macro_regime)
            + Lagged_Flow + Log_TNA_lag + Log_Age + TER
            + C(Instrument)''',
            data=df
        ).fit(cov_type='cluster', cov_kwds={'groups': df['Instrument']})

        results['regime_fund_fe'] = model5

        print(f"\n✓ Model 5 Results:")
        print(f"  R²: {model5.rsquared:.4f}")
        print(f"  N: {int(model5.nobs)}")

        def _fmt5(param):
            if param not in model5.params:
                return 'n/a'
            b = model5.params[param]
            t = model5.tvalues[param]
            p = model5.pvalues[param]
            sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
            return f"β = {b:7.3f} (t={t:6.2f}){sig}"

        def _get_b(param):
            return model5.params.get(param, 0.0)

        # Downturn is the reference regime (alphabetically first)
        ref = sorted(df['macro_regime'].dropna().unique())[0]   # = "Downturn"
        non_ref = [r for r in ['Goldilocks', 'Overheating', 'Stagflation'] if r != ref]

        print(f"\n  ── Goldstein baseline (regime = {ref}) ──────────────────────────")
        print(f"  Alpha (β_α,base)          : {_fmt5('Alpha')}")
        print(f"  Alpha_x_Negative (Δ<0)    : {_fmt5('Alpha_x_Negative')}")
        print(f"  Alpha_Negative (indicator) : {_fmt5('Alpha_Negative')}")

        print(f"\n  ── Regime level shifts on flows (ref = {ref}) ───────────────────")
        print("  " + "-"*60)
        for regime in ['Goldilocks', 'Overheating', 'Stagflation']:
            param = f'C(macro_regime)[T.{regime}]'
            print(f"  Δflow {regime:15s}: {_fmt5(param)}")

        print(f"\n  ── Differential α slope (deviation from {ref}) ──────────────────")
        print("  " + "-"*60)
        for regime in non_ref:
            param = f'Alpha:C(macro_regime)[T.{regime}]'
            print(f"  Δβ_α  {regime:15s}: {_fmt5(param)}")

        print(f"\n  ── Differential α⁻ slope (deviation from {ref}) ─────────────────")
        print("  " + "-"*60)
        for regime in non_ref:
            param = f'Alpha_x_Negative:C(macro_regime)[T.{regime}]'
            print(f"  Δβ_α⁻ {regime:15s}: {_fmt5(param)}")

        print(f"\n  ── Implied total α slope per regime (base + differential) ───────")
        print("  " + "-"*60)
        b_base    = _get_b('Alpha')
        b_neg_base = _get_b('Alpha_x_Negative')
        print(f"  {ref:15s}: α slope = {b_base:7.3f}  │  α⁻ add. = {b_neg_base:7.3f}")
        for regime in non_ref:
            d_a   = _get_b(f'Alpha:C(macro_regime)[T.{regime}]')
            d_an  = _get_b(f'Alpha_x_Negative:C(macro_regime)[T.{regime}]')
            print(f"  {regime:15s}: α slope = {b_base + d_a:7.3f}  │  α⁻ add. = {b_neg_base + d_an:7.3f}")

        print("\n  ── Controls ─────────────────────────────────────────────────────")
        print("  " + "-"*60)
        for param, label in [
            ('Lagged_Flow', 'Lagged_Flow'),
            ('Log_TNA_lag', 'Log_TNA_lag'),
            ('Log_Age',     'Log_Age    '),
            ('TER',         'TER        '),
        ]:
            print(f"  {label}: {_fmt5(param)}")

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        results['regime_fund_fe'] = None

    return results, df


def save_macro_results(results, df):
    """
    Save all results
    """
    print("\n" + "="*80)
    print("STEP 13: SAVING MACRO RESULTS")
    print("="*80)
    
    # Save full regression output
    with open('goldstein_macro_results.txt', 'w') as f:
        f.write("GOLDSTEIN REGRESSION WITH MACRO REGIMES\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for model_name, model in results.items():
            if model is not None:
                f.write("\n" + "="*80 + "\n")
                f.write(f"{model_name.upper()}\n")
                f.write("="*80 + "\n\n")
                f.write(str(model.summary()))
                f.write("\n\n")
    
    print("✓ Saved full regression results to: goldstein_macro_results.txt")
    
    # Save data with macro regimes
    output_cols = ['Date', 'Instrument', 'Flow_Rate', 'Alpha', 'Return', 'TNA',
                   'macro_regime', 'growth', 'inflation']
    df[output_cols].to_csv(f'{_OUT_DIR}/fund_flows_with_macro{RUN_TAG}.csv', index=False)
    print(f"✓ Saved data with macro regimes to: {_OUT_DIR}/fund_flows_with_macro{RUN_TAG}.csv")
    
    # Create summary table
    summary_data = []
    for model_name, model in results.items():
        if model is not None:
            summary_data.append({
                'Model': model_name,
                'N': int(model.nobs),
                'R_squared': model.rsquared,
                'Adj_R_squared': model.rsquared_adj
            })



def extract_model5_coefficients(macro_results):
    """
    Extract coefficients from Model 5 (Goldstein differential + Fund FEs).

    Model 5 formula:
        Flow ~ Alpha + Alpha_x_Negative + Alpha_Negative
             + C(macro_regime)
             + Alpha:C(macro_regime)
             + Alpha_x_Negative:C(macro_regime)
             + Lagged_Flow + Log_TNA_lag + Log_Age + TER + C(Instrument)

    Patsy uses Downturn as the reference (alphabetically first), so:
        Alpha             = Downturn baseline α slope
        Alpha_x_Negative  = Downturn baseline α⁻ additional slope
        Alpha_Negative    = constant level shift for any α < 0 period
        Alpha:C(macro_regime)[T.X]           = deviation of α slope in regime X
        Alpha_x_Negative:C(macro_regime)[T.X] = deviation of α⁻ slope in regime X
        C(macro_regime)[T.X]                   = flow level shift in regime X

    Stored pkl structure (simulation-compatible):
        coefficients['macro_regime']['regimes'][regime] = {
            'alpha_pos':  total slope for α ≥ 0 (base + differential),
            'alpha_neg':  total slope for α < 0  (base + differential, both terms),
            'flow_level': regime intercept shift δ_r (0 for Downturn),
        }
        coefficients['macro_regime']['controls'] = {
            'intercept':                0.0  (fund FE absorbed; overwritten by flow_rate_mean),
            'alpha_negative_indicator': β for Alpha_Negative (level shift when α < 0),
            'lagged_flow', 'log_tna', 'log_age', 'ter',
        }
    """

    model5 = macro_results.get('regime_fund_fe')
    if model5 is None:
        raise ValueError("Model 5 (regime_fund_fe) not found or failed to run")

    regimes = ['Goldilocks', 'Overheating', 'Downturn', 'Stagflation']
    ref = sorted(regimes)[0]   # = 'Downturn'

    # Baseline slopes (= Downturn slopes in the differential parameterisation)
    b_alpha    = float(model5.params.get('Alpha', 0))
    b_alpha_xn = float(model5.params.get('Alpha_x_Negative', 0))
    b_alpha_n  = float(model5.params.get('Alpha_Negative', 0))

    coefficients = {
        'macro_regime': {
            'baseline': {
                'alpha':                    b_alpha,
                'alpha_x_negative':         b_alpha_xn,
                'alpha_negative_indicator': b_alpha_n,
                'reference_regime':         ref,
            },
            'regimes':  {},
            'controls': {},
        }
    }

    for regime in regimes:
        if regime == ref:
            d_alpha    = 0.0
            d_alpha_xn = 0.0
            flow_level = 0.0
        else:
            d_alpha    = float(model5.params.get(f'Alpha:C(macro_regime)[T.{regime}]', 0))
            d_alpha_xn = float(model5.params.get(f'Alpha_x_Negative:C(macro_regime)[T.{regime}]', 0))
            flow_level = float(model5.params.get(f'C(macro_regime)[T.{regime}]', 0))

        coefficients['macro_regime']['regimes'][regime] = {
            'alpha_pos':  b_alpha + d_alpha,
            'alpha_neg':  b_alpha + d_alpha + b_alpha_xn + d_alpha_xn,
            'flow_level': flow_level,
        }

    # Fund FEs absorbed: intercept is set to 0 here; overwritten by flow_rate_mean in pipeline
    coefficients['macro_regime']['controls'] = {
        'intercept':                 0.0,
        'alpha_negative_indicator':  b_alpha_n,
        'lagged_flow': float(model5.params.get('Lagged_Flow', 0)),
        'log_tna':     float(model5.params.get('Log_TNA_lag', 0)),
        'log_age':     float(model5.params.get('Log_Age', 0)),
        'ter':         float(model5.params.get('TER', 0)),
    }

    # ── Model statistics ───────────────────────────────────────────────────────
    coefficients['model_stats'] = {
        'r_squared':     model5.rsquared,
        'adj_r_squared': model5.rsquared_adj,
        'n_obs':         int(model5.nobs),
        'f_statistic':   float(model5.fvalue),
        'f_pvalue':      float(model5.f_pvalue),
        'aic':           float(model5.aic),
        'bic':           float(model5.bic),
        'df_model':      int(model5.df_model),
        'df_resid':      int(model5.df_resid),
    }

    # ── Full inference stats for every param (se, t, p, CI) ───────────────────
    _ci = model5.conf_int()
    def _infer(param):
        if param not in model5.params:
            return {'coef': 0.0, 'se': None, 't': None, 'p': None,
                    'ci_lower': None, 'ci_upper': None}
        return {
            'coef':     float(model5.params[param]),
            'se':       float(model5.bse[param]),
            't':        float(model5.tvalues[param]),
            'p':        float(model5.pvalues[param]),
            'ci_lower': float(_ci.loc[param, 0]),
            'ci_upper': float(_ci.loc[param, 1]),
        }
    coefficients['inference'] = {name: _infer(name) for name in model5.params.index}

    # ── All-models stats ───────────────────────────────────────────────────────
    def _mstats(m):
        return {
            'n_obs':         int(m.nobs),
            'r_squared':     m.rsquared,
            'adj_r_squared': m.rsquared_adj,
            'f_statistic':   float(m.fvalue),
            'f_pvalue':      float(m.f_pvalue),
            'aic':           float(m.aic),
            'bic':           float(m.bic),
            'df_model':      int(m.df_model),
            'df_resid':      int(m.df_resid),
        }
    coefficients['all_models_stats'] = {
        mname: _mstats(m)
        for mname, m in macro_results.items() if m is not None
    }

    print("\n" + "-"*80)
    print("Model 5 Statistics:")
    print("-"*80)
    print(f"R²:           {model5.rsquared:.4f}")
    print(f"Adj. R²:      {model5.rsquared_adj:.4f}")
    print(f"N:            {int(model5.nobs):,}")
    print(f"F-stat:       {model5.fvalue:.2f}  (p={model5.f_pvalue:.4f})")
    print(f"AIC:          {model5.aic:.1f}    BIC: {model5.bic:.1f}")

    return coefficients

def save_coefficients(coefficients, filepath='goldstein_model2_coefficients.pkl'):
    """
    Save coefficients to pickle file
    """
    
    print("\n" + "="*80)
    print("SAVING COEFFICIENTS")
    print("="*80)
    
    # Save as pickle (for Python use)
    with open(filepath, 'wb') as f:
        pickle.dump(coefficients, f)
    print(f"✓ Saved to: {filepath}")
    
    # Also save as JSON for readability (includes inference + all_models_stats)
    import json
    json_path = filepath.replace('.pkl', '.json')
    with open(json_path, 'w') as f:
        json.dump(coefficients, f, indent=2)
    print(f"✓ Saved to: {json_path} (human-readable)")


def display_coefficient_summary(coefficients):
    """Print a summary table for Model 5 (Goldstein differential + Fund FEs)."""

    print("\n" + "="*80)
    print("MODEL 5 COEFFICIENT SUMMARY (FOR THESIS TABLE)")
    print("="*80)

    inf = coefficients.get('inference', {})
    def _fmt(param, coef_val):
        s = inf.get(param)
        stars = ''
        se_str = ''
        if s and s.get('p') is not None:
            p = s['p']
            stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
            se_str = f"  (se={s['se']:.3f}, t={s['t']:.2f})"
        return f"{coef_val:>8.3f}{stars:<3}{se_str}"

    baseline = coefficients['macro_regime']['baseline']
    ref      = baseline.get('reference_regime', 'Downturn')

    print(f"\nBaseline Goldstein parameters (reference regime = {ref}):")
    print("-"*80)
    print(f"  {'Alpha (β_α,base)':<32} {_fmt('Alpha',            baseline['alpha'])}")
    print(f"  {'Alpha_x_Negative (Δ slope, α<0)':<32} {_fmt('Alpha_x_Negative', baseline['alpha_x_negative'])}")
    print(f"  {'Alpha_Negative (level, α<0 indicator)':<32} {_fmt('Alpha_Negative',   baseline['alpha_negative_indicator'])}")

    print(f"\nRegime level shifts on flows (vs {ref}):")
    print("-"*80)
    for regime in ['Goldilocks', 'Overheating', 'Stagflation']:
        param = f'C(macro_regime)[T.{regime}]'
        val   = coefficients['macro_regime']['regimes'][regime]['flow_level']
        print(f"  {'Δflow ' + regime:<32} {_fmt(param, val)}")
    print(f"  {'Δflow ' + ref:<32}   (reference = 0.000)")

    print(f"\nImplied total α slopes per regime (base + differential):")
    print(f"  {'Regime':<15} {'α≥0 slope':>12}  {'α<0 total':>12}  {'flow_level':>12}")
    print("  " + "-"*54)
    for regime in ['Goldilocks', 'Overheating', 'Downturn', 'Stagflation']:
        c = coefficients['macro_regime']['regimes'][regime]
        print(f"  {regime:<15} {c['alpha_pos']:>12.3f}  {c['alpha_neg']:>12.3f}  {c['flow_level']:>12.3f}")

    print("\nControl Variables:")
    print("-"*80)
    controls = coefficients['macro_regime']['controls']
    _ctrl_map = {
        'lagged_flow': 'Lagged_Flow',
        'log_tna':     'Log_TNA_lag',
        'log_age':     'Log_Age',
        'ter':         'TER',
    }
    for ctrl_key, param_name in _ctrl_map.items():
        if ctrl_key in controls:
            print(f"  {ctrl_key:<32} {_fmt(param_name, controls[ctrl_key])}")
    print("  (*** p<0.01  ** p<0.05  * p<0.10)")


# %%
# =============================================================================
# UNIFIED EXECUTION CELL — change config here
# =============================================================================
#
# ASSET CLASS FILTER OPTIONS
#
# Option A — exact match
#   ASSET_CLASSES = ['Bond USD High Yield', 'Bond USD Medium Term', ...]
#   Available bond classes examples:
#     'Bond USD High Yield'         'Bond USD Medium Term'
#     'Bond USD Municipal'          'Bond USD Short Term'
#   Available equity classes examples:
#     'Equity US'  'Equity US Sm&Mid Cap'  'Equity Global'
#   Mixed / money market:
#     'Mixed Asset USD Conservative'  'Money Market USD'
#
# Option B — keyword match (only used when ASSET_CLASSES is None):
#   ASSET_KEYWORDS = ['bond', 'debt', 'fixed income']
#
# To run on ALL funds: set both to None.
# =============================================================================

ASSET_CLASSES  = None   # exact-match list; set manually when needed
ASSET_KEYWORDS = [k.strip() for k in _args.keyword.split(',') if k.strip()] or None

try:
    print("\n" + "="*80)
    print("STARTING PIPELINE")
    print("="*80)

    # --- Step 1: Apply asset class filter to df_raw ---
    fund_types = load_fund_types()
    df = filter_by_asset_class(df_raw, fund_types,
                               asset_classes=ASSET_CLASSES,
                               keywords=ASSET_KEYWORDS)

    # --- Step 2: Load balanced macro regimes (run new_macro_indicators.py first) ---
    macro_regimes = load_ilmanen_macro_regimes()
    macro_regimes.to_csv(f'{_OUT_DIR}/macro_regimes{RUN_TAG}.csv', index=False)
    print(f"\n✓ Saved macro regimes to: {_OUT_DIR}/macro_regimes{RUN_TAG}.csv")

    # --- Step 3: Merge, clean, regress ---
    df_with_macro = merge_macro_with_funds(df, macro_regimes)
    df_clean = clean_data(df_with_macro)
    macro_results, df_final = goldstein_macro_regressions(df_clean)
    save_macro_results(macro_results, df_final)

    # --- Step 4: Extract and save coefficients ---
    coefficients = extract_model5_coefficients(macro_results)

    # Store the empirical long-run mean flow rate from the clean training data.
    # Used by run_simulation_pipeline.py to replace the regression constant (beta_0),
    # which absorbs secular industry trends and is not meaningful out-of-sample.
    # Following Ben-David et al. (2022) and Zhu (2018): the simulation intercept is
    # set to this mean so that baseline flows reflect the structural steady-state.
    flow_rate_mean = float(df_clean['Flow_Rate'].mean())
    coefficients['macro_regime']['controls']['flow_rate_mean'] = flow_rate_mean
    print(f"\n  Long-run Flow_Rate mean (training): {flow_rate_mean:.4f} ({flow_rate_mean:.2%}/qtr)")
    print(f"  (Simulation will use this as intercept instead of the raw β₀={coefficients['macro_regime']['controls']['intercept']:.4f})")

    # Store training-data mean of log(TNA_lag) so the simulation can centre the
    # log_tna term around it.  The raw OLS β₀ was calibrated to offset the large
    # negative contribution of log_tna_coef × mean_log_tna; after I replace beta_0
    # with flow_rate_mean we must add that offset back via the intercept
    mean_log_tna = float(df_clean['Log_TNA_lag'].mean())
    coefficients['macro_regime']['controls']['mean_log_tna'] = mean_log_tna
    print(f"  Mean log(TNA_lag) in training data: {mean_log_tna:.4f}  "
          f"(TNA ≈ €{np.exp(mean_log_tna)/1e6:.0f}M)")
    print(f"  (Simulation will centre log_tna around this value to remove level drag)")

    save_coefficients(coefficients, filepath=f'{_OUT_DIR}/goldstein_model2_coefficients{RUN_TAG}.pkl')
    display_coefficient_summary(coefficients)

    _filter_label = ASSET_CLASSES or ASSET_KEYWORDS or 'ALL funds'
    print("\n" + "="*80)
    print(f"✅ PIPELINE COMPLETE — pickle saved | filter: {_filter_label}")
    print("="*80)

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()


# %%
