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
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import pickle


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


# %%
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

# %%
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
    
    return tna


# %%
def load_fund_characteristics(returns):
    """
    Load TER and inception dates
    
    Strategy for inception dates:
    1. Try to use inception date from TER files
    2. If missing, use first return date as fallback (since returns are "since inception")
    
    Parameters:
    -----------
    returns : DataFrame
        Returns data with columns ['Instrument', 'Date', 'Return']
    
    Returns:
    --------
    DataFrame with columns ['Instrument', 'TER', 'Inception_Date']
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

# %%

# %%
def load_fund_types(filepath='../Clean_funds_with_asset_class.csv'):
    """
    Load fund type / asset class information from Clean_funds_with_asset_class.csv.

    Key columns used:
    - FundClassLipperID  → joined as 'Instrument' to the regression dataset
    - IssueLipperGlobalSchemeName → 36 asset class labels (e.g. 'Bond USD High Yield')
    """
    print("\n" + "="*80)
    print("LOADING FUND TYPE DATA")
    print("="*80)

    fund_types = pd.read_csv(filepath)
    fund_types = fund_types.dropna(subset=['FundClassLipperID'])   # 9 rows have no ID — unusable for join
    fund_types['Instrument'] = fund_types['FundClassLipperID'].astype(int).astype(str)
    fund_types = fund_types[['Instrument', 'IssueLipperGlobalSchemeName']]

   

    return fund_types


def filter_by_asset_class(df, fund_types, asset_classes=None, keywords=None):
    """
    Filter fund data to selected asset classes from IssueLipperGlobalSchemeName.

    Parameters
    ----------
    df : DataFrame
        Main regression dataset (must contain 'Instrument' column).
    fund_types : DataFrame
        Output of load_fund_types() — columns: 'Instrument', 'IssueLipperGlobalSchemeName'.
    asset_classes : list of str, optional
        Exact IssueLipperGlobalSchemeName values to keep.
        e.g. ['Bond USD High Yield', 'Bond USD Medium Term']
        Takes priority over keywords. Pass None to fall back to keyword matching.
    keywords : list of str, optional
        Case-insensitive regex keywords matched against IssueLipperGlobalSchemeName.
        e.g. ['bond', 'debt', 'fixed income']
        Only used when asset_classes is None.
        Pass None (together with asset_classes=None) to skip filtering entirely.

    Returns
    -------
    DataFrame : filtered to selected asset classes, with 'IssueLipperGlobalSchemeName'
                column appended.
    """
    print("\n" + "="*80)
    print("ASSET CLASS FILTER")
    print("="*80)

    # Merge asset class labels onto main dataset
    df = df.merge(fund_types, on='Instrument', how='left')

    n_before = len(df)
    funds_before = df['Instrument'].nunique()

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

    return df_filtered


# %%

def calculate_alpha_rolling(fund_returns, benchmarks, fund_chars, window=8):
    """
    Calculate rolling two year alpha for each fund
    Alpha = intercept from: Excess_Fund_Return ~ Excess_Bond + Excess_Stock
    
    This follows Goldstein et al. (2017) exactly
    
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


# %%

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
    
    # Data validation - check for extreme values
    print(f"\nData validation before cleaning:")
    print(f"  Flow_Rate: min={df['Flow_Rate'].min():.2f}, max={df['Flow_Rate'].max():.2f}, median={df['Flow_Rate'].median():.4f}")
    print(f"  TNA_lag: min={df['TNA_lag'].min():.2f}, max={df['TNA_lag'].max():.2e}")
    
    # Remove extreme Flow_Rate values (beyond +/- 200% - likely data errors)
    # Also remove rows where TNA_lag is extremely small (< $10,000)
    before = len(df)
    df = df[(df['Flow_Rate'] >= -2) & (df['Flow_Rate'] <= 2)]  # Keep only -200% to +200%
    df = df[df['TNA_lag'] >= 10000]  # Remove funds with TNA < $10k
    after = len(df)
    print(f"\n  Removed {before - after:,} rows with extreme values")
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


# %%
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
    
    # Drop rows missing essential simulation variables
    cols_to_check = ['Alpha', 'TER', 'Log_TNA_lag', 'Flow_Rate', 'macro_regime']
    df = df.dropna(subset=cols_to_check)
    
    # More aggressive cleaning before winsorization
    print(f"\nBefore aggressive cleaning: {len(df):,} rows")
    print(f"  Flow_Rate: min={df['Flow_Rate'].min():.2f}, max={df['Flow_Rate'].max():.2f}")
    print(f"  Alpha: min={df['Alpha'].min():.2f}, max={df['Alpha'].max():.2f}")
    print(f"  Log_TNA_lag: min={df['Log_TNA_lag'].min():.2f}, max={df['Log_TNA_lag'].max():.2f}")
    
    # Cap extreme values before winsorization
    df = df[df['Flow_Rate'] >= -2]  # Remove extreme negative flows
    df = df[df['Flow_Rate'] <= 2]   # Remove extreme positive flows
    df = df[df['Log_TNA_lag'] > 0]  # Log(TNA) must be positive
    df = df[df['Log_TNA_lag'] < 30] # Reasonable max for log(TNA)
    
    print(f"After aggressive cleaning: {len(df):,} rows")
    
    # WINSORIZATION: Cap the top and bottom 5% to neutralize outliers
    # This directly addresses the Skew: 136.8 and Kurtosis: 22990
    df['Flow_Rate'] = winsorize(df['Flow_Rate'], limits=[0.05, 0.05])
    df['Alpha'] = winsorize(df['Alpha'], limits=[0.05, 0.05])
    
    print(f"After winsorization:")
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
    df = df.sort_values(['Instrument', 'Date'])
    df['Lagged_Flow'] = df.groupby('Instrument')['Flow_Rate'].shift(1)
    df = df.dropna(subset=['Lagged_Flow'])
    
    print(f"✓ Cleaned dataset: {len(df):,} observations")
    print(f"  New Skewness (Flow_Rate): {df['Flow_Rate'].skew():.2f}")
    return df

# %%

# =============================================================================
# ASSET CLASS FILTER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Choose which fund types to include in the regression.
#
# Option A — exact match (recommended):
#   Set ASSET_CLASSES to a list of exact IssueLipperGlobalSchemeName values.
#   Available bond classes (20 categories, examples):
#     'Bond USD High Yield'         'Bond USD Medium Term'
#     'Bond USD Municipal'          'Bond USD Short Term'
#     'Bond USD Corporates'         'Bond USD Government'
#     'Bond USD Mortgages'          'Bond USD Inflation Linked'
#     'Bond Global USD'             'Bond Global High Yield USD'
#     'Bond Emerging Markets Global HC'
#   Available equity classes (11 categories, examples):
#     'Equity US'   'Equity US Sm&Mid Cap'   'Equity Global'
#     'Equity Emerging Mkts Global'   'Equity Sector Real Est US'
#   Mixed / money market:
#     'Mixed Asset USD Conservative'   'Mixed Asset USD Aggressive'
#     'Money Market USD'
#
# Option B — keyword matching (fuzzy):
#   Set ASSET_KEYWORDS to a list of case-insensitive patterns.
#   e.g. ['bond', 'debt', 'fixed income']   (only used when ASSET_CLASSES is None)
#
# To run on ALL funds without filtering:
#   Set both ASSET_CLASSES = None and ASSET_KEYWORDS = None
# =============================================================================

ASSET_CLASSES = None            # e.g. ['Bond USD High Yield', 'Bond USD Medium Term']
ASSET_KEYWORDS = ['bond']       # e.g. ['bond', 'debt', 'fixed income']
# =============================================================================

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

    # Step 6: Calculate flows
    df = calculate_flows(returns, tna, fund_chars, alphas)

    # Step 7: Filter by asset class (controlled by ASSET_CLASSES / ASSET_KEYWORDS above)
    fund_types = load_fund_types()
    df = filter_by_asset_class(df, fund_types, asset_classes=ASSET_CLASSES, keywords=ASSET_KEYWORDS)

except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


# %%

def download_macro_data_direct(start_date='1988-01-01', end_date='2024-12-31'):
    """
    Download macroeconomic data directly from FRED as CSV
    NO pandas-datareader required!
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
    print("\n1. Applying information lags...")
    print("   GDP: 2-month lag")
    print("   CPI: 1-month lag")
    df['GDP'] = df['GDP'].shift(2)
    df['CPI'] = df['CPI'].shift(1)
    
    # Forward-fill
    print("\n2. Forward-filling for monthly completeness...")
    df[['GDP', 'CPI', 'FedFundsRate', 'NBER']] = df[['GDP', 'CPI', 'FedFundsRate', 'NBER']].fillna(method='ffill')
    
    # YoY growth
    print("\n3. Calculating year-over-year growth rates...")
    df['GDP_YoY'] = df['GDP'].pct_change(periods=12)
    df['CPI_YoY'] = df['CPI'].pct_change(periods=12)
    
    # Growth regime
    print("\n4. Creating growth regime (NBER recession-based)...")
    df['growth_regime'] = df['NBER'].apply(lambda x: 'Low' if x == 1 else 'High')
    
    # Inflation regime
    print("\n5. Creating inflation regime...")
    df['CPI_YoY_lag'] = df['CPI_YoY'].shift(12)
    df['inflation_surprise'] = df['CPI_YoY'] - df['CPI_YoY_lag']
    df['inflation_regime'] = 'Low'
    df.loc[(df['CPI_YoY'] > 0.02) & (df['inflation_surprise'] > 0), 'inflation_regime'] = 'High'
    
    # Fed regime
    print("\n6. Creating Fed rate regime...")
    df['FedChange'] = df['FedFundsRate'].diff()
    df['FedRegime'] = 'Flat'
    df.loc[df['FedChange'] > 0, 'FedRegime'] = 'Hiking'
    df.loc[df['FedChange'] < 0, 'FedRegime'] = 'Cutting'
    
    # Macro regimes
    print("\n7. Creating classic macro regimes...")
    conditions = [
        (df['growth_regime'] == 'High') & (df['inflation_regime'] == 'Low'),
        (df['growth_regime'] == 'High') & (df['inflation_regime'] == 'High'),
        (df['growth_regime'] == 'Low') & (df['inflation_regime'] == 'Low'),
        (df['growth_regime'] == 'Low') & (df['inflation_regime'] == 'High'),
    ]
    choices = ['Goldilocks', 'Overheating', 'Downturn', 'Stagflation']
    df['macro_regime'] = np.select(conditions, choices, default='Unknown')
    
    df = df.dropna()
    
    # CRITICAL: Resample to QUARTERLY to match fund data!
    print("\n8. Resampling to QUARTERLY frequency (to match fund data)...")
    df = df.set_index('Date')
    
    # Resample to quarter-end, taking the last value of each quarter
    df_quarterly = df.resample('QE').last()
    
    # For categorical variables, use the mode (most common value) in the quarter
    # For numeric variables, use the last value
    df_quarterly['macro_regime'] = df.resample('QE')['macro_regime'].apply(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[-1]
    )
    df_quarterly['FedRegime'] = df.resample('QE')['FedRegime'].apply(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[-1]
    )
    
    df_quarterly = df_quarterly.reset_index()
    
    print(f"\n✓ Macro regimes created (QUARTERLY):")
    print(f"  Total observations: {len(df_quarterly)}")
    print(f"  Date range: {df_quarterly['Date'].min().date()} to {df_quarterly['Date'].max().date()}")
    
    print("\n  Macro Regime Distribution:")
    print(df_quarterly['macro_regime'].value_counts())
    
    print("\n  Fed Regime Distribution:")
    print(df_quarterly['FedRegime'].value_counts())
    
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
    
    fund_data = fund_data.sort_values('Date')
    macro_data = macro_data.sort_values('Date')
    
    print("\nUsing nearest date matching (within 45 days)...")
    
    df_merged = pd.merge_asof(
        fund_data,
        macro_data[['Date', 'GDP_YoY', 'CPI_YoY', 'FedFundsRate', 'FedChange',
                    'growth_regime', 'inflation_regime', 'FedRegime', 
                    'macro_regime', 'NBER']],
        on='Date',
        direction='nearest',
        tolerance=pd.Timedelta('45 days')
    )
    
    # Check merge quality
    missing_macro = df_merged[['macro_regime', 'FedRegime']].isnull().sum()
    print(f"\n✓ Merge complete:")
    print(f"  Total observations: {len(df_merged)}")
    print(f"  Missing macro regime: {missing_macro['macro_regime']}")
    print(f"  Missing Fed regime: {missing_macro['FedRegime']}")
    
    df_merged = df_merged.dropna(subset=['macro_regime', 'FedRegime'])
    
    print(f"  Final observations with macro data: {len(df_merged)}")
    print(f"  Unique funds: {df_merged['Instrument'].nunique()}")
    
    print("\n  Macro Regime Distribution in Fund Data:")
    print(df_merged['macro_regime'].value_counts())
    
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
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    
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
               + Lagged_Flow + Log_TNA_lag + TER + C(YearMonth)''',
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
    # MODEL 2: Macro Regime Interactions ⭐ MAIN RESULT
    # ========================================================================
    print("\n" + "-"*80)
    print("MODEL 2: MACRO REGIME INTERACTIONS ⭐ MAIN RESULT")
    print("-"*80)
    
    try:
        # Create interactions for each regime
        for regime in df['macro_regime'].unique():
            if regime != 'Unknown':
                df[f'Alpha_x_{regime}'] = df['Alpha'] * (df['macro_regime'] == regime).astype(int)
                df[f'AlphaNeg_x_{regime}'] = df['Alpha_x_Negative'] * (df['macro_regime'] == regime).astype(int)
        
        # Full model specification:
        # Flow_Rate = β₀ + β₁*Alpha + β₂*Alpha_Negative + β₃*Alpha×Negative
        #           + β₄*Alpha×Goldilocks + β₅*Alpha×Overheating + β₆*Alpha×Downturn + β₇*Alpha×Stagflation
        #           + β₈*AlphaNeg×Goldilocks + β₉*AlphaNeg×Overheating + β₁₀*AlphaNeg×Downturn + β₁₁*AlphaNeg×Stagflation
        #           + γ₁*Lagged_Flow + γ₂*Log_TNA_lag + γ₃*TER
        #           + δ*C(macro_regime)
        #           + ε
        
        model2 = smf.ols(
            '''Flow_Rate ~ Alpha + Alpha_Negative + Alpha_x_Negative
            + Alpha_x_Goldilocks + Alpha_x_Overheating + Alpha_x_Downturn + Alpha_x_Stagflation
            + AlphaNeg_x_Goldilocks + AlphaNeg_x_Overheating + AlphaNeg_x_Downturn + AlphaNeg_x_Stagflation
            + Lagged_Flow + Log_TNA_lag + TER
            + C(macro_regime)''',
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
    # MODEL 3: Fed Regime Interactions
    # ========================================================================
    print("\n" + "-"*80)
    print("MODEL 3: FED REGIME INTERACTIONS")
    print("-"*80)
    
    try:
        for regime in df['FedRegime'].unique():
            df[f'Alpha_x_Fed_{regime}'] = df['Alpha'] * (df['FedRegime'] == regime).astype(int)
            df[f'AlphaNeg_x_Fed_{regime}'] = df['Alpha_x_Negative'] * (df['FedRegime'] == regime).astype(int)
        
        # CRITICAL: Do NOT include C(YearMonth) - it causes multicollinearity with C(FedRegime)!
        model3 = smf.ols(
            '''Flow_Rate ~ Alpha_x_Fed_Hiking + Alpha_x_Fed_Cutting + Alpha_x_Fed_Flat
               + AlphaNeg_x_Fed_Hiking + AlphaNeg_x_Fed_Cutting + AlphaNeg_x_Fed_Flat
               + C(FedRegime)
               + Lagged_Flow + Log_TNA_lag + TER''',
            data=df
        ).fit(cov_type='cluster', cov_kwds={'groups': df['Instrument']})
        
        results['fed_regime'] = model3
        
        print(f"\n✓ Fed Regime Results:")
        print(f"  R²: {model3.rsquared:.4f}")
        print(f"  N: {int(model3.nobs)}")
        
        print("\n  Flow Sensitivity to POSITIVE Alpha by Fed Regime:")
        print("  " + "-"*60)
        for regime in ['Hiking', 'Cutting', 'Flat']:
            var_name = f'Alpha_x_Fed_{regime}'
            if var_name in model3.params:
                beta = model3.params[var_name]
                tval = model3.tvalues[var_name]
                pval = model3.pvalues[var_name]
                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                print(f"  {regime:10s}: β = {beta:7.3f} (t={tval:6.2f}){sig}")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        results['fed_regime'] = None
    
    # ========================================================================
    # MODEL 4: Continuous Macro Variables (Robustness)
    # ========================================================================
    print("\n" + "-"*80)
    print("MODEL 4: CONTINUOUS MACRO VARIABLES (Robustness)")
    print("-"*80)
    
    try:
        df['Alpha_x_GDP'] = df['Alpha'] * df['GDP_YoY']
        df['Alpha_x_CPI'] = df['Alpha'] * df['CPI_YoY']
        df['Alpha_x_FedRate'] = df['Alpha'] * df['FedFundsRate']
        
        model4 = smf.ols(
            '''Flow_Rate ~ Alpha + Alpha_x_Negative + Alpha_Negative
               + Alpha_x_GDP + Alpha_x_CPI + Alpha_x_FedRate
               + GDP_YoY + CPI_YoY + FedFundsRate
               + Lagged_Flow + Log_TNA_lag + TER + C(YearMonth)''',
            data=df
        ).fit(cov_type='cluster', cov_kwds={'groups': df['Instrument']})
        
        results['continuous'] = model4
        
        print(f"\n✓ Continuous Macro Results:")
        print(f"  R²: {model4.rsquared:.4f}")
        print(f"  N: {int(model4.nobs)}")
        
        print("\n  Interaction Effects:")
        print("  " + "-"*60)
        for var in ['Alpha_x_GDP', 'Alpha_x_CPI', 'Alpha_x_FedRate']:
            if var in model4.params:
                beta = model4.params[var]
                tval = model4.tvalues[var]
                pval = model4.pvalues[var]
                sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                print(f"  {var:20s}: β = {beta:7.3f} (t={tval:6.2f}){sig}")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        results['continuous'] = None
    
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
                   'macro_regime', 'FedRegime', 'GDP_YoY', 'CPI_YoY', 'FedFundsRate']
    df[output_cols].to_csv('fund_flows_with_macro.csv', index=False)
    print("✓ Saved data with macro regimes to: fund_flows_with_macro.csv")
    
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
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('model_summary.csv', index=False)
    print("✓ Saved model summary to: model_summary.csv")




# %%
try:
    print("\n" + "="*80)
    print("STARTING MACRO ANALYSIS EXTENSION")
    print("="*80)
    
    # Download macro data (no pandas-datareader needed!)
    macro_raw = download_macro_data_direct(start_date='1988-01-01')
    
    # Create regimes
    macro_regimes = create_macro_regimes(macro_raw)
    macro_regimes.to_csv('macro_regimes.csv', index=False)
    print("\n✓ Saved macro regimes to: macro_regimes.csv")
    
    # Merge with fund data
    df_with_macro = merge_macro_with_funds(df, macro_regimes)

    # NOW clean the data (after macro merge)
    df_clean = clean_data(df_with_macro)
    
    # Run all 4 regressions
    macro_results, df_final = goldstein_macro_regressions(df_clean)

    # Save results
    save_macro_results(macro_results, df_final)
    
    print("\n" + "="*80)
    print("✅ MACRO ANALYSIS COMPLETE!")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ ERROR in macro analysis: {str(e)}")
    import traceback
    traceback.print_exc()

# %%
def extract_model2_coefficients(macro_results):
    """
    Extract coefficients from Model 2 (macro regime interactions)
    
    Returns:
    --------
    dict : Coefficients formatted for ELTIF simulation
    """
    
    model2 = macro_results['macro_regime']
    
    if model2 is None:
        raise ValueError("Model 2 (macro_regime) not found or failed to run")
    
    # Extract regime-specific coefficients
    coefficients = {
        'macro_regime': {
            'baseline': {},      # ← NEW: baseline coefficients
            'regimes': {},
            'controls': {}
        }
    }
    
    # Extract BASELINE coefficients (pooled across all regimes)
    coefficients['macro_regime']['baseline'] = {
        'alpha': model2.params.get('Alpha', 0),
        'alpha_negative': model2.params.get('Alpha_Negative', 0),
        'alpha_x_negative': model2.params.get('Alpha_x_Negative', 0)
    }
    
    # Extract flow-performance sensitivity by regime (INTERACTIONS)
    for regime in ['Goldilocks', 'Overheating', 'Downturn', 'Stagflation']:
        alpha_var = f'Alpha_x_{regime}'
        alpha_neg_var = f'AlphaNeg_x_{regime}'
        
        # Get coefficients (default to 0 if not in model)
        beta_alpha = model2.params.get(alpha_var, 0)
        beta_alpha_neg = model2.params.get(alpha_neg_var, 0)
        
        coefficients['macro_regime']['regimes'][regime] = {
            'alpha': beta_alpha,           # Interaction term
            'alpha_neg': beta_alpha_neg    # Interaction term
        }
    
    # Extract control variable coefficients
    # Try different possible parameter names
    intercept = (model2.params.get('Intercept', 0) or 
                 model2.params.get('const', 0) or 
                 model2.params.get('C(macro_regime)[Goldilocks]', 0))
    
    lagged_flow = model2.params.get('Lagged_Flow', model2.params.get('lagged_flow', 0))
    log_tna = model2.params.get('Log_TNA_lag', model2.params.get('log_tna_lag', 0))
    ter = model2.params.get('TER', model2.params.get('ter', 0))
    
    coefficients['macro_regime']['controls'] = {
        'intercept': intercept,
        'lagged_flow': lagged_flow,
        'log_tna': log_tna,
        'ter': ter
    }
    
    # Add model statistics
    coefficients['model_stats'] = {
        'r_squared': model2.rsquared,
        'adj_r_squared': model2.rsquared_adj,
        'n_obs': int(model2.nobs)
    }
    
    print("\n" + "-"*80)
    print("Model Statistics:")
    print("-"*80)
    print(f"R²:           {model2.rsquared:.4f}")
    print(f"Adj. R²:      {model2.rsquared_adj:.4f}")
    print(f"N:            {int(model2.nobs):,}")
    
    return coefficients

# %%
def save_coefficients(coefficients, filepath='goldstein_model2_coefficients.pkl'):
    """
    Save coefficients to pickle file
    
    Parameters:
    -----------
    coefficients : dict
        From extract_model2_coefficients()
    filepath : str
        Where to save the file
    """
    
    print("\n" + "="*80)
    print("SAVING COEFFICIENTS")
    print("="*80)
    
    # Save as pickle (for Python use)
    with open(filepath, 'wb') as f:
        pickle.dump(coefficients, f)
    print(f"✓ Saved to: {filepath}")
    
    # Also save as JSON for readability
    import json
    json_path = filepath.replace('.pkl', '.json')
    with open(json_path, 'w') as f:
        json.dump(coefficients, f, indent=2)
    print(f"✓ Saved to: {json_path} (human-readable)")
    
    # Create a summary CSV
    csv_path = filepath.replace('.pkl', '_summary.csv')
    
    rows = []
    
    # Add baseline coefficients
    baseline = coefficients['macro_regime']['baseline']
    rows.append({
        'Regime': 'BASELINE',
        'Beta_Alpha': baseline['alpha'],
        'Beta_Alpha_Neg': baseline['alpha_x_negative'],
        'Alpha_Negative_Indicator': baseline['alpha_negative'],
        'Type': 'Baseline (pooled)'
    })
    
    # Add regime-specific interactions
    for regime, coefs in coefficients['macro_regime']['regimes'].items():
        rows.append({
            'Regime': regime,
            'Beta_Alpha': coefs['alpha'],
            'Beta_Alpha_Neg': coefs['alpha_neg'],
            'Alpha_Negative_Indicator': '',
            'Type': 'Regime interaction'
        })
    
    # Add controls
    for control, value in coefficients['macro_regime']['controls'].items():
        rows.append({
            'Regime': control,
            'Beta_Alpha': value,
            'Beta_Alpha_Neg': '',
            'Alpha_Negative_Indicator': '',
            'Type': 'Control'
        })
    
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(csv_path, index=False)
    print(f"✓ Saved summary to: {csv_path}")


# %%
def display_coefficient_summary(coefficients):
    """
    Print a nice summary table
    """
    
    print("\n" + "="*80)
    print("MODEL 2 COEFFICIENT SUMMARY (FOR THESIS TABLE)")
    print("="*80)
    
    # Display baseline coefficients
    baseline = coefficients['macro_regime']['baseline']
    print("\n" + "-"*80)
    print("BASELINE COEFFICIENTS (Pooled across all regimes):")
    print("-"*80)
    print(f"Alpha (positive):        {baseline['alpha']:>8.3f}")
    print(f"Alpha_Negative:          {baseline['alpha_negative']:>8.3f}")
    print(f"Alpha × Negative:        {baseline['alpha_x_negative']:>8.3f}")
    
    # Display regime-specific effects
    print("\n" + "-"*80)
    print(f"{'Regime':<15} {'Interaction':<12} {'Interaction':<12} {'TOTAL':<12} {'TOTAL':<12}")
    print(f"{'':15} {'β(α>0)':<12} {'β(α<0)':<12} {'α>0':<12} {'α<0':<12}")
    print("-"*80)
    
    for regime in ['Goldilocks', 'Overheating', 'Downturn', 'Stagflation']:
        coefs = coefficients['macro_regime']['regimes'][regime]
        
        # Interaction coefficients
        beta1 = coefs['alpha']
        beta2 = coefs['alpha_neg']
        
        # TOTAL effects (baseline + interaction)
        total_pos = baseline['alpha'] + beta1
        total_neg = baseline['alpha'] + baseline['alpha_x_negative'] + beta1 + beta2
        
        print(f"{regime:<15} {beta1:>10.3f}   {beta2:>10.3f}   {total_pos:>10.3f}   {total_neg:>10.3f}")
    
    print("\n" + "-"*80)
    print("Interpretation:")
    print("-"*80)
    print("Interaction coefficients show ADDITIONAL effect in each regime")
    print("TOTAL effects = Baseline + Interaction")
    print("  For α>0: Total = baseline['alpha'] + regime['alpha']")
    print("  For α<0: Total = baseline['alpha'] + baseline['alpha_x_negative']")
    print("                   + regime['alpha'] + regime['alpha_neg']")
    
    print("\n" + "-"*80)
    print("Control Variables:")
    print("-"*80)
    controls = coefficients['macro_regime']['controls']
    print(f"Intercept:     {controls['intercept']:>8.3f}")
    print(f"Lagged Flow:   {controls['lagged_flow']:>8.3f}")
    print(f"Log(TNA):      {controls['log_tna']:>8.3f}")
    print(f"TER:           {controls['ter']:>8.3f}")


# %%
try:
    # Extract coefficients from Model 2
    coefficients = extract_model2_coefficients(macro_results)
    
    # Save coefficients for ELTIF simulation
    save_coefficients(coefficients, filepath='goldstein_model2_coefficients.pkl')
    
    # Display summary table
    display_coefficient_summary(coefficients)
    
except Exception as e:
    print(f"\n❌ ERROR extracting/saving coefficients: {str(e)}")
    import traceback
    traceback.print_exc()

