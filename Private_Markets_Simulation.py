"""
PRIVATE MARKETS + BENCHMARK SIMULATION WITH MACRO REGIMES
==========================================================

Bootstraps 10,000 paths × 80 quarters of correlated PM, cash, and EUR benchmark
returns, stratified by macro regime.  Output feeds Core_fund_simulation.py
via simulated_pm_cash_returns.csv.

Currency note
-------------
- Hamilton Lane indices are published in USD.  They are converted to EUR using
  the EUR/USD spot rate (FRED DEXUSEU) so the ELTIF simulation stays in EUR.
  Conversion: r_EUR = (1 + r_USD) × (EURUSD_{t-1} / EURUSD_t) − 1
- Alpha benchmarks are native EUR instruments:
    EURO STOXX 600  (SXXR Index, Bloomberg)  — EUR equity factor
    Bloomberg Euro Agg Bond (LBEATREU Index) — EUR bond factor
    EUR003          (already in Cash_quarterly_returns.csv) — EUR risk-free rate
- Benchmark data priority:
    1. bloomberg_benchmarks.csv  (export SXXR Index + LBEATREU Index from Bloomberg)
    2. yfinance fallback: ^STOXX50E + IEAG.AS (history from 2009 only)
  Pass --refresh to force re-download of the yfinance fallback.

Inputs (all relative to MT_Python/):
  Hamilton_Lane_unsmoothed_returns.csv
  Cash_quarterly_returns.csv
  LSEG_merged/macro_indicators.csv
  bloomberg_benchmarks.csv  (Bloomberg export: Date, SXXR_Index, LBEATREU_Index)

Output:
  simulated_pm_cash_returns.csv  — (path, t) indexed, consumed by Core_fund_simulation.py
"""

import sys
import io
import urllib.request
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

_REFRESH = '--refresh' in sys.argv
BASE_DIR = Path(__file__).parent
BLOOMBERG_CSV = BASE_DIR / 'bloomberg_benchmarks.csv'
EUR_BENCH_CACHE = BASE_DIR / 'eur_benchmarks.csv'

# =============================================================================
# HELPER: EUR benchmark loader  (Bloomberg CSV preferred; yfinance fallback)
# =============================================================================

def load_eur_benchmarks(start: str = '2000-01-01', end: str = '2026-01-01') -> pd.DataFrame:
    """
    Load EUR equity + bond benchmarks.  Priority:
      1. bloomberg_benchmarks.csv  (SXXR_Index, LBEATREU_Index as total-return levels)
      2. yfinance fallback: ^STOXX50E + IEAG.AS (history from ~2009 only)
    Returns quarterly DataFrame with columns: STOXX600_Return, EUR_Bond_Return.
    Cached in eur_benchmarks.csv.
    """
    cache = EUR_BENCH_CACHE
    if cache.exists() and not _REFRESH and not BLOOMBERG_CSV.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"✓ Loaded EUR benchmarks from cache: {cache.name} "
              f"({df.index[0].date()} – {df.index[-1].date()}, {len(df)} rows)")
        return df

    if BLOOMBERG_CSV.exists():
        print(f"  Loading Bloomberg benchmarks from: {BLOOMBERG_CSV.name}")
        raw = pd.read_csv(BLOOMBERG_CSV, index_col=0, parse_dates=True)
        raw.index = pd.DatetimeIndex(raw.index)
        stoxx_ret = raw['SXXR_Index'].resample('QE').last().pct_change()
        bond_ret  = raw['LBEATREU_Index'].resample('QE').last().pct_change()
        src = 'Bloomberg (SXXR Index, LBEATREU Index)'
    else:
        print("  ⚠  bloomberg_benchmarks.csv not found — using yfinance fallback")
        print("     (EURO STOXX 50 ^STOXX50E; Euro Agg Bond IEAG.AS — history from 2009 only)")
        stoxx = yf.download('^STOXX50E', start=start, end=end, progress=False, auto_adjust=True)
        stoxx_ret = stoxx[('Close', '^STOXX50E')].resample('QE').last().pct_change()
        bond  = yf.download('IEAG.AS',   start=start, end=end, progress=False, auto_adjust=True)
        bond_ret  = bond[('Close', 'IEAG.AS')].resample('QE').last().pct_change()
        src = 'yfinance fallback'

    df = pd.DataFrame({
        'STOXX600_Return': stoxx_ret,
        'EUR_Bond_Return': bond_ret,
    }).dropna()

    df.to_csv(cache)
    print(f"✓ EUR benchmarks ready: {len(df)} quarters "
          f"({df.index[0].date()} – {df.index[-1].date()})")
    print(f"  Source: {src}  |  Cached to: {cache.name}")
    return df


def load_eurusd_fx() -> pd.Series:
    """
    Download daily EUR/USD (USD per EUR) from FRED DEXUSEU, resample to quarterly last.
    Returns pd.Series named 'EURUSD' with Period/datetime index.
    """
    print("  Downloading EUR/USD from FRED (DEXUSEU) …")
    url  = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU'
    raw  = pd.read_csv(io.StringIO(urllib.request.urlopen(url).read().decode()),
                       na_values=['.'])
    # FRED uses the series ID as the value column; date is always the first column
    raw.columns = ['Date', 'EURUSD']
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw  = raw.dropna().set_index('Date')
    eurusd_q = raw['EURUSD'].resample('QE').last()
    print(f"  EUR/USD loaded: {eurusd_q.index[0].date()} – {eurusd_q.index[-1].date()}, "
          f"{len(eurusd_q)} quarters")
    return eurusd_q


# =============================================================================
# STEP 1: LOAD ALL DATA
# =============================================================================

print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

pm_returns_usd = pd.read_csv('Hamilton_Lane_unsmoothed_returns.csv',
                              index_col=0, parse_dates=True)
print(f"\n✓ Loaded PM returns (USD): {pm_returns_usd.shape}")

cash_returns = pd.read_csv('Cash_quarterly_returns.csv', index_col=0, parse_dates=True)
print(f"✓ Loaded cash returns (EUR003): {cash_returns.shape}")

_macro_raw = pd.read_csv('LSEG_merged/macro_indicators.csv', index_col=0, parse_dates=True)
_macro_raw.index.name = 'Date'
_macro_raw = _macro_raw.reset_index()
_macro_raw = _macro_raw.dropna(subset=['growth', 'inflation'])
_regime_map = {
    'GrowthUp/InflationDown': 'Goldilocks',
    'GrowthUp/InflationUp':   'Overheating',
    'GrowthDown/InflationDown': 'Downturn',
    'GrowthDown/InflationUp':  'Stagflation',
}
_macro_raw['macro_regime'] = _macro_raw['regime'].map(_regime_map)
macro_regimes = _macro_raw.dropna(subset=['macro_regime'])[['Date', 'macro_regime', 'growth', 'inflation']].copy()
macro_regimes['Date'] = pd.to_datetime(macro_regimes['Date'])
print(f"✓ Loaded macro regimes (Ilmanen): {macro_regimes.shape}")
print(macro_regimes['macro_regime'].value_counts())

print("\n  EUR benchmarks:")
eur_bench = load_eur_benchmarks()

print("\n  EUR/USD FX rate:")
eurusd_q = load_eurusd_fx()

# =============================================================================
# STEP 2: CONVERT HAMILTON LANE USD → EUR
# =============================================================================

print("\n" + "=" * 80)
print("STEP 2: CONVERTING HAMILTON LANE RETURNS USD → EUR")
print("=" * 80)

# Align EURUSD to the PM quarterly index
pm_idx        = pd.DatetimeIndex(pm_returns_usd.index)
eurusd_aligned = eurusd_q.reindex(pm_idx, method='nearest')

pm_returns = pm_returns_usd.copy()
hl_cols    = list(pm_returns.columns)

for col in hl_cols:
    r_usd = pm_returns_usd[col]
    # r_EUR = (1 + r_USD) × (EURUSD_{t-1} / EURUSD_t) − 1
    # EURUSD = USD per EUR; rising EURUSD = EUR strengthens → EUR investor loses on USD assets
    r_eur = (1 + r_usd) * (eurusd_aligned.shift(1) / eurusd_aligned) - 1
    pm_returns[col] = r_eur

pm_returns = pm_returns.dropna()
print(f"✓ Converted {len(hl_cols)} Hamilton Lane series to EUR")
print(f"  Annualised mean return (USD): "
      f"{pm_returns_usd[hl_cols].mean().mean() * 4:.2%}")
print(f"  Annualised mean return (EUR): "
      f"{pm_returns[hl_cols].mean().mean() * 4:.2%}")

# =============================================================================
# STEP 3: BUILD EUR BENCHMARK COLUMNS
# =============================================================================

print("\n" + "=" * 80)
print("STEP 3: BUILDING EUR BENCHMARK COLUMNS")
print("=" * 80)

# Use EUR003 as risk-free rate (already in cash_returns)
# Align to quarterly end-of-quarter date index
cash_q = cash_returns.copy()
cash_q.index = pd.DatetimeIndex(cash_q.index)
eur_rf_series = cash_q['EUR003_Index']

# Align EUR benchmarks + RF to a common quarterly index
bench_idx = eur_bench.index   # DatetimeIndex, QE freq

eur_rf_aligned = eur_rf_series.reindex(bench_idx, method='nearest')
eur_bench['EUR_RF']            = eur_rf_aligned.values
eur_bench['Excess_STOXX600']   = eur_bench['STOXX600_Return'] - eur_bench['EUR_RF']
eur_bench['Excess_EUR_Bond']   = eur_bench['EUR_Bond_Return'] - eur_bench['EUR_RF']

print(f"✓ EUR benchmark columns built:")
print(f"  STOXX600_Return: mean = {eur_bench['STOXX600_Return'].mean() * 4:.2%} ann.")
print(f"  EUR_Bond_Return: mean = {eur_bench['EUR_Bond_Return'].mean() * 4:.2%} ann.")
print(f"  EUR_RF:          mean = {eur_bench['EUR_RF'].mean() * 4:.2%} ann.")

# =============================================================================
# STEP 4: MERGE EVERYTHING TOGETHER
# =============================================================================

print("\n" + "=" * 80)
print("STEP 4: MERGING ALL DATA WITH MACRO REGIMES")
print("=" * 80)

pm_with_date    = pm_returns.reset_index().rename(columns={pm_returns.index.name or 'index': 'Date'})
cash_with_date  = cash_returns.reset_index().rename(columns={cash_returns.index.name or 'index': 'Date'})
bench_with_date = eur_bench.reset_index().rename(columns={eur_bench.index.name or 'index': 'Date'})

for df in [pm_with_date, cash_with_date, bench_with_date, macro_regimes]:
    df['Date'] = pd.to_datetime(df['Date'])

pm_with_date['Quarter']    = pm_with_date['Date'].dt.to_period('Q')
cash_with_date['Quarter']  = cash_with_date['Date'].dt.to_period('Q')
bench_with_date['Quarter'] = bench_with_date['Date'].dt.to_period('Q')
macro_regimes['Quarter']   = macro_regimes['Date'].dt.to_period('Q')

macro_quarterly = macro_regimes.sort_values('Date').groupby('Quarter').last().reset_index()

eur_bench_cols = ['STOXX600_Return', 'EUR_Bond_Return', 'EUR_RF',
                  'Excess_STOXX600', 'Excess_EUR_Bond']

combined_data = (
    pm_with_date
    .merge(cash_with_date[['Quarter', 'EUR003_Index']],               on='Quarter', how='left')
    .merge(bench_with_date[['Quarter'] + eur_bench_cols],             on='Quarter', how='left')
    .merge(macro_quarterly[['Quarter', 'macro_regime',
                             'growth', 'inflation']],                  on='Quarter', how='left')
    .drop(columns=['Quarter'])
    .dropna(subset=['macro_regime', 'Excess_STOXX600'])   # require benchmarks present
)

print(f"\n✓ Combined data shape: {combined_data.shape}")
print(f"  Date range: {combined_data['Date'].min().date()} to {combined_data['Date'].max().date()}")
print(f"  (Bootstrap history: Bloomberg SXXR/LBEATREU from 2000; yfinance fallback from ~2009)")
print(f"\nRegime distribution:")
print(combined_data['macro_regime'].value_counts())

# =============================================================================
# STEP 5: REGIME TRANSITION MATRIX
# =============================================================================

print("\n" + "=" * 80)
print("STEP 5: CALCULATING REGIME TRANSITION MATRIX")
print("=" * 80)

combined_sorted = combined_data.sort_values('Date').copy()
current_regime  = combined_sorted['macro_regime']
next_regime     = current_regime.shift(-1)

transition_matrix = pd.crosstab(current_regime, next_regime, normalize='index')

print("\n✓ Transition Matrix:")
print(transition_matrix.round(3))
print("\n" + "-" * 80)
print("REGIME PERSISTENCE")
print("-" * 80)
for regime in transition_matrix.index:
    stay_prob = transition_matrix.loc[regime, regime] if regime in transition_matrix.columns else 0
    avg_dur   = 1 / (1 - stay_prob) if stay_prob < 1 else float('inf')
    print(f"{regime:15s}: Stay prob = {stay_prob:.1%}, Avg duration = {avg_dur:.1f} quarters")

# =============================================================================
# STEP 6: SIMULATION FUNCTIONS
# =============================================================================

def simulate_regime_path(transition_matrix, n_quarters, start_regime='Goldilocks', seed=0):
    """Simulate a sequence of macro regimes using the estimated transition matrix."""
    rng     = np.random.default_rng(seed)
    regimes = transition_matrix.index.tolist()
    if start_regime not in regimes:
        start_regime = regimes[0]

    path           = [start_regime]
    current_regime = start_regime

    for _ in range(n_quarters - 1):
        if current_regime in transition_matrix.index:
            probs = transition_matrix.loc[current_regime].values
            if not np.isclose(probs.sum(), 1.0):
                probs = probs / probs.sum()
            next_r = rng.choice(transition_matrix.columns, p=probs)
        else:
            next_r = rng.choice(regimes)
        path.append(next_r)
        current_regime = next_r

    return np.array(path)


def bootstrap_all_returns(combined_data, regime_path, seed=0):
    """
    Bootstrap PM + Cash + EUR Benchmark returns conditional on regime.

    Samples entire rows from combined_data so that cross-asset correlations
    (EUR PM ↔ EUR cash ↔ EUR benchmarks) are preserved within each regime.
    """
    rng          = np.random.default_rng(seed)
    exclude_cols = ['Date', 'macro_regime', 'growth', 'inflation']
    return_cols  = [c for c in combined_data.columns if c not in exclude_cols]

    rows = []
    for regime in regime_path:
        regime_data = combined_data[combined_data['macro_regime'] == regime][return_cols]
        if len(regime_data) == 0:
            print(f"Warning: No data for {regime}, using overall mean")
            rows.append(combined_data[return_cols].mean())
        else:
            rows.append(regime_data.iloc[rng.integers(0, len(regime_data))])

    sim_df = pd.DataFrame(rows).reset_index(drop=True)
    sim_df['macro_regime'] = regime_path
    sim_df.index.name      = 't'
    return sim_df


def simulate_multipaths(
    combined_data,
    transition_matrix,
    n_paths=1000,
    n_quarters=80,
    start_regime='Downturn',
    seed=0,
    verbose=True,
):
    """Generate multiple paths of correlated EUR PM + Cash + Benchmark returns."""
    if verbose:
        print("=" * 80)
        print(f"SIMULATING {n_paths} PATHS × {n_quarters} QUARTERS")
        print("=" * 80)
        print(f"Starting regime: {start_regime}")

    all_paths = []
    for path_id in range(n_paths):
        if verbose and (path_id + 1) % 200 == 0:
            print(f"  Path {path_id + 1}/{n_paths} ({100 * (path_id + 1) / n_paths:.0f}%)")

        regime_path  = simulate_regime_path(
            transition_matrix, n_quarters=n_quarters,
            start_regime=start_regime, seed=seed + path_id,
        )
        path_returns = bootstrap_all_returns(
            combined_data, regime_path, seed=seed + path_id + n_paths,
        )
        path_returns['path'] = path_id
        path_returns         = path_returns.set_index(['path', path_returns.index])
        all_paths.append(path_returns)

    boot_df       = pd.concat(all_paths)
    boot_df.index = boot_df.index.set_names(['path', 't'])

    if verbose:
        print(f"\n✓ Simulation complete — shape: {boot_df.shape}")
        print("\n  Regime distribution in simulated data:")
        for regime, count in boot_df['macro_regime'].value_counts().items():
            print(f"    {regime:15s}: {count:6d} ({100 * count / len(boot_df):5.1f}%)")

    return boot_df


# =============================================================================
# STEP 7: RUN SIMULATION
# =============================================================================

print("\n" + "=" * 80)
print("STEP 7: RUNNING SIMULATION")
print("=" * 80)

sim_df = simulate_multipaths(
    combined_data=combined_data,
    transition_matrix=transition_matrix,
    n_paths=10000,
    n_quarters=80,
    start_regime='Goldilocks',
    seed=42,
)

# =============================================================================
# STEP 8: SAVE FOR ELTIF SIMULATION
# =============================================================================

print("\n" + "=" * 80)
print("STEP 8: SAVING SIMULATED DATA")
print("=" * 80)

pm_cols = [c for c in combined_data.columns if 'Hamilton' in c or 'Private' in c]
sim_df['avg_pm_return'] = sim_df[pm_cols].mean(axis=1)

sim_df_export = sim_df.reset_index()
sim_df_export.to_csv('simulated_pm_cash_returns.csv', index=False)

print(f"\n✓ Saved to: simulated_pm_cash_returns.csv")
print(f"\nColumns exported:")
for col in sim_df_export.columns:
    print(f"  - {col}")

print("\n" + "=" * 80)
print("✅ SIMULATION COMPLETE")
print("=" * 80)
