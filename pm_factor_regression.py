#!/usr/bin/env python3
"""
pm_factor_regression.py
═══════════════════════════════════════════════════════════════════════════════
Robustness check: OLS regressions of unsmoothed Hamilton Lane PM returns on
public-market factors and macro-regime dummies.

Purpose
-------
The ELTIF simulation uses a regime-conditional block bootstrap to simulate PM
returns (Private_Markets_Simulation.py).  This script verifies that the regime
conditioning is empirically justified by showing:
  1. PM returns load significantly on public-market factors (STOXX 600, EUR bond)
  2. Macro regime dummies are jointly significant and correctly signed
  3. Regime-conditional means differ meaningfully across Goldilocks / Overheating
     / Downturn / Stagflation states

Three nested OLS models are estimated for each Hamilton Lane index
(composite, credit, equity):

  M1 (market):   r_PM = α + β_mkt·r_STOXX600 + ε
  M2 (two-fac):  r_PM = α + β_mkt·r_STOXX600 + β_bond·r_EUR_Bond + ε
  M3 (regime):   r_PM = α + β_mkt·r_STOXX600 + β_bond·r_EUR_Bond
                       + γ_OH·D_Overheating + γ_DT·D_Downturn
                       + γ_SF·D_Stagflation + ε
                 (Goldilocks = base category)

Standard errors: Newey-West HAC (4 lags) — appropriate for quarterly series.

Inputs (relative to script directory):
  Hamilton_Lane_unsmoothed_returns.csv — unsmoothed HL returns (USD)
  LSEG_merged/macro_indicators.csv     — Ilmanen macro regimes (new_macro_indicators.py)
  bloomberg_benchmarks.csv             — SXXR_Index, LBEATREU_Index (levels, quarterly)
  FRED EUR/USD (downloaded live)       — for USD → EUR conversion of HL returns

Outputs (written to Regression_Results/):
  pm_factor_regression_coefs.csv     — full coefficient table
  pm_factor_regression_regimes.csv   — regime-conditional statistics
  pm_factor_regression.tex           — LaTeX tables for thesis
"""

import io
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings('ignore', category=FutureWarning)

BASE_DIR    = Path(__file__).parent
OUT_DIR     = BASE_DIR / 'Regression_Results'
OUT_DIR.mkdir(exist_ok=True)

PM_INDEX_COLS = {
    'composite': 'Hamilton Lane Private Markets Index',
    'credit':    'Hamilton Lane Private Credit Index',
    'equity':    'Hamilton Lane Private Equity Index',
}
BASE_REGIME = 'Goldilocks'

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 72)
print("PM FACTOR REGRESSION  —  robustness check")
print("=" * 72)

# Unsmoothed PM returns (USD)
hl_raw = pd.read_csv(
    BASE_DIR / 'Hamilton_Lane_unsmoothed_returns.csv',
    index_col=0, parse_dates=True,
)
hl_raw.index = pd.DatetimeIndex(hl_raw.index)
print(f"\n✓ Hamilton Lane returns (USD): {hl_raw.shape}  "
      f"({hl_raw.index[0].date()} – {hl_raw.index[-1].date()})")

# Ilmanen macro regimes from macro_indicators.csv
_mi = pd.read_csv(BASE_DIR / 'LSEG_merged/macro_indicators.csv',
                  index_col=0, parse_dates=True)
_mi.index.name = 'Date'
_mi = _mi.reset_index().dropna(subset=['growth', 'inflation'])
_regime_map = {
    'GrowthUp/InflationDown':   'Goldilocks',
    'GrowthUp/InflationUp':     'Overheating',
    'GrowthDown/InflationDown': 'Downturn',
    'GrowthDown/InflationUp':   'Stagflation',
}
_mi['macro_regime'] = _mi['regime'].map(_regime_map)
_mi = _mi.dropna(subset=['macro_regime'])[['Date', 'macro_regime']].copy()
_mi['Date'] = pd.to_datetime(_mi['Date'])
_mi['_yq']  = _mi['Date'].dt.to_period('Q').astype(str)

hl_raw2 = hl_raw.copy().reset_index().rename(columns={'index': 'Date'})
hl_raw2['Date'] = pd.to_datetime(hl_raw2['Date'])
hl_raw2['_yq']  = hl_raw2['Date'].dt.to_period('Q').astype(str)

pm_macro = (
    hl_raw2.merge(_mi[['_yq', 'macro_regime']], on='_yq', how='left')
    .drop(columns=['_yq'])
    .set_index('Date')
)
print(f"✓ PM returns + macro:  {pm_macro.shape}  "
      f"({pm_macro.index[0].date()} – {pm_macro.index[-1].date()})")
print(f"  Regime distribution:\n{pm_macro['macro_regime'].value_counts().to_string()}")

# EUR benchmarks (levels → quarterly returns)
bench_raw = pd.read_csv(
    BASE_DIR / 'bloomberg_benchmarks.csv',
    index_col=0, parse_dates=True,
)
bench_raw.index = pd.DatetimeIndex(bench_raw.index)
stoxx_ret = bench_raw['SXXR_Index'].resample('QE').last().pct_change().rename('STOXX600_Return')
bond_ret  = bench_raw['LBEATREU_Index'].resample('QE').last().pct_change().rename('EUR_Bond_Return')
bench_q   = pd.concat([stoxx_ret, bond_ret], axis=1).dropna()
print(f"✓ EUR benchmarks:      {bench_q.shape}  "
      f"({bench_q.index[0].date()} – {bench_q.index[-1].date()})")

# EUR/USD from FRED — for USD → EUR conversion
print("  Downloading EUR/USD from FRED …", end=' ', flush=True)
_url  = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU'
_raw  = pd.read_csv(
    io.StringIO(urllib.request.urlopen(_url).read().decode()),
    na_values=['.'],
)
_raw.columns     = ['Date', 'EURUSD']
_raw['Date']     = pd.to_datetime(_raw['Date'])
_raw             = _raw.dropna().set_index('Date')
eurusd_q         = _raw['EURUSD'].resample('QE').last()
print(f"done  ({eurusd_q.index[0].date()} – {eurusd_q.index[-1].date()})")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  BUILD ANALYSIS DATASET
# ─────────────────────────────────────────────────────────────────────────────

pm_macro.index = pd.DatetimeIndex(pm_macro.index)

# USD → EUR for Hamilton Lane returns
hl_cols = list(PM_INDEX_COLS.values())
eurusd_aligned = eurusd_q.reindex(pm_macro.index, method='nearest')

pm_eur = pm_macro[hl_cols].copy()
for col in hl_cols:
    r_usd        = pm_macro[col]
    pm_eur[col]  = (1 + r_usd) * (eurusd_aligned.shift(1) / eurusd_aligned) - 1

# Merge
data = (
    pm_eur
    .join(pm_macro[['macro_regime']], how='left')
    .join(bench_q, how='inner')
    .dropna(subset=['macro_regime', 'STOXX600_Return'])
)

# Regime dummies (Goldilocks = base)
regimes       = [r for r in ['Overheating', 'Downturn', 'Stagflation']
                 if r in data['macro_regime'].unique()]
for r in regimes:
    data[f'D_{r}'] = (data['macro_regime'] == r).astype(float)

dummy_cols = [f'D_{r}' for r in regimes]

print(f"\n✓ Analysis dataset:    {data.shape}  "
      f"({data.index[0].date()} – {data.index[-1].date()})")
print(f"\nRegime distribution:")
print(data['macro_regime'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3.  OLS REGRESSIONS  (Newey-West HAC, 4 lags)
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    'M1_market':  ['STOXX600_Return'],
    'M2_twofac':  ['STOXX600_Return', 'EUR_Bond_Return'],
    'M3_regime':  ['STOXX600_Return', 'EUR_Bond_Return'] + dummy_cols,
}

HAC_LAGS = 4   # Newey-West; 4 lags ≈ 1 year for quarterly data

coef_records  = []
fit_records   = []

print("\n" + "=" * 72)
print("OLS RESULTS  (Newey-West HAC, 4 lags)")
print("=" * 72)

for idx_key, idx_col in PM_INDEX_COLS.items():
    y = data[idx_col].dropna()
    common_idx = y.index.intersection(data.index)
    y = y.loc[common_idx]

    print(f"\n{'─'*72}")
    print(f"  Index: {idx_col}")
    print(f"{'─'*72}")

    for model_name, x_cols in MODELS.items():
        X = sm.add_constant(data.loc[common_idx, x_cols])
        res = sm.OLS(y, X).fit(
            cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS}
        )

        dw = durbin_watson(res.resid)

        # Print summary
        print(f"\n  {model_name}   n={res.nobs:.0f}   "
              f"R²={res.rsquared:.3f}   adj-R²={res.rsquared_adj:.3f}   "
              f"DW={dw:.2f}")
        print(f"  {'Variable':<22}  {'Coef':>9}  {'t-stat':>8}  {'p-val':>7}")
        print(f"  {'─'*22}  {'─'*9}  {'─'*8}  {'─'*7}")
        for var in res.params.index:
            p = res.pvalues[var]
            stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
            print(f"  {var:<22}  {res.params[var]:>9.4f}  "
                  f"{res.tvalues[var]:>8.2f}  {p:>7.4f}{stars}")

        # Store
        for var in res.params.index:
            coef_records.append({
                'pm_index':   idx_key,
                'model':      model_name,
                'variable':   var,
                'coef':       res.params[var],
                'tstat':      res.tvalues[var],
                'pval':       res.pvalues[var],
                'n':          int(res.nobs),
                'r2':         res.rsquared,
                'r2_adj':     res.rsquared_adj,
                'dw':         dw,
            })

        fit_records.append({
            'pm_index': idx_key,
            'model':    model_name,
            'n':        int(res.nobs),
            'r2':       res.rsquared,
            'r2_adj':   res.rsquared_adj,
            'aic':      res.aic,
            'bic':      res.bic,
            'dw':       dw,
            'f_pval':   res.f_pvalue,
        })

# ─────────────────────────────────────────────────────────────────────────────
# 4.  REGIME-CONDITIONAL STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("REGIME-CONDITIONAL MEANS AND VOLATILITIES  (quarterly, EUR)")
print("=" * 72)

regime_rows = []
all_regimes = ['Goldilocks', 'Overheating', 'Downturn', 'Stagflation']
for idx_key, idx_col in PM_INDEX_COLS.items():
    for regime in all_regimes:
        sub = data.loc[data['macro_regime'] == regime, idx_col].dropna()
        if len(sub) == 0:
            continue
        regime_rows.append({
            'pm_index':  idx_key,
            'regime':    regime,
            'n':         len(sub),
            'mean_qtr':  sub.mean(),
            'mean_ann':  sub.mean() * 4,
            'vol_qtr':   sub.std(),
            'vol_ann':   sub.std() * np.sqrt(4),
            'sharpe_ann': (sub.mean() * 4) / (sub.std() * np.sqrt(4)) if sub.std() > 0 else np.nan,
        })

regime_df = pd.DataFrame(regime_rows)

print(f"\n{'Index':<12}  {'Regime':<14}  {'n':>4}  "
      f"{'Mean (ann)':>10}  {'Vol (ann)':>9}  {'Sharpe':>7}")
print('─' * 65)
for _, row in regime_df.iterrows():
    print(f"  {row['pm_index']:<10}  {row['regime']:<14}  {row['n']:>4}  "
          f"  {row['mean_ann']:>9.2%}  {row['vol_ann']:>9.2%}  "
          f"{row['sharpe_ann']:>7.2f}")

# Also show STOXX600 regime-conditional means for comparison
print(f"\n  {'STOXX600':<10}  {'Regime':<14}  {'n':>4}  {'Mean (ann)':>10}  {'Vol (ann)':>9}")
print('  ' + '─' * 55)
for regime in all_regimes:
    sub = data.loc[data['macro_regime'] == regime, 'STOXX600_Return'].dropna()
    if len(sub) == 0:
        continue
    print(f"  {'stoxx600':<10}  {regime:<14}  {len(sub):>4}  "
          f"  {sub.mean()*4:>9.2%}  {sub.std()*np.sqrt(4):>9.2%}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  SAVE CSV OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

coef_df   = pd.DataFrame(coef_records)
fit_df    = pd.DataFrame(fit_records)

coef_df.to_csv(OUT_DIR / 'pm_factor_regression_coefs.csv',   index=False)
regime_df.to_csv(OUT_DIR / 'pm_factor_regression_regimes.csv', index=False)
fit_df.to_csv(OUT_DIR / 'pm_factor_regression_fit.csv',       index=False)

print(f"\n✓ CSVs saved to: {OUT_DIR.relative_to(BASE_DIR)}/")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  LATEX OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def _lx(v, fmt='.3f', stars=False, pval=None):
    """Format a float for LaTeX; optionally append significance stars."""
    if pd.isna(v):
        return '—'
    s = f'{v:{fmt}}'
    if stars and pval is not None:
        if pval < 0.01:   s += r'^{***}'
        elif pval < 0.05: s += r'^{**}'
        elif pval < 0.10: s += r'^{*}'
    return s


def build_coef_table() -> str:
    """
    Panel table: rows = variables, columns = (index × model).
    Shows coefficient with superscript stars; t-stat in parentheses below.
    """
    var_order = ['const', 'STOXX600_Return', 'EUR_Bond_Return'] + dummy_cols
    var_labels = {
        'const':            r'Intercept',
        'STOXX600_Return':  r'$r_{\text{STOXX}600}$',
        'EUR_Bond_Return':  r'$r_{\text{EUR Bond}}$',
        'D_Overheating':    r'$D_{\text{Overheat}}$',
        'D_Downturn':       r'$D_{\text{Downturn}}$',
        'D_Stagflation':    r'$D_{\text{Stagflation}}$',
    }

    idx_order  = ['composite', 'credit', 'equity']
    mdl_order  = ['M1_market', 'M2_twofac', 'M3_regime']
    mdl_labels = {'M1_market': 'M1', 'M2_twofac': 'M2', 'M3_regime': 'M3'}
    idx_labels = {'composite': 'Composite', 'credit': 'Credit', 'equity': 'Equity'}

    n_cols = len(idx_order) * len(mdl_order)
    col_spec = 'l' + 'r' * n_cols

    # Header rows
    mc_parts = []
    for i, idx_key in enumerate(idx_order):
        last = (i == len(idx_order) - 1)
        mc_parts.append(
            r'\multicolumn{' + str(len(mdl_order)) + r'}{' +
            ('c' if last else 'c|') + r'}{' + idx_labels[idx_key] + r'}'
        )
    header1 = ' & ' + ' & '.join(mc_parts) + r' \\'
    header2 = ' & ' + ' & '.join(
        mdl_labels[m] for idx_key in idx_order for m in mdl_order
    ) + r' \\'

    # Index into coef_df
    _coef = coef_df.set_index(['pm_index', 'model', 'variable'])

    rows_tex = []
    for var in var_order:
        if var not in var_order:
            continue
        label = var_labels.get(var, var)
        # coefficient row
        cells_c = [label]
        # t-stat row
        cells_t = ['']
        for idx_key in idx_order:
            for mdl in mdl_order:
                try:
                    row = _coef.loc[(idx_key, mdl, var)]
                    c   = row['coef']
                    t   = row['tstat']
                    p   = row['pval']
                    cells_c.append(f'${_lx(c, ".3f", stars=True, pval=p)}$')
                    cells_t.append(f'$({_lx(t, ".2f")})$')
                except KeyError:
                    cells_c.append('')
                    cells_t.append('')
        rows_tex.append(' & '.join(cells_c) + r' \\')
        rows_tex.append(' & '.join(cells_t) + r' \\[2pt]')

    # Fit stats rows
    rows_tex.append(r'\midrule')
    for stat_key, stat_label, fmt in [
        ('r2',     r'$R^2$',         '.3f'),
        ('r2_adj', r'Adj.\ $R^2$',   '.3f'),
        ('n',      r'$n$',           '.0f'),
    ]:
        _fit = fit_df.set_index(['pm_index', 'model'])
        cells = [stat_label]
        for idx_key in idx_order:
            for mdl in mdl_order:
                try:
                    v = _fit.loc[(idx_key, mdl), stat_key]
                    cells.append(f'{v:{fmt}}')
                except KeyError:
                    cells.append('')
        rows_tex.append(' & '.join(cells) + r' \\')

    tex = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\footnotesize',
        r'\caption{OLS regressions of unsmoothed Hamilton Lane returns on public-market factors '
        r'and macro-regime dummies. '
        r'M1: STOXX\,600 only; M2: STOXX\,600 + EUR Bond; M3: M2 + regime dummies '
        r'(Goldilocks = base). '
        r'Standard errors: Newey-West HAC (4 lags). '
        r'Superscripts $^{***}$/$^{**}$/$^{*}$ denote significance at 1\%/5\%/10\%. '
        r't-statistics in parentheses.}',
        r'\label{tab:pm_factor_regression}',
        r'\begin{tabular}{' + col_spec + r'}',
        r'\toprule',
        header1,
        header2,
        r'\midrule',
    ] + rows_tex + [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(tex)


def build_regime_table() -> str:
    """Regime-conditional means, volatilities, and Sharpe ratios per PM index."""
    regime_order = ['Goldilocks', 'Overheating', 'Downturn', 'Stagflation']
    idx_order    = ['composite', 'credit', 'equity']
    idx_labels   = {'composite': 'Composite', 'credit': 'Credit', 'equity': 'Equity'}

    # 3 stats × 3 indices = 9 data columns
    mc_parts = []
    for i, idx_key in enumerate(idx_order):
        last = (i == len(idx_order) - 1)
        mc_parts.append(
            r'\multicolumn{3}{' + ('c' if last else 'c|') + r'}{' + idx_labels[idx_key] + r'}'
        )

    header1 = r'Regime & ' + ' & '.join(mc_parts) + r' \\'
    header2 = r' & ' + ' & '.join(
        [r'Mean (ann.) & Vol (ann.) & Sharpe'] * len(idx_order)
    ) + r' \\'

    col_spec = 'l' + ('rrr|' * (len(idx_order) - 1)) + 'rrr'

    _rdf = regime_df.set_index(['pm_index', 'regime'])
    rows_tex = []
    for regime in regime_order:
        cells = [regime]
        for idx_key in idx_order:
            try:
                r = _rdf.loc[(idx_key, regime)]
                cells.append(f"{r['mean_ann']:.2%}".replace('%', r'\%'))
                cells.append(f"{r['vol_ann']:.2%}".replace('%', r'\%'))
                cells.append(f"{r['sharpe_ann']:.2f}")
            except KeyError:
                cells += ['—', '—', '—']
        rows_tex.append(' & '.join(cells) + r' \\')

    tex = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\footnotesize',
        r'\caption{Regime-conditional annualised returns and volatilities of '
        r'unsmoothed Hamilton Lane indices (EUR, quarterly bootstrap history). '
        r'Sharpe ratio uses EUR 3-month OIS as risk-free rate. '
        r'This table motivates the regime-conditional bootstrap in the ELTIF simulation.}',
        r'\label{tab:pm_regime_stats}',
        r'\begin{tabular}{' + col_spec + r'}',
        r'\toprule',
        header1,
        header2,
        r'\midrule',
    ] + rows_tex + [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(tex)


tex_content = (
    build_coef_table()
    + '\n\n'
    + build_regime_table()
    + '\n'
)

tex_path = OUT_DIR / 'pm_factor_regression.tex'
tex_path.write_text(tex_content, encoding='utf-8')
print(f"✓ LaTeX tables saved: {tex_path.relative_to(BASE_DIR)}")

print("\n" + "=" * 72)
print("✅ PM FACTOR REGRESSION COMPLETE")
print("=" * 72)
