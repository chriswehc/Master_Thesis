#!/usr/bin/env python3
"""
Standalone simulation runner — called by the Streamlit dashboard.

Usage:
    python run_simulation_pipeline.py \
        --cash_weight 0.15 \
        --credit_cap  0.05 \
        --credit_spr  0.03 \
        --n_paths     1000 \
        --initial_tna 100000000

Intercept and log_tna handling
------------------------------
The regression constant (β₀) is evaluated at all predictors = 0, which is an
out-of-sample extrapolation that absorbs secular industry trends (bond AUM tripled
2008-2014).  Following Ben-David et al. (2022) and Zhu (2018) we replace β₀ with
the empirical long-run mean flow rate from the training data ('flow_rate_mean').

However, in OLS the constant is also calibrated to offset the large level effect of
log_tna_coef × mean_log_tna (≈ −8 %/qtr at typical fund sizes).  Replacing β₀ with
flow_rate_mean without restoring that offset creates a structural drag that causes
the fund to shrink rapidly.  We therefore add −log_tna_coef × mean_log_tna back into
the simulation intercept (stored as 'mean_log_tna' in the pickle).  After this
adjustment the log_tna term captures only the size deviation of the fund from the
average historical fund — the mean-reversion signal — while the baseline flow for
an average-sized fund equals the empirical long-run mean.
"""

import argparse
import copy
import os
import sys
from pathlib import Path

# ── Parse arguments ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Run ELTIF Monte Carlo simulation')
parser.add_argument('--cash_weight',  type=float, default=0.15,
                    help='Cash buffer as fraction of TNA (e.g. 0.15 = 15%%)')
parser.add_argument('--credit_cap',   type=float, default=0.05,
                    help='Credit line capacity as fraction of TNA (e.g. 0.05 = 5%%)')
parser.add_argument('--credit_spr',   type=float, default=0.03,
                    help='Credit spread over base rate (e.g. 0.03 = 3%%)')
parser.add_argument('--n_paths',      type=int,   default=1000,
                    help='Number of Monte Carlo paths to simulate')
parser.add_argument('--initial_tna',  type=float, default=100_000_000,
                    help='Initial fund TNA in EUR')
parser.add_argument('--horizon_years', type=int,   default=10,
                    help='Simulation horizon in years (e.g. 10 = 40 quarters)')
parser.add_argument('--haircut_rate',  type=float, default=0.05,
                    help='Haircut on forced IL liquidations (e.g. 0.05 = 5%%)')
parser.add_argument('--buffer_gate_fraction', type=float, default=0.5,
                    help='ELTIF 2.0: max outflow as fraction of current cash buffer (default 0.5 = 50%%)')
parser.add_argument('--gate_mode', type=str, default='strict',
                    choices=['strict', 'economic'],
                    help='strict: gate caps total outflow (ELTIF 2.0) | economic: fixed TNA-based gate')
parser.add_argument('--redemption_gate_pct', type=float, default=0.20,
                    help='Economic mode: fixed TNA-based redemption cap (default 0.20 = 20%%)')
parser.add_argument('--pm_index', type=str, default='composite',
                    choices=['composite', 'credit', 'equity', 'equal_weight'],
                    help='Hamilton Lane index used as PM return '
                         '(composite|credit|equity|equal_weight, default composite)')
parser.add_argument('--stats', action='store_true',
                    help='Print summary statistics after simulation completes.')
parser.add_argument('--plot', action='store_true',
                    help='Save fan chart and benchmark comparison plots to PNG files.')
parser.add_argument('--coef_pkl', type=str, default='',
                    help='Path to a specific Goldstein coefficients .pkl file.')
parser.add_argument('--run_tag', type=str, default='',
                    help='Regression run tag used to auto-locate the coefficients pkl '
                         '(e.g. _bond_hy).  Ignored if --coef_pkl is set.')
args = parser.parse_args()

print("=" * 70)
print("ELTIF MONTE CARLO SIMULATION")
print("=" * 70)
print(f"  Cash buffer:      {args.cash_weight:.1%}")
print(f"  Credit capacity:  {args.credit_cap:.1%}")
print(f"  Credit spread:    {args.credit_spr:.2%}")
print(f"  Haircut rate:     {args.haircut_rate:.1%}")
print(f"  Buffer gate:      {args.buffer_gate_fraction:.0%} of cash buffer/qtr")
print(f"  PM index:         {args.pm_index}")
print(f"  Paths:            {args.n_paths:,}")
print(f"  Initial TNA:      €{args.initial_tna:,.0f}")
print(f"  Horizon:          {args.horizon_years} years ({args.horizon_years * 4} quarters)")
print("=" * 70)

# ── Set working directory so relative paths in Core_fund_simulation work ─────
BASE_DIR = Path(__file__).parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

# ── Import Core_fund_simulation ───────────────────────────────────────────────
print("\nLoading simulation data and coefficients...")
import importlib
import re as _re
import Core_fund_simulation as cs

# ── Load the right coefficients pkl ──────────────────────────────────────────
_coef_pkl = args.coef_pkl
if not _coef_pkl and args.run_tag:
    _rt      = args.run_tag if args.run_tag.startswith('_') else f'_{args.run_tag}'
    _reg_tag = _re.sub(r'_h\d+_s\d+$', '', _rt)
    _runs_pkl    = BASE_DIR / f'LSEG_merged/runs/goldstein_model2_coefficients{_reg_tag}.pkl'
    _default_pkl = BASE_DIR / f'LSEG_merged/goldstein_model2_coefficients{_reg_tag}.pkl'
    if _runs_pkl.exists():
        _coef_pkl = str(_runs_pkl)
    elif _default_pkl.exists():
        _coef_pkl = str(_default_pkl)

if _coef_pkl:
    if Path(_coef_pkl).exists():
        cs.coefficients = cs.load_goldstein_coefficients(_coef_pkl)
        print(f"  Coefficients loaded: {Path(_coef_pkl).name}")
    else:
        print(f"  ⚠ pkl not found ({_coef_pkl}) — falling back to default")
        importlib.reload(cs)
else:
    importlib.reload(cs)   # reload so newly saved pickle is picked up

if cs.coefficients is None:
    raise RuntimeError(
        "No Goldstein coefficients loaded. Pass --coef_pkl or --run_tag, "
        "or run the regression first."
    )

print(f"  PM returns loaded:   {len(cs.pm_cash_sim_returns):,} rows, "
      f"{cs.pm_cash_sim_returns['path'].nunique():,} paths available")

# ── Replace regression constant with long-run empirical mean ─────────────────
# The raw β₀ absorbs secular industry-wide flow trends and is not meaningful
# for forward simulation.  The regression saves the training-data mean Flow_Rate
# in the pickle; we swap β₀ for that value here.
coefficients = copy.deepcopy(cs.coefficients)
ctrl = coefficients['macro_regime']['controls']
raw_intercept  = ctrl['intercept']
flow_rate_mean = ctrl.get('flow_rate_mean', None)

if flow_rate_mean is not None:
    ctrl['intercept'] = flow_rate_mean
    print(f"\n  Raw regression β₀:       {raw_intercept:+.4f} ({raw_intercept:.2%}/qtr)")
    print(f"  Long-run Flow_Rate mean: {flow_rate_mean:+.4f} ({flow_rate_mean:.2%}/qtr)")
    print(f"  ➜ Intercept replaced with long-run mean (Ben-David et al. 2022)")
else:
    print(f"\n  ⚠ flow_rate_mean not found in pickle — using raw β₀={raw_intercept:.4f}")
    print(f"    Re-run the regression to store the long-run mean.")

# ── Centre log(TNA) around the training-data mean ────────────────────────────
# In OLS the constant β₀ is calibrated so that E[ε]=0 at the training means of
# all regressors.  In particular it absorbs  log_tna_coef × mean_log_tna.  After
# replacing β₀ with flow_rate_mean that offset is lost, leaving an uncompensated
# level drag of  log_tna_coef × log(TNA) ≈ −8 %/qtr at TNA = €100 M.
#
# Fix: add  −log_tna_coef × mean_log_tna  back into the simulation intercept.
# Economically this means the log_tna term now captures only the SIZE DEVIATION
# of the fund from the average historical fund — exactly the mean-reversion signal
# we want — while the level effect at the mean is absorbed into the intercept.
mean_log_tna = ctrl.get('mean_log_tna', None)
if mean_log_tna is not None:
    import math
    log_tna_coef   = ctrl.get('log_tna', 0.0)
    centering_adj  = -log_tna_coef * mean_log_tna   # positive when coef < 0
    ctrl['intercept'] += centering_adj
    print(f"\n  log_tna coefficient:       {log_tna_coef:+.6f}")
    print(f"  Mean log(TNA) in training: {mean_log_tna:.4f}  "
          f"(TNA ≈ €{math.exp(mean_log_tna)/1e6:.0f}M)")
    print(f"  Centering adjustment:      {centering_adj:+.4f} ({centering_adj:.2%}/qtr)")
    print(f"  Final simulation intercept:{ctrl['intercept']:+.4f} ({ctrl['intercept']:.2%}/qtr)")
    print(f"  ➜ log_tna now measures size deviation from training-mean fund")
else:
    print(f"\n  ⚠ mean_log_tna not in pickle — log_tna level drag not corrected")
    print(f"    Re-run the regression so the training mean is stored.")

# ── Filter to requested number of paths ──────────────────────────────────────
all_paths    = cs.pm_cash_sim_returns['path'].unique()
n_use        = min(args.n_paths, len(all_paths))
if n_use < args.n_paths:
    print(f"  ⚠ Only {n_use:,} paths available (requested {args.n_paths:,})")
paths_to_use = all_paths[:n_use]
pm_filtered  = cs.pm_cash_sim_returns[
    cs.pm_cash_sim_returns['path'].isin(paths_to_use)
].copy()

# ── Truncate each path to the requested horizon ───────────────────────────
n_quarters_max = args.horizon_years * 4
pm_filtered = (
    pm_filtered
    .groupby('path')
    .head(n_quarters_max)
    .reset_index(drop=True)
)

print(f"  Using {n_use:,} paths  ({len(pm_filtered):,} rows)"
      f"  |  {n_quarters_max} quarters ({args.horizon_years} years) per path")

# ── Run simulation ────────────────────────────────────────────────────────────
print("\nStarting Monte Carlo simulation...")
results = cs.simulate_eltif_multipaths(
    pm_filtered,
    coefficients,           # ← deepcopy with β₀ replaced by long-run mean
    cash_returns=None,   # per-path EUR003_Index extracted inside simulate_eltif_multipaths
    initial_tna=args.initial_tna,
    cash_weight=args.cash_weight,
    credit_capacity=args.credit_cap,
    credit_spread=args.credit_spr,
    haircut_rate=args.haircut_rate,
    buffer_gate_fraction=args.buffer_gate_fraction,
    gate_mode=args.gate_mode,
    redemption_gate_pct=args.redemption_gate_pct,
    pm_index=args.pm_index,
    verbose=True
)

# ── Save results ──────────────────────────────────────────────────────────────
out_path = BASE_DIR / 'eltif_results_revolving_credit.csv'
results.to_csv(out_path)

print("\n" + "=" * 70)
print(f"✅ SIMULATION COMPLETE")
print(f"   Paths simulated: {results.index.get_level_values('path').nunique():,}")
print(f"   Results saved → {out_path.name}")
print("=" * 70)

# ── Output directory + run tag ────────────────────────────────────────────────
_single_runs_dir = BASE_DIR / 'single_runs'
_single_runs_dir.mkdir(exist_ok=True)

_run_tag = (
    f"cw{int(round(args.cash_weight * 100))}"
    f"_cc{int(round(args.credit_cap * 100))}"
    f"_{args.gate_mode}"
    f"_n{n_use}"
)

# ── Summary statistics ────────────────────────────────────────────────────────
if args.stats:
    metrics = cs.calculate_optimization_metrics(
        results,
        initial_tna=args.initial_tna,
        horizon_years=args.horizon_years,
    )
    import numpy as np
    stoxx600_log_ann   = 4.0 * float(np.log1p(pm_filtered['STOXX600_Return']).mean())
    illiq_premium_bps  = (metrics['expected_log_growth_ann'] - stoxx600_log_ann) * 10_000
    haircut_cost_pct   = (metrics['expected_shortfall_fl_cond'] * args.haircut_rate
                          / args.initial_tna * 100)

    # Credit stats
    avg_credit_outstanding = results['Credit_Outstanding'].mean()
    total_credit_interest  = results['Credit_Interest'].sum() / n_use   # avg across paths
    credit_interest_pct    = total_credit_interest / args.initial_tna * 100

    # Regime distribution across all path-quarters
    regime_counts = results['Regime'].value_counts()
    regime_pct    = (regime_counts / len(results) * 100).round(1)
    _regime_order = ['Goldilocks', 'Overheating', 'Downturn', 'Stagflation']
    _regime_lines = []
    for r in _regime_order:
        if r in regime_pct:
            _regime_lines.append(f"    {r:<16} {regime_pct[r]:>5.1f}%")

    _stats_lines = [
        "=" * 70,
        "SIMULATION STATISTICS",
        "=" * 70,
        f"  Cash weight:                {args.cash_weight:.0%}",
        f"  Credit capacity:            {args.credit_cap:.0%}",
        f"  Gate mode:                  {args.gate_mode}",
        f"  Haircut rate:               {args.haircut_rate:.0%}",
        f"  Credit spread:              {args.credit_spr:.2%}",
        f"  PM index:                   {args.pm_index}",
        f"  Paths simulated:            {results.index.get_level_values('path').nunique():,}",
        f"  Horizon:                    {args.horizon_years} years",
        "─" * 70,
        "  RETURNS",
        f"  E[log growth] (ann.):       {metrics['expected_log_growth_ann']:+.2%}",
        f"  Std[log growth]:            {metrics['log_growth_std']:.2%}",
        f"  Illiq. premium vs STOXX600: {illiq_premium_bps:+.0f} bps/yr",
        f"  Median final TNA:           €{metrics['median_final_tna']:,.0f}",
        f"  P10 final TNA:              €{metrics['p10_final_tna']:,.0f}",
        "─" * 70,
        "  LIQUIDITY STRESS",
        f"  P(≥1 forced liquidation):   {metrics['p_fl_any_path']:.2%}",
        f"  P(≥1 credit draw):          {metrics['p_credit_any_path']:.2%}",
        f"  E[shortfall | FL]:          €{metrics['expected_shortfall_fl_cond']:,.0f}",
        f"  Haircut cost (% of TNA):    {haircut_cost_pct:.2f}%",
        f"  Avg credit outstanding/qtr: €{avg_credit_outstanding:,.0f}",
        f"  Avg total credit interest:  €{total_credit_interest:,.0f}  ({credit_interest_pct:.2f}% of TNA)",
        "─" * 70,
        "  MACRO REGIME DISTRIBUTION",
    ] + _regime_lines + ["=" * 70]
    _stats_text = "\n".join(_stats_lines)
    print("\n" + _stats_text)

    _stats_path = _single_runs_dir / f'stats_{_run_tag}.txt'
    _stats_path.write_text(_stats_text + "\n", encoding='utf-8')
    print(f"\n✓ Statistics saved → single_runs/{_stats_path.name}")

# ── Plots ─────────────────────────────────────────────────────────────────────
if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # Map pm_index name → column in pm_filtered
    _hl_cols = [c for c in pm_filtered.columns if 'Hamilton Lane' in c]
    _pm_col_map = {
        'composite':   next((c for c in _hl_cols if 'Private Markets' in c), _hl_cols[0]),
        'credit':      next((c for c in _hl_cols if 'Private Credit'  in c), _hl_cols[0]),
        'equity':      next((c for c in _hl_cols if 'Private Equity'  in c), _hl_cols[0]),
        'equal_weight': None,
    }
    pm_col = _pm_col_map.get(args.pm_index)

    if args.pm_index == 'equal_weight':
        pm_label = 'HL Equal Weight'
    else:
        pm_label = pm_col.replace('Hamilton Lane ', 'HL ')

    # Add within-path quarter index for reliable pivoting
    _pf = pm_filtered.copy()
    _pf['_t'] = _pf.groupby('path').cumcount()

    if args.pm_index == 'equal_weight':
        _pf['_pm_ret'] = _pf[_hl_cols].mean(axis=1)
    else:
        _pf['_pm_ret'] = _pf[pm_col]

    # Cumulative growth matrices: shape (n_quarters × n_paths)
    pm_cum = (
        _pf.pivot(index='_t', columns='path', values='_pm_ret')
        .add(1).cumprod()
    )
    stoxx_cum = (
        _pf.pivot(index='_t', columns='path', values='STOXX600_Return')
        .add(1).cumprod()
    )

    # ELTIF TNA per path (quarters × paths), normalised to start = 1
    tna_wide = (
        results['TNA']
        .unstack(level='path')
        .divide(args.initial_tna)
    )
    quarters = range(len(tna_wide))

    # ── Figure 1: Fan chart ───────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    for col in tna_wide.columns:
        ax1.plot(quarters, tna_wide[col] * args.initial_tna / 1e6,
                 color='lightgray', linewidth=0.4, alpha=0.5)

    tna_median = tna_wide.median(axis=1) * args.initial_tna / 1e6
    tna_p10    = tna_wide.quantile(0.10, axis=1) * args.initial_tna / 1e6
    tna_p90    = tna_wide.quantile(0.90, axis=1) * args.initial_tna / 1e6

    ax1.fill_between(quarters, tna_p10, tna_p90, alpha=0.25, color='steelblue', label='P10–P90')
    ax1.plot(quarters, tna_median, color='steelblue', linewidth=2.5, label='Median TNA')
    ax1.axhline(args.initial_tna / 1e6, color='black', linewidth=1,
                linestyle='--', alpha=0.5, label='Initial TNA')

    ax1.set_title(
        f'ELTIF TNA — All Paths Fan Chart\n'
        f'cash={args.cash_weight:.0%}  credit={args.credit_cap:.0%}  '
        f'gate={args.gate_mode}  n={tna_wide.shape[1]:,}',
        fontsize=11,
    )
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('TNA (€M)')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'€{x:.0f}M'))
    ax1.legend(frameon=False)
    plt.tight_layout()
    fan_path = _single_runs_dir / f'fan_chart_{_run_tag}.png'
    fig1.savefig(fan_path, dpi=150)
    plt.close(fig1)
    print(f"\n✓ Fan chart saved → single_runs/{fan_path.name}")

    # ── Figure 2: Median-path benchmark comparison ────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    # ELTIF median (already in €M above)
    ax2.plot(quarters, tna_median, linewidth=2.5, label='ELTIF (median path)', color='steelblue')

    # PM index median growth
    pm_median = pm_cum.median(axis=1) * args.initial_tna / 1e6
    ax2.plot(range(len(pm_median)), pm_median, linewidth=2,
             label=f'{pm_label} (median path)', color='darkorange', linestyle='--')

    # STOXX 600 median growth
    stoxx_median = stoxx_cum.median(axis=1) * args.initial_tna / 1e6
    ax2.plot(range(len(stoxx_median)), stoxx_median, linewidth=2,
             label='STOXX 600 (median path)', color='forestgreen', linestyle=':')

    ax2.axhline(args.initial_tna / 1e6, color='black', linewidth=1,
                linestyle='--', alpha=0.5, label='Initial TNA')

    ax2.set_title(
        f'ELTIF vs Benchmarks — Median Path\n'
        f'cash={args.cash_weight:.0%}  credit={args.credit_cap:.0%}  '
        f'gate={args.gate_mode}  n={tna_wide.shape[1]:,}',
        fontsize=11,
    )
    ax2.set_xlabel('Quarter')
    ax2.set_ylabel('Value (€M)')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'€{x:.0f}M'))
    ax2.legend(frameon=False)
    plt.tight_layout()
    bench_path = _single_runs_dir / f'benchmark_comparison_{_run_tag}.png'
    fig2.savefig(bench_path, dpi=150)
    plt.close(fig2)
    print(f"✓ Benchmark comparison saved → single_runs/{bench_path.name}")

    # ── Figure 3: Final TNA distribution ─────────────────────────────────────
    import numpy as _np
    final_tna = results.groupby(level='path')['TNA'].last() / 1e6

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.hist(final_tna.clip(upper=2000), bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
    ax3.set_xlim(right=2000)
    ax3.axvline(final_tna.median(), color='red',   linewidth=2,
                linestyle='--', label=f'Median  €{final_tna.median():.1f}M')
    ax3.axvline(final_tna.quantile(0.10), color='darkorange', linewidth=2,
                linestyle=':', label=f'P10  €{final_tna.quantile(0.10):.1f}M')
    ax3.axvline(args.initial_tna / 1e6, color='black', linewidth=1.5,
                linestyle='--', alpha=0.6, label=f'Initial  €{args.initial_tna/1e6:.0f}M')
    ax3.set_title(
        f'Final TNA Distribution (quarter {args.horizon_years * 4})\n'
        f'cash={args.cash_weight:.0%}  credit={args.credit_cap:.0%}  gate={args.gate_mode}',
        fontsize=11,
    )
    ax3.set_xlabel('Final TNA (€M)')
    ax3.set_ylabel('Number of paths')
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'€{x:.0f}M'))
    ax3.legend(frameon=False)
    plt.tight_layout()
    tna_dist_path = _single_runs_dir / f'final_tna_dist_{_run_tag}.png'
    fig3.savefig(tna_dist_path, dpi=150)
    plt.close(fig3)
    print(f"✓ Final TNA distribution saved → single_runs/{tna_dist_path.name}")

    # ── Figure 4: Credit usage by regime ─────────────────────────────────────
    # Per-path, per-regime: did this path ever draw credit in this regime?
    _res_reset = results.reset_index()
    regime_credit_prob = (
        _res_reset.groupby(['path', 'Regime'])['Shortfall_Flag']
        .apply(lambda x: (x >= 1).any())
        .reset_index()
        .groupby('Regime')['Shortfall_Flag']
        .mean()
        .reindex(_regime_order)
        .dropna()
    )

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    colors = ['#2ecc71', '#e67e22', '#e74c3c', '#9b59b6']
    bars = ax4.bar(regime_credit_prob.index, regime_credit_prob.values,
                   color=colors[:len(regime_credit_prob)], edgecolor='white')
    for bar, val in zip(bars, regime_credit_prob.values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.1%}', ha='center', va='bottom', fontsize=10)
    ax4.set_title(
        f'Credit Draw Probability by Macro Regime\n'
        f'cash={args.cash_weight:.0%}  credit={args.credit_cap:.0%}  gate={args.gate_mode}',
        fontsize=11,
    )
    ax4.set_ylabel('P(≥1 credit draw | regime)')
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax4.set_ylim(0, min(1.05, regime_credit_prob.max() * 1.25))
    plt.tight_layout()
    regime_path = _single_runs_dir / f'credit_by_regime_{_run_tag}.png'
    fig4.savefig(regime_path, dpi=150)
    plt.close(fig4)
    print(f"✓ Credit by regime saved → single_runs/{regime_path.name}")
