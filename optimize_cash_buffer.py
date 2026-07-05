#!/usr/bin/env python3
"""
Cash Buffer Optimisation — Efficient Frontier
===================================================================
Runs a grid search over cash_weight for two credit-capacity scenarios:
  • ample  (credit line = 10% of TNA — easy replenishment after redemptions)
  • tight  (credit line =  2% of TNA — fund must sell PM to replenish buffer)

For each (scenario, cash_weight) combination the simulation is run and the
following metrics are computed:
  Y-axis: E[log(TNA_T/TNA_0)] / horizon_years  (annualised log-growth)
  X-axis: P(≥1 forced liquidation per path)     (investor-relevant tail risk)

Optimal cash weights are derived via the BSV (2009) CRRA utility:
  w* = argmax (1/N) Σ_paths U_γ(W_T)

  where  U_γ(W) = W^(1-γ) / (1-γ)   for γ ≠ 1   (CRRA / power utility)
         U_1(W) = log(W)              for γ = 1   (log utility)
         W_T    = investor wealth ratio (per-unit compounded return)

Risk-aversion grid: γ ∈ {1, 1.25, …, 5}  (default)
  γ = 1  → log utility; least risk-averse
  γ = 2  → standard macro-finance benchmark
  γ = 5  → highly risk-averse

Grid search is parallelised across CPU cores via multiprocessing.Pool
(initializer pattern — shared data sent once per worker, not once per call).

Outputs
-------
  eltif_optimization_frontier{tag}.csv  — one row per (scenario, cash_weight)
  eltif_optimization_optimal{tag}.csv   — one row per (scenario, γ): w*, E[U_γ], …

Usage
-----
  python optimize_cash_buffer.py                       # all defaults
  python optimize_cash_buffer.py --n_paths 100         # fast test
  python optimize_cash_buffer.py --run_tag _bond_usd   # tagged run
"""

import argparse
import copy
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

# ── Parse arguments ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Cash buffer grid search — efficient frontier (BSV 2009 CRRA)'
)
parser.add_argument('--n_paths',       type=int,   default=500,
                    help='Monte Carlo paths per grid point (default 500)')
parser.add_argument('--horizon_years', type=int,   default=10,
                    help='Simulation horizon in years (default 10 = 40 quarters)')
parser.add_argument('--initial_tna',   type=float, default=100_000_000,
                    help='Initial fund TNA in EUR (default 100M)')
parser.add_argument('--credit_cap',    type=float, default=0.05,
                    help='Credit line capacity as fraction of TNA (default 0.05)')
parser.add_argument('--credit_spr',    type=float, default=0.03,
                    help='Credit spread over base rate (default 0.03)')
parser.add_argument('--haircut_rate',  type=float, default=0.20,
                    help='Haircut on forced PM liquidations (default 0.20)')
parser.add_argument('--ter',           type=float, default=0.015,
                    help='Annual TER / management fee (default 0.015)')
parser.add_argument('--gammas',        type=str,   default='1,2,5',
                    help='Comma-separated CRRA γ values (default 1,2,5)')
parser.add_argument('--pm_index',      type=str,   default='composite',
                    choices=['composite', 'credit', 'equity', 'equal_weight'],
                    help='Hamilton Lane index for PM return (default composite)')
parser.add_argument('--run_tag',       type=str,   default='',
                    help='Suffix appended to output filenames, e.g. _bond_usd_h10_s300')
parser.add_argument('--coef_pkl',      type=str,   default='',
                    help='Path to a specific coefficients .pkl file.')
parser.add_argument('--max_cash_weight', type=float, default=0.50,
                    help='Upper bound of cash weight grid (default 0.50)')
parser.add_argument('--gate_mode', type=str, default='strict',
                    choices=['strict', 'economic'],
                    help='strict: gate caps total outflow (ELTIF 2.0) | economic: fixed TNA-based gate')
parser.add_argument('--redemption_gate_pct', type=float, default=0.20,
                    help='Economic mode: fixed TNA-based redemption cap (default 0.20 = 20%%)')
args = parser.parse_args()


# ── Grids ─────────────────────────────────────────────────────────────────────
def float_range(start, stop, step):
    """Floating-point safe range using round-trip via integer arithmetic."""
    n = round((stop - start) / step)
    return [round(start + i * step, 10) for i in range(n + 1)]

CASH_WEIGHT_GRID = float_range(0.05, args.max_cash_weight, 0.01)
GAMMA_VALUES     = float_range(1.0, 5.0, 0.25)    # 17 points: 1.0 … 5.0

SCENARIOS = {
    'ample': 0.20,   # credit line = 20% of TNA (easy replenishment)
    'tight': 0.05,
    'reg_max': 0.50  # credit line = 50% of TNA
}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))


# ══════════════════════════════════════════════════════════════════════════════
# MULTIPROCESSING WORKER
# ══════════════════════════════════════════════════════════════════════════════

# Shared read-only data injected by Pool initializer (set once per worker process)
_PM_ALL: 'pd.DataFrame | None' = None
_COEFS_MASTER: 'dict | None'   = None


def _init_worker(pm_data: pd.DataFrame, coefs_master: dict) -> None:
    """Pool initializer — stores shared data as process-global variables."""
    global _PM_ALL, _COEFS_MASTER
    _PM_ALL       = pm_data
    _COEFS_MASTER = coefs_master


def simulate_one_point(
    scenario_name:        str,
    credit_cap_scenario:  float,
    cash_weight:          float,
    initial_tna:          float,
    ter:                  float,
    credit_spr:           float,
    haircut_rate:         float,
    pm_index:             str,
    horizon_years:        int,
    gate_mode:            str,
    redemption_gate_pct:  float,
) -> dict:
    """
    Simulate one (scenario, cash_weight) grid point.
    Called in a worker process; accesses _PM_ALL / _COEFS_MASTER globals.
    Returns a frontier row dict.
    """
    import Core_fund_simulation as _cs
    coefs_copy = copy.deepcopy(_COEFS_MASTER)
    results = _cs.simulate_eltif_multipaths(
        _PM_ALL,
        coefs_copy,
        cash_returns=None,   # per-path EUR003_Index extracted inside simulate_eltif_multipaths
        initial_tna=initial_tna,
        ter=ter,
        cash_weight=cash_weight,
        credit_capacity=credit_cap_scenario,
        credit_spread=credit_spr,
        haircut_rate=haircut_rate,
        gate_mode=gate_mode,
        redemption_gate_pct=redemption_gate_pct,
        pm_index=pm_index,
        verbose=False,
    )
    metrics = _cs.calculate_optimization_metrics(
        results,
        initial_tna=initial_tna,
        horizon_years=horizon_years,
    )
    crra = {
        gamma: _cs.calculate_crra_utility(results, initial_tna, gamma)
        for gamma in GAMMA_VALUES
    }
    stoxx600_log_ann  = 4.0 * float(np.log1p(_PM_ALL['STOXX600_Return']).mean())
    illiq_premium_ann = metrics['expected_log_growth_ann'] - stoxx600_log_ann
    return {
        'scenario':            scenario_name,
        'gate_mode':           gate_mode,
        'credit_cap_scenario': credit_cap_scenario,
        'cash_weight':         cash_weight,
        'pm_index':            pm_index,
        'illiq_premium_ann':   illiq_premium_ann,
        **metrics,
        **{f'crra_g{g}': v for g, v in crra.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    print("=" * 70)
    print("CASH BUFFER OPTIMISATION — EFFICIENT FRONTIER  (BSV 2009 CRRA)")
    print("=" * 70)
    print(f"  Grid:             {[f'{w:.0%}' for w in CASH_WEIGHT_GRID]}")
    if args.gate_mode == 'strict':
        print(f"  Gate mode:        strict — gate = 50% × target weight w (as % of TNA) per qtr (ELTIF 2.0)")
    else:
        print(f"  Gate mode:        economic — fixed {args.redemption_gate_pct:.0%} of TNA/qtr")
    print(f"  Scenarios:        {list(SCENARIOS.keys())}  "
          f"(credit caps: {list(SCENARIOS.values())})")
    print(f"  Paths/point:      {args.n_paths:,}")
    print(f"  Horizon:          {args.horizon_years} years ({args.horizon_years * 4} quarters)")
    print(f"  Initial TNA:      €{args.initial_tna:,.0f}")
    print(f"  Credit cap:       {args.credit_cap:.0%}")
    print(f"  Credit spread:    {args.credit_spr:.2%}")
    print(f"  Haircut rate:     {args.haircut_rate:.0%}")
    print(f"  TER:              {args.ter:.2%}")
    print(f"  PM index:         {args.pm_index}")
    print(f"  γ (CRRA) values:  {GAMMA_VALUES}")
    print("=" * 70)

    import importlib
    import re as _re
    import Core_fund_simulation as cs

    # ── Load coefficients ──────────────────────────────────────────────────────
    _coef_pkl = args.coef_pkl
    if not _coef_pkl and args.run_tag:
        _reg_tag     = _re.sub(r'_h\d+_s\d+$', '', args.run_tag)
        _runs_pkl    = BASE_DIR / f'LSEG_merged/runs/goldstein_model2_coefficients{_reg_tag}.pkl'
        _default_pkl = BASE_DIR / f'LSEG_merged/goldstein_model2_coefficients{_reg_tag}.pkl'
        if _runs_pkl.exists():
            _coef_pkl = str(_runs_pkl)
        elif _default_pkl.exists():
            _coef_pkl = str(_default_pkl)

    if _coef_pkl:
        if Path(_coef_pkl).exists():
            cs.coefficients = cs.load_goldstein_coefficients(_coef_pkl)
        else:
            print(f"⚠  pkl not found ({_coef_pkl}) — falling back to default coefficients")
            importlib.reload(cs)
    else:
        importlib.reload(cs)   # default path

    if cs.coefficients is None:
        raise RuntimeError(
            "No Goldstein coefficients loaded. Run a regression first "
            "or pass --coef_pkl / --run_tag explicitly."
        )

    # ── Prepare patched coefficients ───────────────────────────────────────────
    coefficients_master = copy.deepcopy(cs.coefficients)
    ctrl = coefficients_master['macro_regime']['controls']

    raw_intercept  = ctrl['intercept']
    flow_rate_mean = ctrl.get('flow_rate_mean', None)
    if flow_rate_mean is not None:
        ctrl['intercept'] = flow_rate_mean
        print(f"\n  β₀ replaced with flow_rate_mean: "
              f"{flow_rate_mean:+.4f} ({flow_rate_mean:.2%}/qtr)")
    else:
        print(f"\n  ⚠ flow_rate_mean not found — using raw β₀={raw_intercept:.4f}")

    mean_log_tna = ctrl.get('mean_log_tna', None)
    if mean_log_tna is not None:
        log_tna_coef  = ctrl.get('log_tna', 0.0)
        centering_adj = -log_tna_coef * mean_log_tna
        ctrl['intercept'] += centering_adj
        print(f"  Centering adj:  {centering_adj:+.4f}  "
              f"(log_tna_coef={log_tna_coef:+.4f}, mean_log_tna={mean_log_tna:.4f})")
        print(f"  Final intercept:{ctrl['intercept']:+.4f} ({ctrl['intercept']:.2%}/qtr)")
    else:
        print(f"  ⚠ mean_log_tna not found — log_tna level drag not corrected")

    # ── Select and truncate paths ──────────────────────────────────────────────
    all_paths    = cs.pm_cash_sim_returns['path'].unique()
    n_use        = min(args.n_paths, len(all_paths))
    if n_use < args.n_paths:
        print(f"  ⚠ Only {n_use:,} paths available (requested {args.n_paths:,})")
    paths_to_use = all_paths[:n_use]

    n_quarters_max = args.horizon_years * 4
    pm_all = (
        cs.pm_cash_sim_returns[cs.pm_cash_sim_returns['path'].isin(paths_to_use)]
        .copy()
        .groupby('path')
        .head(n_quarters_max)
        .reset_index(drop=True)
    )
    print(f"\n  Using {n_use:,} paths × {n_quarters_max} quarters "
          f"({args.horizon_years} years)")

    # ── Parallel grid search ───────────────────────────────────────────────────
    total_runs = len(SCENARIOS) * len(CASH_WEIGHT_GRID)
    n_cpu      = max(1, os.cpu_count() - 1)

    args_list = [
        (sc, credit_cap_sc, w,
         args.initial_tna, args.ter, args.credit_spr,
         args.haircut_rate, args.pm_index, args.horizon_years,
         args.gate_mode, args.redemption_gate_pct)
        for sc, credit_cap_sc in SCENARIOS.items()
        for w in CASH_WEIGHT_GRID
    ]

    print(f"\nRunning {total_runs} grid points across {n_cpu} cores …")

    with Pool(
        processes=n_cpu,
        initializer=_init_worker,
        initargs=(pm_all, coefficients_master),
    ) as pool:
        frontier_rows = pool.starmap(simulate_one_point, args_list)

    df_frontier = (
        pd.DataFrame(frontier_rows)
        .sort_values(['scenario', 'cash_weight'])
        .reset_index(drop=True)
    )

    # ── Optimal weight per (scenario, γ) via BSV CRRA ─────────────────────────
    optimal_rows = []
    for scenario_name in SCENARIOS:
        df_s = df_frontier[df_frontier['scenario'] == scenario_name].copy()
        for gamma in GAMMA_VALUES:
            col      = f'crra_g{gamma}'
            best_idx = df_s[col].idxmax()
            best_row = df_s.loc[best_idx]
            optimal_rows.append({
                'scenario':                scenario_name,
                'gate_mode':               args.gate_mode,
                'gamma':                   gamma,
                'optimal_cash_weight':     best_row['cash_weight'],
                'crra_utility':            best_row[col],
                'expected_log_growth_ann': best_row['expected_log_growth_ann'],
                'illiq_premium_ann':       best_row['illiq_premium_ann'],
                'median_final_tna':        best_row['median_final_tna'],
                'p_fl_any_path':           best_row['p_fl_any_path'],
                'p_credit_any_path':       best_row['p_credit_any_path'],
            })

    df_optimal = pd.DataFrame(optimal_rows)

    # ── Save outputs ───────────────────────────────────────────────────────────
    if args.run_tag:
        _opt_out_dir = BASE_DIR / 'optimization_runs'
        _opt_out_dir.mkdir(exist_ok=True)
    else:
        _opt_out_dir = BASE_DIR

    frontier_path = _opt_out_dir / f'eltif_optimization_frontier{args.run_tag}_{args.gate_mode}.csv'
    optimal_path  = _opt_out_dir / f'eltif_optimization_optimal{args.run_tag}_{args.gate_mode}.csv'

    df_frontier.to_csv(frontier_path, index=False)
    df_optimal.to_csv(optimal_path,   index=False)

    print("\n" + "=" * 70)
    print("✅ OPTIMISATION COMPLETE")
    print(f"   Frontier saved  → {frontier_path.relative_to(BASE_DIR)}")
    print(f"   Optimal weights → {optimal_path.relative_to(BASE_DIR)}")
    print("=" * 70)
