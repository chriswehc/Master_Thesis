#!/usr/bin/env python3
"""
run_grid_batch.py
═══════════════════════════════════════════════════════════════════════════════
Batch runner: executes optimization_grid_analysis.py for every combination of
gate_mode × redemption_gate_pct defined below.

Edit RUNS to add/remove combinations, then:
    python run_grid_batch.py
    python run_grid_batch.py --reg_tag _bond_hy
    python run_grid_batch.py --skip_existing          # skip already-done cells
    python run_grid_batch.py --summary_only           # rebuild summaries only
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR   = Path(__file__).parent
GRID_SCRIPT = BASE_DIR / 'optimization_grid_analysis.py'

# ── Define runs ───────────────────────────────────────────────────────────────
# Each entry: (gate_mode, redemption_gate_pct)
# redemption_gate_pct is ignored for gate_mode='strict'
RUNS = [
    ('strict',   0.50),
    ('economic', 0.10),   # 10% of TNA per quarter
    ('economic', 0.20),   # 20% of TNA per quarter (baseline)
    ('economic', 0.50),   # 50% of TNA per quarter
]

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Batch grid runner over gate modes.')
parser.add_argument('--reg_tag',   type=str, default='',
                    help='Regression tag passed through to grid analysis (e.g. _bond_hy).')
parser.add_argument('--n_paths',   type=int, default=1000,
                    help='Monte Carlo paths per grid point.')
parser.add_argument('--haircuts',  type=str, default='5,10,20,30',
                    help='Comma-separated haircut values in percent.')
parser.add_argument('--spreads',   type=str, default='100,200,300,500',
                    help='Comma-separated credit-spread values in basis points.')
parser.add_argument('--pm_index',  type=str, default='composite',
                    help='Comma-separated pm_index values to run, e.g. "equity,credit". '
                         'Choices per value: composite, credit, equity, equal_weight.')
parser.add_argument('--skip_existing', action='store_true',
                    help='Skip optimizer cells whose output CSV already exists.')
parser.add_argument('--summary_only',  action='store_true',
                    help='Skip all optimizer runs; only rebuild summary CSVs/TXTs/TEX.')
parser.add_argument('--frontier_plots', action='store_true',
                    help='Generate efficient frontier PNG plots for each grid cell.')
args = parser.parse_args()

# ── Run ───────────────────────────────────────────────────────────────────────
_VALID_PM_INDICES = {'composite', 'credit', 'equity', 'equal_weight'}
pm_indices = [v.strip() for v in args.pm_index.split(',')]
invalid = [v for v in pm_indices if v not in _VALID_PM_INDICES]
if invalid:
    print(f'ERROR: unknown pm_index value(s): {invalid}')
    print(f'       valid choices: {sorted(_VALID_PM_INDICES)}')
    sys.exit(1)

n_total  = len(RUNS) * len(pm_indices)
failed   = []

print('=' * 60)
print('  BATCH GRID OPTIMIZATION')
print(f'  {len(RUNS)} gate run(s) × {len(pm_indices)} pm_index = {n_total} total')
print(f'  reg_tag      : {args.reg_tag!r}')
print(f'  n_paths      : {args.n_paths:,}')
print(f'  haircuts     : {args.haircuts}')
print(f'  spreads      : {args.spreads}')
print(f'  pm_index     : {pm_indices}')
print(f'  skip_existing: {args.skip_existing}')
print(f'  summary_only : {args.summary_only}')
print('=' * 60)

run_num = 0
for pm_idx in pm_indices:
    for gate_mode, gate_pct in RUNS:
        run_num += 1
        label = (f'pm_index={pm_idx}  gate={gate_mode}'
                 + (f'  gate_pct={gate_pct:.0%}' if gate_mode == 'economic' else ''))
        print(f'\n{"═"*60}')
        print(f'  [{run_num}/{n_total}]  {label}')
        print(f'{"═"*60}\n')

        cmd = [
            sys.executable, str(GRID_SCRIPT),
            '--reg_tag',             args.reg_tag,
            '--n_paths',             str(args.n_paths),
            '--haircuts',            args.haircuts,
            '--spreads',             args.spreads,
            '--pm_index',            pm_idx,
            '--gate_mode',           gate_mode,
            '--redemption_gate_pct', str(gate_pct),
        ]
        if args.skip_existing:
            cmd.append('--skip_existing')
        if args.summary_only:
            cmd.append('--summary_only')
        if args.frontier_plots:
            cmd.append('--frontier_plots')

        rc = subprocess.run(cmd, cwd=str(BASE_DIR)).returncode
        if rc != 0:
            print(f'\n❌  FAILED (rc={rc})  {label}')
            failed.append((pm_idx, gate_mode, gate_pct))
        else:
            print(f'\n✅  Done  [{run_num}/{n_total}]  {label}')

print('\n' + '=' * 60)
print(f'  BATCH COMPLETE  —  {n_total - len(failed)}/{n_total} succeeded')
if failed:
    print(f'  Failed: {failed}')
print('=' * 60)
