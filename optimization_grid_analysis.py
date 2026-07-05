"""
optimization_grid_analysis.py
══════════════════════════════════════════════════════════════════════════════
Batch runner for optimize_cash_buffer.py over a haircut × credit-spread grid.

"""

import argparse
import sys
import subprocess
import pickle
import re
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OPT_SCRIPT  = BASE_DIR / 'optimize_cash_buffer.py'
LSEG_DIR    = BASE_DIR / 'LSEG_merged'
RUNS_DIR    = LSEG_DIR / 'runs'
OPT_DIR     = BASE_DIR / 'optimization_runs'

# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Grid-search optimizer over haircut × credit-spread.')
parser.add_argument('--reg_tag',   type=str, default='',
                    help='Regression run tag, e.g. _bond_usd  (empty = pooled default pkl).')
parser.add_argument('--haircuts',  type=str, default='5,10,20,30',
                    help='Comma-separated haircut values in percent, e.g. "5,10,20,30".')
parser.add_argument('--spreads',   type=str, default='100,200,300,500',
                    help='Comma-separated credit-spread values in basis points, e.g. "100,200,300,500".')
parser.add_argument('--n_paths',   type=int, default=1000,
                    help='Monte Carlo paths per grid point (passed to optimizer).')
parser.add_argument('--gammas',    type=str,
                    default='1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0',
                    help='Comma-separated CRRA γ values.')
parser.add_argument('--pm_index',  type=str, default='composite',
                    choices=['composite', 'credit', 'equity', 'equal_weight'],
                    help='Hamilton Lane index.')
parser.add_argument('--skip_existing', action='store_true',
                    help='Skip cells whose output CSV already exists in optimization_runs/.')
parser.add_argument('--summary_only', action='store_true',
                    help='Skip all optimizer runs and only rebuild the summary CSV/TXT from existing files.')
parser.add_argument('--gate_mode', type=str, default='strict',
                    choices=['strict', 'economic'],
                    help='strict: gate caps total outflow (ELTIF 2.0) | economic: fixed TNA-based gate')
parser.add_argument('--redemption_gate_pct', type=float, default=0.20,
                    help='Economic mode: fixed TNA-based redemption cap (default 0.20 = 20%%)')
parser.add_argument('--frontier_plots', action='store_true',
                    help='Generate efficient frontier PNG plots for each grid cell '
                         'and cross-haircut sensitivity figures.')
args = parser.parse_args()

# ── Parse grid ────────────────────────────────────────────────────────────
haircuts = [float(x) / 100.0 for x in args.haircuts.split(',')]
spreads  = [float(x) / 10000.0 for x in args.spreads.split(',')]
reg_tag  = args.reg_tag.strip()

# gate_tag_suffix distinguishes economic runs by redemption_gate_pct in filenames
# while leaving reg_tag clean for pkl coefficient lookup.
if args.gate_mode == 'economic':
    gate_tag_suffix = f'_g{int(round(args.redemption_gate_pct * 100))}'
else:
    gate_tag_suffix = ''

n_cells = len(haircuts) * len(spreads)

# ── Locate coefficients pkl ───────────────────────────────────────────────
_runs_pkl    = RUNS_DIR / f'goldstein_model2_coefficients{reg_tag}.pkl'
_default_pkl = LSEG_DIR / f'goldstein_model2_coefficients{reg_tag}.pkl'
if _runs_pkl.exists():
    coef_pkl = _runs_pkl
elif _default_pkl.exists():
    coef_pkl = _default_pkl
else:
    print(f"❌ Coefficients pkl not found for reg_tag={reg_tag!r}.")
    print(f"   Looked in: {_runs_pkl}")
    print(f"          and: {_default_pkl}")
    print("   Run the regression first via regression_input_analysis.py or the dashboard.")
    sys.exit(1)

print(f"✓ Using coefficients: {coef_pkl.relative_to(BASE_DIR)}")

OPT_DIR.mkdir(exist_ok=True)

# ── Helper: auto tag ──────────────────────────────────────────────────────
_pm_tag = f"_pm_{args.pm_index}" if args.pm_index != "composite" else ""

def auto_tag(h: float, s: float) -> str:
    return f"{reg_tag}{_pm_tag}{gate_tag_suffix}_h{int(round(h * 100))}_s{int(round(s * 10000))}"


# ── Helper: run one optimizer call ───────────────────────────────────────
def run_optimizer(h: float, s: float, idx: int) -> int:
    tag = auto_tag(h, s)
    print(f"\n{'═' * 60}")
    print(f"  [{idx:>{len(str(n_cells))}}/{n_cells}]  "
          f"haircut={int(round(h*100)):>3}%  "
          f"spread={int(round(s*10000)):>4}bps  "
          f"tag={tag}")
    print(f"{'═' * 60}\n")

    # Skip if output already exists
    if args.skip_existing:
        out = OPT_DIR / f'eltif_optimization_optimal{tag}_{args.gate_mode}.csv'
        if out.exists():
            print(f"⏭  Skipping — output already exists: {out.name}")
            return 0

    cmd = [
        sys.executable, str(OPT_SCRIPT),
        '--run_tag',      tag,
        '--coef_pkl',     str(coef_pkl),
        '--haircut_rate', str(h),
        '--credit_spr',   str(s),
        '--n_paths',      str(args.n_paths),
        '--gammas',           args.gammas,
        '--pm_index',         args.pm_index,
        '--max_cash_weight',  '0.50',
        '--gate_mode',            args.gate_mode,
        '--redemption_gate_pct',  str(args.redemption_gate_pct),
    ]
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    return result.returncode


# ══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 60}")
print(f"  OPTIMIZATION GRID ANALYSIS")
print(f"  reg_tag   : {reg_tag!r}  (pkl: {coef_pkl.name})")
print(f"  haircuts  : {[f'{int(h*100)}%' for h in haircuts]}")
print(f"  spreads   : {[f'{int(s*10000)}bps' for s in spreads]}")
print(f"  grid cells: {n_cells}  ({len(haircuts)} × {len(spreads)})")
print(f"  n_paths   : {args.n_paths:,} per cell")
print(f"  pm_index  : {args.pm_index}")
print(f"  gate_mode : {args.gate_mode}"
      + (f"  (redemption_gate={args.redemption_gate_pct:.0%} of TNA)" if args.gate_mode == 'economic' else ""))
print(f"  skip_exist: {args.skip_existing}")
print(f"{'═' * 60}")

failed = []
idx    = 0

if args.summary_only:
    print("⏭  --summary_only: skipping all optimizer runs, rebuilding summary from existing files.")
else:
    for h in haircuts:
        for s in spreads:
            idx += 1
            rc = run_optimizer(h, s, idx)
            if rc != 0:
                print(f"\n❌ FAILED (rc={rc})  haircut={int(h*100)}%  spread={int(s*10000)}bps")
                failed.append(auto_tag(h, s))
            else:
                print(f"\n✅ Done  [{idx}/{n_cells}]")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 60}")
print(f"  GRID COMPLETE  —  {n_cells - len(failed)}/{n_cells} runs succeeded")
if failed:
    print(f"  Failed tags: {failed}")
print(f"{'═' * 60}\n")

# Collect all optimal CSVs + join expected_shortfall_fl_cond from frontier CSVs
records = []
for h in haircuts:
    for s in spreads:
        tag      = auto_tag(h, s)
        opt_path = OPT_DIR / f'eltif_optimization_optimal{tag}_{args.gate_mode}.csv'
        frt_path = OPT_DIR / f'eltif_optimization_frontier{tag}_{args.gate_mode}.csv'
        if not opt_path.exists():
            continue
        df = pd.read_csv(opt_path)

        # Join metrics from frontier: shortfall at w* and baseline return at w=0
        if frt_path.exists():
            frt = pd.read_csv(frt_path)
            frt['_cw_r'] = frt['cash_weight'].round(4)
            df['_cw_r']  = df['optimal_cash_weight'].round(4)

            # expected_shortfall_fl_cond at optimal w*
            frt_sub = frt[['scenario', '_cw_r',
                            'expected_shortfall_fl_cond']].drop_duplicates()
            df = df.merge(frt_sub, on=['scenario', '_cw_r'], how='left')
            df.drop(columns=['_cw_r'], inplace=True)

            # baseline (w=5%) annualised log return — for return-drag computation
            frt_w0 = (
                frt[frt['cash_weight'].round(4) == 0.05]
                [['scenario', 'expected_log_growth_ann']]
                .rename(columns={'expected_log_growth_ann': 'baseline_log_return_ann'})
            )
            df = df.merge(frt_w0, on='scenario', how='left')
        else:
            df['expected_shortfall_fl_cond'] = float('nan')
            df['baseline_log_return_ann']    = float('nan')
            df['illiq_premium_ann']          = float('nan')

        df['h_rate']      = h                      # float, e.g. 0.10
        df['haircut_pct'] = int(round(h * 100))
        df['spread_bps']  = int(round(s * 10000))
        df['run_tag']     = tag
        df['pm_index']    = args.pm_index
        records.append(df)

if not records:
    print("⚠  No output CSVs found — nothing to summarise.")
    sys.exit(0)

summary = pd.concat(records, ignore_index=True)

# Backfill gate_mode for CSVs generated before this column was added
if 'gate_mode' not in summary.columns:
    summary['gate_mode'] = args.gate_mode

# ── Compute derived metrics ────────────────────────────────────────────────
# Return drag: how many bps/yr the investor gives up at w* vs holding no buffer
summary['return_drag_bps'] = (
    summary['baseline_log_return_ann'] - summary['expected_log_growth_ann']
) * 10_000

# Expected haircut cost per FL event as % of initial TNA (€100M)
# = (forced PM liquidation size × haircut rate) / initial TNA
summary['haircut_cost_pct_tna'] = (
    summary['expected_shortfall_fl_cond'] * summary['h_rate'] / 100_000_000 * 100
)

if 'illiq_premium_ann' not in summary.columns:
    summary['illiq_premium_ann'] = float('nan')
summary['illiq_premium_bps'] = summary['illiq_premium_ann'] * 10_000

# Save full grid CSV
csv_path = OPT_DIR / f'grid_summary{reg_tag}{_pm_tag}{gate_tag_suffix}_{args.gate_mode}.csv'
summary.to_csv(csv_path, index=False)
print(f"✓ Full grid CSV saved to: {csv_path.relative_to(BASE_DIR)}")

# ── Print metric heatmaps (γ=2, both scenarios) ────────────────────────────
GAMMA_DISP = 2.0
_SC_ORDER  = ['ample', 'tight', 'reg_max']
SCENARIOS  = (
    [s for s in _SC_ORDER if s in summary['scenario'].unique()]
    + [s for s in summary['scenario'].unique() if s not in _SC_ORDER]
)
spread_labels  = [f"{int(round(s*10000))}bps" for s in spreads]
haircut_labels = [f"{int(round(h*100))}%" for h in haircuts]
W = 9 if len(SCENARIOS) >= 3 else 11  # column width per cell

# Each metric: (title, column name in summary, formatter)
METRICS = [
    ("Panel A: Optimal cash weight  w*",
     'optimal_cash_weight',
     lambda v: f"{v:.0%}"),
    ("Panel B: Ann. log return at w*  (E[r_log])",
     'expected_log_growth_ann',
     lambda v: f"{v:.2%}"),
    ("Panel C: Return drag vs no-buffer  (bps/yr)",
     'return_drag_bps',
     lambda v: f"{v:.0f}bps" if not pd.isna(v) else "—"),
    ("Panel D: E[haircut cost | FL]  as % of initial TNA",
     'haircut_cost_pct_tna',
     lambda v: f"{v:.2f}%" if not pd.isna(v) else "—"),
    ("Panel E: Illiquidity premium at w*  vs STOXX 600  (bps/yr)",
     'illiq_premium_bps',
     lambda v: f"{v:.0f}bps" if not pd.isna(v) else "—"),
]

_SC_ABBR = {'ample': 'amp', 'tight': 'tgt', 'reg_max': 'reg'}

def _make_header() -> str:
    hdr = f"{'Haircut':>10}  "
    for sc in SCENARIOS:
        for sl in spread_labels:
            sc_abbr = _SC_ABBR.get(sc, sc[:3])
            hdr += f"{sc_abbr}/{sl}".center(W)
    return hdr

lines = []
emit  = lines.append

for metric_title, col, fmt in METRICS:
    emit("")
    emit(f"{'─' * 4}  {metric_title}  (γ={GAMMA_DISP})  {'─' * 4}")
    hdr = _make_header()
    emit(hdr)
    emit("─" * len(hdr))
    for h, hl in zip(haircuts, haircut_labels):
        row = f"{hl:>10}  "
        for sc in SCENARIOS:
            for s in spreads:
                tag = auto_tag(h, s)
                sub = summary[
                    (summary['run_tag']  == tag) &
                    (summary['scenario'] == sc)  &
                    (summary['gamma']    == GAMMA_DISP) &
                    (summary['gate_mode'] == args.gate_mode)
                ]
                if sub.empty or col not in sub.columns:
                    row += "—".center(W)
                else:
                    row += fmt(sub.iloc[0][col]).center(W)
        emit(row)
    emit("─" * len(hdr))

emit("")
emit(f"Note: γ={GAMMA_DISP} (standard macro-finance investor). "
     f"Return drag = ann. log return at w=5% minus return at w*. "
     f"Haircut cost = E[forced PM liquidated | FL] × h / initial TNA (€100M). "
     f"Illiquidity premium = ann. log return at w* minus STOXX 600 annualised log return (same paths).")

output = "\n".join(lines)
print(output)

txt_path = OPT_DIR / f'grid_summary{reg_tag}{_pm_tag}{gate_tag_suffix}_{args.gate_mode}.txt'
txt_path.write_text(output + "\n", encoding='utf-8')
print(f"\n✓ Summary table saved to: {txt_path.relative_to(BASE_DIR)}")

# ══════════════════════════════════════════════════════════════════════════
# POST-RUN: Panel F LaTeX and gamma sensitivity (economic gate only)
# ══════════════════════════════════════════════════════════════════════════
if args.gate_mode == 'economic':
    gate_pct = int(round(args.redemption_gate_pct * 100))

    print(f"\n{'═' * 60}")
    print(f"  POST-RUN: Generating Panel F LaTeX  (gate_pct={gate_pct}%)")
    print(f"{'═' * 60}")
    panel_f_script = BASE_DIR / 'panel_f_crra_improvement.py'
    rc = subprocess.run(
        [sys.executable, str(panel_f_script), '--gate_pct', str(gate_pct)],
        cwd=str(BASE_DIR)
    ).returncode
    if rc == 0:
        print(f"✅ Panel F LaTeX generated.")
    else:
        print(f"❌ panel_f_crra_improvement.py failed (rc={rc}).")

    if gate_pct == 10:
        print(f"\n{'═' * 60}")
        print(f"  POST-RUN: Generating gamma sensitivity figure & table")
        print(f"{'═' * 60}")
        gamma_script = BASE_DIR / 'gamma_sensitivity_analysis.py'
        rc = subprocess.run(
            [sys.executable, str(gamma_script)],
            cwd=str(BASE_DIR)
        ).returncode
        if rc == 0:
            print(f"✅ gamma_sensitivity_figure.png and LaTeX table generated.")
        else:
            print(f"❌ gamma_sensitivity_analysis.py failed (rc={rc}).")

# ── LaTeX table output ─────────────────────────────────────────────────────────
# Build credit-cap label per scenario from summary data
_SC_LABEL_TMPL = {
    'ample':   'Ample ({cap:.0%} credit)',
    'tight':   'Tight ({cap:.0%} credit)',
    'reg_max': r'Reg.~Max ({cap:.0%} credit)',
}

def _sc_label(sc: str) -> str:
    sub = summary[summary['scenario'] == sc]
    if sub.empty or 'credit_cap_scenario' not in sub.columns:
        return sc
    cap = sub['credit_cap_scenario'].iloc[0]
    tmpl = _SC_LABEL_TMPL.get(sc, '{sc} ({cap:.0%} credit)')
    return tmpl.format(cap=cap, sc=sc)


def _latex_escape(s: str) -> str:
    return s.replace('%', r'\%')


n_spreads = len(spreads)

def build_latex_panel(panel_title: str, col: str, fmt_latex) -> str:
    """Return a complete LaTeX table environment string for one metric panel."""
    # column spec: l | (r×n_spreads) per scenario, separated by |
    col_parts   = ['l'] + [('r' * n_spreads) for _ in SCENARIOS]
    col_spec    = '|'.join(col_parts)

    # scenario multicolumn headers
    sc_headers = []
    for i, sc in enumerate(SCENARIOS):
        last = (i == len(SCENARIOS) - 1)
        mc_fmt = r'\multicolumn{' + str(n_spreads) + r'}{' + ('c' if last else 'c|') + r'}'
        sc_headers.append(mc_fmt + '{' + _latex_escape(_sc_label(sc)) + '}')
    sc_header_row = ' & ' + ' & '.join(sc_headers) + r' \\'

    # spread sub-header
    spread_sub = ' & '.join(str(int(round(s * 10000))) for s in spreads)
    spread_row = r'Haircut & ' + ' & '.join([spread_sub] * len(SCENARIOS)) + r' \\'

    # data rows
    data_rows = []
    for h, hl in zip(haircuts, haircut_labels):
        cells = [_latex_escape(hl)]
        for sc in SCENARIOS:
            for s in spreads:
                tag = auto_tag(h, s)
                sub = summary[
                    (summary['run_tag']   == tag) &
                    (summary['scenario']  == sc)  &
                    (summary['gamma']     == GAMMA_DISP) &
                    (summary['gate_mode'] == args.gate_mode)
                ]
                if sub.empty or col not in sub.columns:
                    cells.append('—')
                else:
                    cells.append(_latex_escape(fmt_latex(sub.iloc[0][col])))
        data_rows.append(' & '.join(cells) + r' \\')

    # label: e.g. tab:grid_panelA_bond_hy_economic
    panel_letter = panel_title.split(':')[0].replace('Panel ', 'panel').replace(' ', '')
    label = f"tab:grid_{panel_letter}{reg_tag}_{args.gate_mode}"

    gate_label = 'strict gate (ELTIF 2.0)' if args.gate_mode == 'strict' else f'economic gate ({args.redemption_gate_pct:.0%} of TNA)'

    lines_tex = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\footnotesize',
        r'\caption{' + _latex_escape(panel_title) + r' ($\gamma=' + f'{GAMMA_DISP:.1f}' + r'$, ' + gate_label + r')}',
        r'\label{' + label + r'}',
        r'\begin{tabular}{' + col_spec + r'}',
        r'\toprule',
        sc_header_row,
        spread_row,
        r'\midrule',
    ] + data_rows + [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines_tex)


METRICS_LATEX = [
    ("Panel A: Optimal cash weight $w^*$",
     'optimal_cash_weight',
     lambda v: f"{v:.0%}"),
    (r"Panel B: Ann.\ log return at $w^*$ $E[r_{\log}]$",
     'expected_log_growth_ann',
     lambda v: f"{v:.2%}"),
    (r"Panel C: Return drag vs.\ no-buffer (bps/yr)",
     'return_drag_bps',
     lambda v: f"{v:.0f} bps" if not pd.isna(v) else "—"),
    (r"Panel D: $E[\text{haircut cost} \mid FL]$ as \% of initial TNA",
     'haircut_cost_pct_tna',
     lambda v: f"{v:.2f}%" if not pd.isna(v) else "—"),
    (r"Panel E: Illiquidity premium at $w^*$ vs.\ STOXX 600 (bps/yr)",
     'illiq_premium_bps',
     lambda v: f"{v:.0f} bps" if not pd.isna(v) else "—"),
]

tex_blocks = [build_latex_panel(t, c, f) for t, c, f in METRICS_LATEX]
tex_content = '\n\n'.join(tex_blocks) + '\n'

tex_path = OPT_DIR / f'grid_table{reg_tag}{_pm_tag}{gate_tag_suffix}_{args.gate_mode}.tex'
tex_path.write_text(tex_content, encoding='utf-8')
print(f"✓ LaTeX tables saved to:   {tex_path.relative_to(BASE_DIR)}")

# ── Efficient frontier plots ───────────────────────────────────────────────────
if args.frontier_plots:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    _SC_COLORS  = {'ample': '#2563eb', 'tight': '#dc2626', 'reg_max': '#16a34a'}
    _SC_NAMES   = {'ample': 'Ample', 'tight': 'Tight', 'reg_max': 'Reg. Max'}
    GAMMA_STAR  = 2.0   # highlight optimal w* for this γ

    plot_dir = OPT_DIR / f'frontier_plots{reg_tag}{gate_tag_suffix}_{args.gate_mode}'
    plot_dir.mkdir(exist_ok=True)

    # ── A. Per-cell frontier plots ─────────────────────────────────────────────
    print(f"\nGenerating per-cell frontier plots → {plot_dir.relative_to(BASE_DIR)}/")

    for h in haircuts:
        for s in spreads:
            tag      = auto_tag(h, s)
            frt_path = OPT_DIR / f'eltif_optimization_frontier{tag}_{args.gate_mode}.csv'
            opt_path = OPT_DIR / f'eltif_optimization_optimal{tag}_{args.gate_mode}.csv'
            if not frt_path.exists():
                continue

            frt = pd.read_csv(frt_path).sort_values('cash_weight')
            opt = pd.read_csv(opt_path) if opt_path.exists() else pd.DataFrame()

            has_shortfall = (
                'expected_shortfall_fl_cond' in frt.columns and
                frt['expected_shortfall_fl_cond'].notna().any()
            )
            n_sub = 2 if has_shortfall else 1
            fig, axes = plt.subplots(1, n_sub, figsize=(6 * n_sub, 5))
            if n_sub == 1:
                axes = [axes]

            for sc in SCENARIOS:
                df_s  = frt[frt['scenario'] == sc]
                color = _SC_COLORS.get(sc, 'gray')
                label = _SC_NAMES.get(sc, sc)

                # optimal w* for γ=GAMMA_STAR
                w_star = None
                if not opt.empty:
                    opt_row = opt[(opt['scenario'] == sc) & (opt['gamma'] == GAMMA_STAR)]
                    if not opt_row.empty:
                        w_star = float(opt_row.iloc[0]['optimal_cash_weight'])

                for ax_i, (xcol, xtitle) in enumerate([
                    ('p_fl_any_path',            'P(≥1 forced liquidation)'),
                    ('expected_shortfall_fl_cond', 'E[shortfall | FL]  (€)'),
                ]):
                    if ax_i >= n_sub:
                        break
                    ax = axes[ax_i]
                    if xcol not in df_s.columns or df_s[xcol].isna().all():
                        continue

                    ax.plot(df_s[xcol], df_s['expected_log_growth_ann'],
                            color=color, linewidth=1.8, marker='o', markersize=4,
                            label=label)

                    # Cash weight labels every other point
                    for i, row in enumerate(df_s.itertuples()):
                        if i % 2 == 0:
                            ax.annotate(
                                f"{row.cash_weight:.0%}",
                                (getattr(row, xcol), row.expected_log_growth_ann),
                                textcoords='offset points', xytext=(0, 6),
                                fontsize=7, ha='center', color=color,
                            )

                    # Star for optimal w*
                    if w_star is not None:
                        star_row = df_s[df_s['cash_weight'].round(4) == round(w_star, 4)]
                        if not star_row.empty:
                            ax.scatter(
                                star_row[xcol], star_row['expected_log_growth_ann'],
                                marker='*', s=180, color=color, zorder=5,
                            )

                    if xcol == 'p_fl_any_path':
                        ax.xaxis.set_major_formatter(
                            mticker.FuncFormatter(lambda x, _: f'{x:.0%}'))
                    else:
                        ax.xaxis.set_major_formatter(
                            mticker.FuncFormatter(lambda x, _: f'€{x/1e6:.1f}M'))
                    ax.yaxis.set_major_formatter(
                        mticker.FuncFormatter(lambda x, _: f'{x:.1%}'))
                    ax.set_xlabel(xtitle, fontsize=9)
                    ax.set_ylabel('E[log growth] ann.', fontsize=9)

            axes[0].legend(fontsize=8, frameon=False)
            fig.suptitle(
                f'Efficient Frontier   h={int(round(h*100))}%  '
                f's={int(round(s*10000))}bps  gate={args.gate_mode}\n'
                f'★ = optimal w* at γ={GAMMA_STAR}',
                fontsize=10,
            )
            plt.tight_layout()
            _gate_label = gate_tag_suffix if gate_tag_suffix else '_strict'
            _plot_tag   = f"{reg_tag}{_pm_tag}{_gate_label}_h{int(round(h * 100))}_s{int(round(s * 10000))}"
            out = plot_dir / f'frontier_{_plot_tag}.png'
            fig.savefig(out, dpi=150)
            plt.close(fig)

    print(f"  ✓ {len(haircuts) * len(spreads)} per-cell plots saved")

    # ── B. Cross-haircut sensitivity figures ───────────────────────────────────
    print("Generating cross-haircut sensitivity plots …")

    _haircut_cmap = plt.get_cmap('plasma')
    _hcolors = [_haircut_cmap(i / max(len(haircuts) - 1, 1))
                for i in range(len(haircuts))]

    for sc in SCENARIOS:
        for s in spreads:
            fig, ax = plt.subplots(figsize=(8, 5))
            plotted = 0
            for h, hcolor in zip(haircuts, _hcolors):
                tag      = auto_tag(h, s)
                frt_path = OPT_DIR / f'eltif_optimization_frontier{tag}_{args.gate_mode}.csv'
                if not frt_path.exists():
                    continue
                frt  = pd.read_csv(frt_path).sort_values('cash_weight')
                df_s = frt[frt['scenario'] == sc]
                if df_s.empty:
                    continue
                ax.plot(df_s['p_fl_any_path'], df_s['expected_log_growth_ann'],
                        color=hcolor, linewidth=1.8, marker='o', markersize=4,
                        label=f'h={int(round(h*100))}%')
                plotted += 1

            if plotted == 0:
                plt.close(fig)
                continue

            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0%}'))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1%}'))
            ax.set_xlabel('P(≥1 forced liquidation)', fontsize=10)
            ax.set_ylabel('E[log growth] ann.', fontsize=10)
            ax.legend(fontsize=9, frameon=False, title='Haircut')
            ax.set_title(
                f'Frontier Sensitivity to Haircut Rate\n'
                f'Scenario: {_SC_NAMES.get(sc, sc)}   '
                f's={int(round(s*10000))}bps   gate={args.gate_mode}',
                fontsize=10,
            )
            plt.tight_layout()
            _gate_label = gate_tag_suffix if gate_tag_suffix else '_strict'
            out = plot_dir / f'sensitivity_{sc}{_pm_tag}{_gate_label}_s{int(round(s*10000))}.png'
            fig.savefig(out, dpi=150)
            plt.close(fig)

    n_sens = len(SCENARIOS) * len(spreads)
    print(f"  ✓ {n_sens} sensitivity plots saved")
    print(f"✓ All frontier plots → {plot_dir.relative_to(BASE_DIR)}/")
