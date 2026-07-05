"""
panel_f_crra_improvement.py
════════════════════════════════════════════════════════════════════════
Computes Panel F: CRRA utility improvement of the economic-gate optimum
over the regulatory benchmark (strict gate at w* = 20%).

    delta_u = (U_opt − U_reg) / |U_reg| × 100   [%]

Usage:
    python panel_f_crra_improvement.py              # defaults to g=10
    python panel_f_crra_improvement.py --gate_pct 20
    python panel_f_crra_improvement.py --gate_pct 50

Reads  : optimization_runs/eltif_optimization_frontier_bond_hy_g{g}_h{h}_s{s}_economic.csv
         optimization_runs/eltif_optimization_frontier_bond_hy_h{h}_s{s}_strict.csv
Appends: optimization_runs/grid_summary_bond_hy_g{g}_economic.txt
         optimization_runs/grid_table_bond_hy_g{g}_economic.tex
"""

import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gate_pct", type=int, default=10,
                    help="Redemption gate as %% of TNA (10, 20, or 50).")
args = parser.parse_args()
GATE_PCT = args.gate_pct

BASE    = Path(__file__).parent
OPT_DIR = BASE / "optimization_runs"

HAIRCUTS   = [5, 10, 20, 30]
SPREADS    = [100, 200, 300, 500]      # bps integers
SCENARIOS  = ["ample", "tight", "reg_max"]
REG_W      = 0.20                      # regulatory benchmark cash weight
GAMMA_DISP = 2.0                       # display γ
GAMMA_COL  = "crra_g2.0"              # column name in frontier CSVs

_SC_ABBR   = {"ample": "amp", "tight": "tgt", "reg_max": "reg"}
_SC_ORDER  = ["ample", "tight", "reg_max"]

# ── helpers ──────────────────────────────────────────────────────────────────

def econ_path(h: int, s: int) -> Path:
    return OPT_DIR / f"eltif_optimization_frontier_bond_hy_g{GATE_PCT}_h{h}_s{s}_economic.csv"

def strict_path(h: int, s: int) -> Path:
    return OPT_DIR / f"eltif_optimization_frontier_bond_hy_h{h}_s{s}_strict.csv"

def _latex_escape(s: str) -> str:
    return s.replace("%", r"\%")

# ── compute panel F values ────────────────────────────────────────────────────

results: dict[tuple, float] = {}  # (h, s, scenario) -> delta_u

for h in HAIRCUTS:
    for s in SPREADS:
        ep = econ_path(h, s)
        sp = strict_path(h, s)
        if not ep.exists():
            print(f"WARNING: economic file missing — {ep.name}")
            continue
        if not sp.exists():
            print(f"WARNING: strict file missing — {sp.name}")
            continue

        econ   = pd.read_csv(ep)
        strict = pd.read_csv(sp)

        for scen in SCENARIOS:
            # A) U_opt: max crra_g2.0 over all cash weights at economic gate
            eco_sc = econ[econ["scenario"] == scen]
            if eco_sc.empty or GAMMA_COL not in eco_sc.columns:
                print(f"WARNING: no economic rows for scenario={scen} h={h} s={s}")
                continue
            u_opt = eco_sc[GAMMA_COL].max()

            # B) U_reg: crra_g2.0 at cash_weight == 0.20, strict gate
            str_sc = strict[(strict["scenario"] == scen) &
                            (strict["cash_weight"].round(4) == round(REG_W, 4))]
            if str_sc.empty:
                print(f"WARNING: no strict row at w=0.20 for scenario={scen} h={h} s={s}")
                continue
            u_reg = float(str_sc.iloc[0][GAMMA_COL])

            delta_u = (u_opt - u_reg) / abs(u_reg) * 100
            results[(h, s, scen)] = round(delta_u, 2)

# ── text heatmap (same format as Panels A–E in grid_summary.txt) ─────────────

spread_labels  = [f"{s}bps"  for s in SPREADS]
haircut_labels = [f"{h}%"    for h in HAIRCUTS]
W = 9  # cell width (matches W for 3 scenarios in optimization_grid_analysis.py)

def make_header() -> str:
    hdr = f"{'Haircut':>10}  "
    for sc in SCENARIOS:
        for sl in spread_labels:
            sc_abbr = _SC_ABBR.get(sc, sc[:3])
            hdr += f"{sc_abbr}/{sl}".center(W)
    return hdr

panel_title = (
    "Panel F: CRRA utility improvement of economic-gate optimum "
    "over regulatory benchmark  (Δu, %)"
)

lines = []
lines.append("")
lines.append(f"{'─' * 4}  {panel_title}  (γ={GAMMA_DISP})  {'─' * 4}")
hdr = make_header()
lines.append(hdr)
lines.append("─" * len(hdr))

for h, hl in zip(HAIRCUTS, haircut_labels):
    row = f"{hl:>10}  "
    for sc in SCENARIOS:
        for s in SPREADS:
            val = results.get((h, s, sc))
            if val is None:
                row += "—".center(W)
            else:
                row += f"{val:+.2f}%".center(W)
    lines.append(row)

lines.append("─" * len(hdr))
lines.append("")
lines.append(
    f"Note: Δu = (U_opt − U_reg) / |U_reg| × 100.  "
    f"U_opt = max CRRA utility at economic-gate optimum w*.  "
    f"U_reg = CRRA utility at strict-gate w=20% regulatory benchmark.  "
    f"γ={GAMMA_DISP}.  Positive = economic gate dominates."
)

text_panel = "\n".join(lines)
print(text_panel)

# Append to grid_summary txt
txt_path = OPT_DIR / f"grid_summary_bond_hy_g{GATE_PCT}_economic.txt"
if txt_path.exists():
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write("\n" + text_panel + "\n")
    print(f"\n✓ Panel F appended to: {txt_path.relative_to(BASE)}")
else:
    print(f"\nWARNING: {txt_path.name} not found — Panel F not appended to txt.")

# ── LaTeX table ───────────────────────────────────────────────────────────────

_SC_LABEL = {
    "ample":   "Ample (20\\% credit)",
    "tight":   "Tight (5\\% credit)",
    "reg_max": "Reg.~Max (50\\% credit)",
}

n_spreads = len(SPREADS)
col_parts = ["l"] + ["r" * n_spreads for _ in SCENARIOS]
col_spec  = "|".join(col_parts)

sc_headers = []
for i, sc in enumerate(SCENARIOS):
    last   = (i == len(SCENARIOS) - 1)
    mc_fmt = r"\multicolumn{" + str(n_spreads) + r"}{" + ("c" if last else "c|") + r"}"
    sc_headers.append(mc_fmt + "{" + _SC_LABEL[sc] + "}")
sc_header_row = " & " + " & ".join(sc_headers) + r" \\"

spread_sub = " & ".join(str(s) for s in SPREADS)
spread_row = r"Haircut & " + " & ".join([spread_sub] * len(SCENARIOS)) + r" \\"

data_rows = []
for h, hl in zip(HAIRCUTS, haircut_labels):
    cells = [_latex_escape(hl)]
    for sc in SCENARIOS:
        for s in SPREADS:
            val = results.get((h, s, sc))
            cells.append("—" if val is None else _latex_escape(f"{val:+.2f}%"))
    data_rows.append(" & ".join(cells) + r" \\")

latex_lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\footnotesize",
    r"\caption{Panel F: CRRA Utility Improvement of Economic-Gate Optimum over "
    r"Regulatory Benchmark ($\Delta u$, \%, $\gamma=2.0$, economic gate $g="
    + str(GATE_PCT) + r"\%$ of TNA)}",
    r"\label{tab:grid_panelF_bond_hy_g" + str(GATE_PCT) + r"_economic}",
    r"\begin{tabular}{" + col_spec + r"}",
    r"\toprule",
    sc_header_row,
    spread_row,
    r"\midrule",
] + data_rows + [
    r"\bottomrule",
    r"\end{tabular}",
    r"\smallskip",
    r"{\footnotesize\textit{Notes:} "
    r"$\Delta u = (U_{\text{opt}} - U_{\text{reg}}) / |U_{\text{reg}}| \times 100$. "
    r"$U_{\text{opt}}$: CRRA utility at the economic-gate optimal weight $w^*$ "
    r"(maximising $E[U_\gamma(W_T)]$ over $w \in [5\%, 50\%]$, gate $= "
    + str(GATE_PCT) + r"\%$ of TNA). "
    r"$U_{\text{reg}}$: CRRA utility at the strict-gate regulatory benchmark $w = 20\%$. "
    r"$\gamma = 2.0$. Positive values indicate the economic-gate optimum dominates.}",
    r"\end{table}",
]
latex_block = "\n".join(latex_lines)

# Append to grid_table tex
tex_path = OPT_DIR / f"grid_table_bond_hy_g{GATE_PCT}_economic.tex"
if tex_path.exists():
    with open(tex_path, "a", encoding="utf-8") as f:
        f.write("\n\n" + latex_block + "\n")
    print(f"✓ Panel F LaTeX appended to: {tex_path.relative_to(BASE)}")
else:
    print(f"WARNING: {tex_path.name} not found — printing LaTeX to stdout instead.")
    print("\n" + "=" * 72)
    print(latex_block)

# ── Quick sanity check ────────────────────────────────────────────────────────
vals = list(results.values())
if vals:
    n_zero = sum(v == 0.0 for v in vals)
    print(f"\nSanity check: Δu range = [{min(vals):+.2f}%, {max(vals):+.2f}%]  "
          f"({len(vals)} cells)  all non-negative = {all(v >= 0 for v in vals)}  "
          f"zero cells = {n_zero} (economic opt coincides with w=20% benchmark)")
