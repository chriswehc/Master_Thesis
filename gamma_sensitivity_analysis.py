"""
gamma_sensitivity_analysis.py
------------------------------
Generates the gamma sensitivity appendix figure and table for the thesis.

Reads : optimization_runs/grid_summary_bond_hy_g10_economic.csv
Writes: gamma_sensitivity_figure.png   (appendix figure)
Prints: LaTeX table snippet for appendix (copy into thesis)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "optimization_runs" / "grid_summary_bond_hy_g10_economic.csv"

df = pd.read_csv(DATA)

BENCHMARK_H  = 10   # haircut_pct for table
BENCHMARK_S  = 300  # spread_bps for table and figure
SCENARIOS    = ["ample", "tight", "reg_max"]
SCENARIO_LABELS = {
    "ample":   "Ample credit",
    "tight":   "Tight credit",
    "reg_max": "Regulatory cap",
}
HAIRCUTS = [5, 10, 20, 30]
HAIRCUT_COLORS = {
    5:  "#2e75b6",
    10: "#70ad47",
    20: "#ff9800",
    30: "#e53935",
}

# ---------------------------------------------------------------------------
# Figure: w*(gamma) lines for each scenario, one subplot per scenario
# ---------------------------------------------------------------------------

s300 = df[df["spread_bps"] == BENCHMARK_S].copy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, scen in zip(axes, SCENARIOS):
    sub = s300[s300["scenario"] == scen].sort_values(["haircut_pct", "gamma"])
    for h in HAIRCUTS:
        h_data = sub[sub["haircut_pct"] == h]
        ax.plot(
            h_data["gamma"],
            h_data["optimal_cash_weight"] * 100,
            color=HAIRCUT_COLORS[h],
            linewidth=1.8,
            label=f"h = {h}%",
        )
    ax.set_title(SCENARIO_LABELS[scen], fontsize=12, pad=8)
    ax.set_xlabel("CRRA risk aversion $\\gamma$", fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel("Optimal cash weight $w^*$ (%)", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_xlim(1.0, 5.0)
    ax.set_ylim(0, 57)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=7.5, framealpha=0.85, loc="upper left")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

fig.suptitle(
    "Optimal cash buffer $w^*(\\gamma)$ by scenario and haircut level"
    "  ($s = 300$ bps, economic gate, $g = 10\\%$)",
    fontsize=11, y=1.01,
)

plt.tight_layout()
out_fig = BASE / "gamma_sensitivity_figure.png"
fig.savefig(out_fig, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved to: {out_fig}")

# ---------------------------------------------------------------------------
# LaTeX table: benchmark cell (h=10%, s=300bps), gamma in {1.0,2.0,3.0,5.0}
# ---------------------------------------------------------------------------

GAMMA_SUBSET = [1.0, 2.0, 3.0, 5.0]

bench = df[(df["haircut_pct"] == BENCHMARK_H) & (df["spread_bps"] == BENCHMARK_S)]
tbl = (
    bench[bench["gamma"].isin(GAMMA_SUBSET)]
    .pivot(index="gamma", columns="scenario", values="optimal_cash_weight")
    [SCENARIOS]
    .mul(100)
)

def latex_gamma_table(df_t: pd.DataFrame) -> str:
    col_names = [SCENARIO_LABELS[s] for s in SCENARIOS]
    col_spec = "c" + "r" * len(col_names)
    header = " & ".join(
        ["\\textbf{$\\gamma$}"] + [f"\\textbf{{{c}}}" for c in col_names]
    )
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Optimal Cash Buffer $w^*(\\gamma)$ at Benchmark Cell "
        "($h = 10\\%$, $s = 300$\\,bps)}",
        "  \\label{tab:gamma_sensitivity}",
        "  \\begin{tabular}{" + col_spec + "}",
        "    \\toprule",
        f"    {header} \\\\",
        "    \\midrule",
    ]
    for gamma, row in df_t.iterrows():
        vals = " & ".join(
            [f"{float(gamma):.1f}"] + [f"{v:.0f}\\%" for v in row.values]
        )
        lines.append(f"    {vals} \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\smallskip",
        "  {\\footnotesize\\textit{Notes:} Optimal cash weight $w^*$ maximising CRRA utility "
        "(Brunnermeier--Sannikov--Vayanos 2009) at each $\\gamma$ for three credit-capacity "
        "scenarios. Benchmark cell: haircut $h = 10\\%$, credit spread $s = 300$\\,bps, "
        "economic gate $g = 10\\%$. \\textit{Ample}: credit line rarely binding; "
        "\\textit{Tight}: credit line frequently binding; \\textit{Regulatory cap}: "
        "cash weight capped at regulatory maximum.}",
        "\\end{table}",
    ]
    return "\n".join(lines)

latex_str = latex_gamma_table(tbl)

print("\n" + "=" * 72)
print("LaTeX table for thesis appendix (copy below):")
print("=" * 72)
print(latex_str)

# ---------------------------------------------------------------------------
# Console summary: inline sentence numbers
# ---------------------------------------------------------------------------

ample_g1 = tbl.loc[1.0, "ample"]
ample_g5 = tbl.loc[5.0, "ample"]
print(f"\nInline sentence numbers (ample credit, h=10%, s=300bps):")
print(f"  w*(gamma=1.0) = {ample_g1:.0f}%")
print(f"  w*(gamma=5.0) = {ample_g5:.0f}%")
print(
    f'\nSuggested inline sentence:\n'
    f'"The monotonic relationship between risk aversion and optimal cash weight is '
    f'robust across the full gamma grid: at the benchmark cell (h = 10%, s = 300 bps) '
    f'under ample credit, w* rises from {ample_g1:.0f}% at gamma = 1.0 to '
    f'{ample_g5:.0f}% at gamma = 5.0 (Appendix Figure~\\ref{{fig:gamma_sensitivity}})."'
)
