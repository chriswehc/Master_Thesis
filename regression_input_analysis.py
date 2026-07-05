#!/usr/bin/env python3
"""
regression_input_analysis.py
─────────────────────────────
Batch-runs the Goldstein regression pipeline for every asset-class subset
defined in RUNS, then prints a publication-style comparison table
(coefficient, t-stat, significance stars) across all runs.

Usage:
    python regression_input_analysis.py

Outputs (in Regression_Results/):
    comparison_table.txt   — human-readable academic table
    comparison_table.csv   — machine-readable (one row per param × run)
"""

import subprocess
import sys
import pickle
from pathlib import Path

import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION 
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = Path(__file__).parent               # MT_Python/
LSEG_DIR   = BASE_DIR / "LSEG_merged"
RUNS_DIR   = LSEG_DIR / "runs"                  # tagged pkls land here
OUT_DIR    = BASE_DIR / "Regression_Results"    # comparison table output
REG_SCRIPT = LSEG_DIR / "Fund_Flows_Regression.py"

# (keyword passed to --keyword, run_tag passed to --run_tag)
# keyword="" means all funds; run_tag "_all" saves to LSEG_merged/ root (not runs/)
# Keywords are regex patterns matched case-insensitively against IssueLipperGlobalSchemeName.
# Broad classes mirror summary.py BROAD_ORDER / UNI_SUBGROUPS exactly.
RUNS = [
    # ── All funds ────────────────────────────────────────────────────────────
    ("",                                    "_all"),
    # ── Aggregates ───────────────────────────────────────────────────────────
    ("^Equity",                             "_equity"),
    ("^Bond|^Absolute Return Bond",         "_bond"),
    ("High Yield",                          "_bond_hy"),
    ("Corporates",                          "_bond_corp"),
    # ── Broad asset classes (= BROAD_ORDER in summary.py) ───────────────────
    ("^Equity US",                          "_equity_us"),
    ("^Equity(?! US)",                      "_equity_intl"),
    ("^Bond USD|^Absolute Return Bond USD", "_bond_usd"),
    ("^Bond(?! USD)",                       "_bond_global"),
    ("^Mixed Asset",                        "_mixed"),
    ("^Alternative",                        "_alternative"),
    # ── Sub-groups (= UNI_SUBGROUPS in summary.py) ───────────────────────────
    ("^Bond USD High Yield$",               "_bond_usd_hy"),
    ("^Bond USD Corporates$",               "_bond_usd_corp"),
]

# Human-readable column labels for the comparison table
RUN_LABELS = {
    "_all":            "All Funds",
    "_equity":         "All Equity",
    "_bond":           "All Bond",
    "_bond_hy":        "All Bond HY",
    "_bond_corp":      "All Bond Corp",
    "_equity_us":      "Equity US",
    "_equity_intl":    "Equity Intl.",
    "_bond_usd":       "Bond USD",
    "_bond_global":    "Bond Global",
    "_mixed":          "Mixed Asset",
    "_alternative":    "Alternative",
    "_bond_usd_hy":    "Bond USD HY",
    "_bond_usd_corp":  "Bond USD Corp",
}


# ══════════════════════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════════════════════

def run_regression(keyword: str, run_tag: str) -> int:
    """Invoke Fund_Flows_Regression.py via subprocess with --keyword and --run_tag."""
    cmd = [sys.executable, str(REG_SCRIPT), "--run_tag", run_tag]
    if keyword.strip():
        cmd += ["--keyword", keyword.strip()]
    if args.tna_min != 10_000:
        cmd += ["--tna_min", str(args.tna_min)]
    if args.alpha_lag != 0:
        cmd += ["--alpha_lag", str(args.alpha_lag)]
    result = subprocess.run(cmd, cwd=str(LSEG_DIR))
    return result.returncode


# ══════════════════════════════════════════════════════════════════════════════
# LOAD STATS FROM ENRICHED PKL
# ══════════════════════════════════════════════════════════════════════════════

def load_stats(run_tag: str) -> tuple[dict, dict]:
    """
    Return (inference_dict, model_stats_dict) from the enriched pkl.
    inference_dict:  {param_name: {coef, se, t, p, ci_lower, ci_upper}}
    model_stats_dict: {r_squared, adj_r_squared, n_obs, f_statistic, ...}

    For run_tag "_all" the pkl lives in LSEG_merged/ (untagged default).
    All other tags are in LSEG_merged/runs/.
    """
    if run_tag == "_all":
        pkl_path = LSEG_DIR / "goldstein_model2_coefficients.pkl"
    else:
        pkl_path = RUNS_DIR / f"goldstein_model2_coefficients{run_tag}.pkl"

    if not pkl_path.exists():
        print(f"  ⚠  pkl not found: {pkl_path.relative_to(BASE_DIR)}")
        return {}, {}

    with open(pkl_path, "rb") as f:
        coefs = pickle.load(f)

    return coefs.get("inference", {}), coefs.get("model_stats", {})


# ══════════════════════════════════════════════════════════════════════════════
# ACADEMIC TABLE
# ══════════════════════════════════════════════════════════════════════════════

# Row definitions: ("panel", title) or ("var", statsmodels_param_name, display_label)
# Param names match Model 5 (Goldstein differential + Fund FEs) Patsy output.
# Reference regime = Downturn (alphabetically first).
TABLE_ROWS = [
    ("panel", "Panel A: Goldstein Baseline Parameters (Downturn reference)"),
    ("var", "Alpha",            "α baseline (Downturn)"),
    ("var", "Alpha_x_Negative", "α⁻ additional (Downturn)"),
    ("var", "Alpha_Negative",   "α<0 indicator"),

    ("panel", "Panel B: Regime Level Shifts vs Downturn"),
    ("var", "C(macro_regime)[T.Goldilocks]",  "Δflow Goldilocks"),
    ("var", "C(macro_regime)[T.Overheating]", "Δflow Overheating"),
    ("var", "C(macro_regime)[T.Stagflation]", "Δflow Stagflation"),

    ("panel", "Panel C: Differential α Slopes vs Downturn"),
    ("var", "Alpha:C(macro_regime)[T.Goldilocks]",             "Δα Goldilocks"),
    ("var", "Alpha:C(macro_regime)[T.Overheating]",            "Δα Overheating"),
    ("var", "Alpha:C(macro_regime)[T.Stagflation]",            "Δα Stagflation"),
    ("var", "Alpha_x_Negative:C(macro_regime)[T.Goldilocks]",  "Δα⁻ Goldilocks"),
    ("var", "Alpha_x_Negative:C(macro_regime)[T.Overheating]", "Δα⁻ Overheating"),
    ("var", "Alpha_x_Negative:C(macro_regime)[T.Stagflation]", "Δα⁻ Stagflation"),

    ("panel", "Panel D: Controls"),
    ("var", "Lagged_Flow", "Lagged Flow"),
    ("var", "Log_TNA_lag", "log(TNA)"),
    ("var", "Log_Age",     "log(Age)"),
    ("var", "TER",         "TER"),
]


def _stars(p) -> str:
    if p is None:
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def print_academic_table(
    runs:       list,
    all_stats:  dict,   # {run_tag: inference_dict}
    model_meta: dict,   # {run_tag: model_stats_dict}
    save_path:  Path | None = None,
) -> None:
    """
    Prints (and optionally saves) a publication-style coefficient table.
    Columns = run tags; rows = regression coefficients + t-statistics.
    Reads directly from inference dicts (coef, t, p keys).
    """
    COL   = 16   # width per data column
    LABEL = 30   # width of the label column

    tags  = [tag for _, tag in runs if tag in all_stats]
    width = LABEL + COL * len(tags)

    def _col_label(tag):
        return RUN_LABELS.get(tag, tag)

    lines = []
    emit  = lines.append

    def sep(c="─"):  emit(c * width)
    def dsep():      emit("═" * width)

    emit("")
    dsep()
    emit("Flow-Performance Sensitivity by Macro Regime".center(width))
    emit("Dependent variable: Quarterly Fund Flow Rate".center(width))
    emit("(OLS, Fund FE, Clustered SE at Fund Level)".center(width))
    dsep()
    emit(f"{'Variable':<{LABEL}}" + "".join(_col_label(t).center(COL) for t in tags))
    sep()

    for row in TABLE_ROWS:
        if row[0] == "panel":
            emit("")
            emit(row[1])
            sep()
            continue

        _, param, label = row
        coef_line  = f"{label:<{LABEL}}"
        tstat_line = f"{'':<{LABEL}}"

        for tag in tags:
            s = all_stats.get(tag, {}).get(param)
            if s is None or s.get("coef") is None:
                coef_line  += "—".center(COL)
                tstat_line += " " * COL
            else:
                coef_line  += f"{s['coef']:.3f}{_stars(s.get('p'))}".rjust(COL)
                t_str = f"({s['t']:.2f})" if s.get("t") is not None else ""
                tstat_line += t_str.rjust(COL)

        emit(coef_line)
        emit(tstat_line)
        emit("")

    sep()

    def footer_row(label, values):
        emit(f"{label:<{LABEL}}" + "".join(str(v).rjust(COL) for v in values))

    footer_row("R²",         [f"{model_meta.get(t, {}).get('r_squared',     float('nan')):.4f}" for t in tags])
    footer_row("Adj. R²",    [f"{model_meta.get(t, {}).get('adj_r_squared', float('nan')):.4f}" for t in tags])
    footer_row("N",          [f"{int(model_meta.get(t, {}).get('n_obs', 0)):,}"                 for t in tags])
    footer_row("F-stat",     [f"{model_meta.get(t, {}).get('f_statistic',   float('nan')):.1f}" for t in tags])
    footer_row("Fund FE",    ["Yes" for _ in tags])
    footer_row("SE Cluster", ["Fund" for _ in tags])

    dsep()
    emit("Note: t-statistics in parentheses. * p<0.10, ** p<0.05, *** p<0.01.")
    emit("      Quarter fixed effects included in all specifications.")
    emit("      Standard errors clustered at the fund level.")
    emit("")

    output = "\n".join(lines)
    print(output)

    if save_path is not None:
        save_path.write_text(output, encoding="utf-8")
        print(f"✓ Table saved to: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

import argparse as _ap
_parser = _ap.ArgumentParser(description='Batch regression runner.')
_parser.add_argument('--tna_min',   type=float, default=10_000,
                     help='Minimum TNA_lag passed to Fund_Flows_Regression.py (default 10000).')
_parser.add_argument('--alpha_lag', type=int,   default=0,
                     help='Quarters to lag alpha (default 0 = contemporaneous).')
_parser.add_argument('--sim_tag',   type=str,   default='_bond_hy',
                     help='Run tag whose pkl feeds the simulation; only this run gets Model 5 '
                          '(default _bond_hy). Pass "" to run Model 5 for all tags.')
_parser.add_argument('--table_only', action='store_true',
                     help='Skip regressions; load existing pkls and regenerate the table only.')
args, _ = _parser.parse_known_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_stats:  dict = {}   # {run_tag: inference_dict}
    model_meta: dict = {}   # {run_tag: model_stats_dict}
    failed:     list = []

    for keyword, run_tag in RUNS:
        label = f"'{keyword}'" if keyword.strip() else '"" (All Funds)'

        if args.table_only:
            inf, mstats = load_stats(run_tag)
            if inf:
                all_stats[run_tag]  = inf
                model_meta[run_tag] = mstats
                print(f"  ✓ [{run_tag}] Loaded {len(inf)} inference entries from existing pkl")
            else:
                print(f"  ⚠  [{run_tag}] pkl missing or empty — skipping")
                failed.append(run_tag)
            continue

        print(f"\n{'#' * 70}")
        print(f"#  keyword={label}  run_tag={run_tag!r}")
        print(f"{'#' * 70}\n")

        rc = run_regression(keyword, run_tag)

        if rc != 0:
            print(f"\n⚠  Regression FAILED (rc={rc}). Skipping.\n")
            failed.append(run_tag)
            continue

        inf, mstats = load_stats(run_tag)
        if inf:
            all_stats[run_tag]  = inf
            model_meta[run_tag] = mstats
            print(f"  ✓ Loaded {len(inf)} inference entries for {run_tag!r}")
        else:
            print(f"  ⚠  inference dict empty for {run_tag!r} — pkl may be from an older run")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  LOOP COMPLETE  —  {len(RUNS) - len(failed)}/{len(RUNS)} runs succeeded")
    if failed:
        print(f"  Failed tags: {failed}")
    print(f"  Output folder: {OUT_DIR}")
    print(f"{'=' * 70}")

    if not all_stats:
        print("\n⚠  No successful runs — table not generated.")
        return

    # ── Academic comparison table ─────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUT_DIR / "comparison_table.txt"
    print_academic_table(RUNS, all_stats, model_meta, save_path=table_path)

    # ── Machine-readable CSV ──────────────────────────────────────────────────
    records = []
    for run_tag, inf in all_stats.items():
        kw = next((kw for kw, t in RUNS if t == run_tag), run_tag)
        for param, s in inf.items():
            records.append({
                "keyword":  kw or "all",
                "run_tag":  run_tag,
                "param":    param,
                "coef":     s.get("coef"),
                "se":       s.get("se"),
                "t":        s.get("t"),
                "p":        s.get("p"),
                "ci_lower": s.get("ci_lower"),
                "ci_upper": s.get("ci_upper"),
                "stars":    _stars(s.get("p")),
            })
    csv_path = OUT_DIR / "comparison_table.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"✓ Full coefficient CSV saved to: {csv_path}\n")


if __name__ == "__main__":
    main()
