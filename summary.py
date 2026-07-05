"""
summary.py
----------
Generates descriptive statistics tables for the thesis data chapter.

Output: summary_tables.txt  (LaTeX table)

Statistics per series (from 2000 onwards, all in EUR):
  Ann. Return   = mean(r_q) * 4
  Ann. Vol      = std(r_q, ddof=1) * sqrt(4)
  Sharpe        = (Ann. Return - mean_rf_ann) / Ann. Vol
  AC(k)         = autocorrelation at lag k, k = 1, 2, 3
  N             = number of quarterly observations

Notes:
  - Hamilton Lane indices: raw USD levels from PM_Indices.xlsx -> quarterly returns
    -> converted to EUR via FRED DEXUSEU: r_EUR = (1+r_USD)*(FX_{t-1}/FX_t) - 1
  - US benchmarks (S&P 500, US Bond Market): also USD -> converted to EUR same way
  - EUR benchmarks (STOXX 600, Euro Agg Bond): native EUR levels -> pct_change
  - Risk-free rate: EUR003 (Euribor 3M) for all series
"""

import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path
from io import StringIO

BASE = Path(__file__).parent
START = "2000-01-01"

# ---------------------------------------------------------------------------
# 1. Load EUR/USD FX rate from FRED (quarterly last)
# ---------------------------------------------------------------------------

def load_eurusd_quarterly() -> pd.Series:
    """Download DEXUSEU from FRED (USD per 1 EUR), resample to quarter-end."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU"
    with urllib.request.urlopen(url) as resp:
        raw = pd.read_csv(StringIO(resp.read().decode()),
                          parse_dates=["observation_date"],
                          index_col="observation_date")
    raw.columns = ["EURUSD"]
    raw["EURUSD"] = pd.to_numeric(raw["EURUSD"], errors="coerce")
    eurusd_q = raw["EURUSD"].resample("QE").last().dropna()
    print(f"  EUR/USD loaded: {eurusd_q.index[0].date()} – {eurusd_q.index[-1].date()}, "
          f"{len(eurusd_q)} quarters")
    return eurusd_q

def usd_to_eur(r_usd: pd.Series, eurusd_q: pd.Series) -> pd.Series:
    """
    Convert USD quarterly return series to EUR.
    r_EUR = (1 + r_USD) * (EURUSD_{t-1} / EURUSD_t) - 1
    DEXUSEU = USD per EUR; rising FX = EUR strengthens = USD asset loses in EUR terms.
    """
    fx = eurusd_q.reindex(r_usd.index, method="nearest")
    r_eur = (1 + r_usd) * (fx.shift(1) / fx) - 1
    return r_eur.dropna()

print("Loading EUR/USD FX rate …")
eurusd_q = load_eurusd_quarterly()

# ---------------------------------------------------------------------------
# 2. Load Hamilton Lane raw (USD) levels from PM_Indices.xlsx -> quarterly returns
# ---------------------------------------------------------------------------

print("Loading Hamilton Lane indices (raw USD) …")
hl_levels = pd.read_excel(BASE / "PM_Indices.xlsx",
                          sheet_name="Hamilton_Lane",
                          index_col=0, parse_dates=True)
hl_levels.index = pd.to_datetime(hl_levels.index)

hl_ret_usd = hl_levels.pct_change().dropna()
hl_ret_usd = hl_ret_usd.loc[START:]

# Convert each HL series USD -> EUR
hl_ret_eur = pd.DataFrame(index=hl_ret_usd.index)
for col in hl_ret_usd.columns:
    hl_ret_eur[col] = usd_to_eur(hl_ret_usd[col], eurusd_q)

# ---------------------------------------------------------------------------
# 3. Load EUR benchmarks (levels -> returns, native EUR)
# ---------------------------------------------------------------------------

print("Loading EUR benchmarks …")
bbg = pd.read_csv(BASE / "bloomberg_benchmarks.csv",
                  index_col="Date", parse_dates=True)
bbg_ret = bbg.pct_change().dropna()
bbg_ret = bbg_ret.loc[START:]
bbg_ret.columns = ["EURO STOXX 600 TR", "Euro Agg Bond TR"]

# ---------------------------------------------------------------------------
# 4. Load US benchmarks (USD) and convert to EUR
# ---------------------------------------------------------------------------

print("Loading US benchmarks (USD -> EUR) …")
us = pd.read_csv(BASE / "LSEG_merged/benchmarks.csv",
                 index_col="Date", parse_dates=True)
us = us.loc[START:]

sp500_eur = usd_to_eur(us["SP500_Return"], eurusd_q)
bond_eur  = usd_to_eur(us["Bond_Market_Return"], eurusd_q)

# ---------------------------------------------------------------------------
# 5. Risk-free rate: EUR003 for all series
# ---------------------------------------------------------------------------

cash = pd.read_csv(BASE / "Cash_quarterly_returns.csv",
                   index_col="Date", parse_dates=True)
rf_eur = cash.loc[START:, "EUR003_Index"]

# ---------------------------------------------------------------------------
# 6. Helper: compute stats
# ---------------------------------------------------------------------------

def compute_stats(r: pd.Series, rf: pd.Series) -> dict:
    rf_aligned = rf.reindex(r.index).dropna()
    r_clean    = r.reindex(rf_aligned.index).dropna()

    ann_ret = r_clean.mean() * 4
    ann_vol = r_clean.std(ddof=1) * np.sqrt(4)
    mean_rf = rf_aligned.mean() * 4
    sharpe  = (ann_ret - mean_rf) / ann_vol

    return {
        "Ann. Return (%)": ann_ret * 100,
        "Ann. Vol (%)":    ann_vol * 100,
        "Sharpe Ratio":    sharpe,
        "AC(1)":           r_clean.autocorr(lag=1),
        "AC(2)":           r_clean.autocorr(lag=2),
        "AC(3)":           r_clean.autocorr(lag=3),
        "N":               len(r_clean),
    }

# ---------------------------------------------------------------------------
# 7. Build rows
# ---------------------------------------------------------------------------

rows = {}

for col in hl_ret_eur.columns:
    rows[col] = compute_stats(hl_ret_eur[col], rf_eur)

for col in bbg_ret.columns:
    rows[col] = compute_stats(bbg_ret[col], rf_eur)

rows["S\\&P 500 (EUR)"]         = compute_stats(sp500_eur, rf_eur)
rows["US Bond Market (EUR)"]    = compute_stats(bond_eur,  rf_eur)

df = pd.DataFrame(rows).T
df.index.name = "Series"

pd.set_option("display.float_format", "{:.2f}".format)
print("\n=== Descriptive Statistics ===\n")
print(df.to_string())

# ---------------------------------------------------------------------------
# 8. Build LaTeX table
# ---------------------------------------------------------------------------

pm_idx  = list(hl_ret_eur.columns)
eur_idx = list(bbg_ret.columns)
us_idx  = ["S\\&P 500 (EUR)", "US Bond Market (EUR)"]

def fmt_row(row):
    return {
        "Ann. Return": f"{row['Ann. Return (%)']:.2f}\\%",
        "Ann. Vol":    f"{row['Ann. Vol (%)']:.2f}\\%",
        "Sharpe":      f"{row['Sharpe Ratio']:.2f}",
        "AC(1)":       f"{row['AC(1)']:.2f}",
        "AC(2)":       f"{row['AC(2)']:.2f}",
        "AC(3)":       f"{row['AC(3)']:.2f}",
        "N":           str(int(row["N"])),
    }

df_fmt = df.apply(fmt_row, axis=1, result_type="expand")
df_fmt.index.name = "Series"

def build_latex(df_fmt, pm_idx, eur_idx, us_idx) -> str:
    cols = ["Ann. Return", "Ann. Vol", "Sharpe", "AC(1)", "AC(2)", "AC(3)", "N"]
    ncols = len(cols) + 1
    col_spec = "l" + "r" * len(cols)
    header = " & ".join(
        ["\\textbf{Series}"] + [f"\\textbf{{{c}}}" for c in cols]
    )

    def section(label):
        return [
            f"    \\midrule",
            f"    \\multicolumn{{{ncols}}}{{l}}{{\\textit{{{label}}}}} \\\\",
            f"    \\midrule",
        ]

    def add_rows(idx):
        out = []
        for name in idx:
            vals = " & ".join([name] + [df_fmt.loc[name, c] for c in cols])
            out.append(f"    {vals} \\\\")
        return out

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Descriptive Statistics of Private Market Indices and Benchmarks}",
        "  \\label{tab:descriptive_stats}",
        "  \\resizebox{\\textwidth}{!}{%",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    {header} \\\\",
        "    \\midrule",
    ]

    lines += section("Private Markets (Hamilton Lane, USD$\\to$EUR, raw returns)")
    lines += add_rows(pm_idx)
    lines += section("EUR Benchmarks")
    lines += add_rows(eur_idx)
    lines += section("US Benchmarks (USD$\\to$EUR)")
    lines += add_rows(us_idx)

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}%",
        "  }",
        "  {\\footnotesize\\textit{Notes:} Quarterly data from 2000 onwards. "
        "All returns are denominated in EUR. "
        "USD-denominated series (Hamilton Lane indices, S\\&P~500, US Bond Market) are converted "
        "using end-of-quarter EUR/USD spot rates from FRED (DEXUSEU): "
        "$r^{\\text{EUR}}_t = (1+r^{\\text{USD}}_t)\\cdot(S_{t-1}/S_t)-1$. "
        "Ann.~Return and Ann.~Vol are annualised ($\\times 4$ and $\\times\\sqrt{4}$). "
        "The Sharpe ratio uses Euribor 3M (EUR003) as risk-free rate for all series. "
        "AC($k$) denotes the $k$th-order autocorrelation of quarterly returns. "
        "$N$ is the number of quarterly observations.}",
        "\\end{table}",
    ]

    return "\n".join(lines)

latex_str = build_latex(df_fmt, pm_idx, eur_idx, us_idx)

# ---------------------------------------------------------------------------
# 9. Write to summary_tables.txt
# ---------------------------------------------------------------------------

out_path = BASE / "summary_tables.txt"
with open(out_path, "w") as f:
    f.write("% ================================================================\n")
    f.write("% THESIS SUMMARY TABLES\n")
    f.write("% Generated by summary.py\n")
    f.write("% ================================================================\n\n")
    f.write("% TABLE 1: Descriptive Statistics\n\n")
    f.write(latex_str)
    f.write("\n")

print(f"\nLaTeX table written to: {out_path}")

# ---------------------------------------------------------------------------
# 10. Lipper fund universe summary tables
# ---------------------------------------------------------------------------

funds_meta = pd.read_csv(BASE / "LSEG_merged/Clean_funds_with_asset_class.csv")
fund_flows  = pd.read_csv(BASE / "LSEG_merged/fund_flows_with_macro.csv",
                          parse_dates=["Date"])
# Full universe (all asset classes) for Tables 3 & 4
fund_flows_all = pd.read_csv(BASE / "LSEG_merged/runs/fund_flows_with_macro_all.csv",
                              parse_dates=["Date"])

# --- Broad asset class mapping -------------------------------------------
def broad_class(name: str) -> str:
    if pd.isna(name):
        return "Other"
    n = str(name)
    if n.startswith("Equity US") or n in ("Equity US Income",):
        return "Equity US"
    if n.startswith("Equity"):
        return "Equity International"
    if n.startswith("Bond USD") or n.startswith("Absolute Return Bond USD"):
        return "Bond USD"
    if n.startswith("Bond"):
        return "Bond Global"
    if n.startswith("Mixed Asset"):
        return "Mixed Asset"
    if n.startswith("Money Market"):
        return "Money Market"
    if n.startswith("Alternative"):
        return "Alternative"
    return "Other"

funds_meta["lipper_name"] = funds_meta["IssueLipperGlobalSchemeName"].str.lstrip("*")
funds_meta["broad_class"] = funds_meta["lipper_name"].apply(broad_class)

# Sub-groups to highlight within broad classes (full universe)
UNI_SUBGROUPS = {
    "Bond USD":    ["Bond USD High Yield", "Bond USD Corporates"],
    "Bond Global": ["Bond Global High Yield USD"],
}

BROAD_ORDER = ["Equity US", "Equity International", "Bond USD", "Bond Global",
               "Mixed Asset", "Money Market", "Alternative"]

def count_status(g):
    return {
        "Fund Classes": len(g),
        "Active":       (g["FundClassStatusName"] == "Active").sum(),
        "Liquidated":   (g["FundClassStatusName"] == "Liquidated").sum(),
        "Merged":       (g["FundClassStatusName"] == "Merged").sum(),
    }

# --- TABLE 2: Full Lipper universe by broad asset class ------------------
uni_rows = {}
for cls in BROAD_ORDER:
    g = funds_meta[funds_meta["broad_class"] == cls]
    uni_rows[cls] = count_status(g)
    for sub in UNI_SUBGROUPS.get(cls, []):
        g_sub = funds_meta[funds_meta["lipper_name"] == sub]
        uni_rows[f"  {sub}"] = count_status(g_sub)

uni_rows["Total"] = count_status(funds_meta)
df_uni = pd.DataFrame(uni_rows).T.astype(int)
df_uni.index.name = "Asset Class"

print("\n=== Lipper Universe ===")
print(df_uni.to_string())

# --- Shared: merge asset class + TER into panel --------------------------
panel_ids  = fund_flows["Instrument"].unique()
panel_meta = funds_meta[funds_meta["RIC_clean"].isin(panel_ids)].copy()

# --- Full-universe fund chars (for Table 3) --------------------------------
ter_df = pd.read_csv(BASE / "LSEG_merged/ter_merged.csv", index_col=0)
ter_df.index.name = "Instrument"
ter_df = ter_df[["Total Expense Ratio"]].rename(columns={"Total Expense Ratio": "TER"})
ter_df["TER"] = pd.to_numeric(ter_df["TER"], errors="coerce")

uni_chars = (
    funds_meta[["RIC_clean", "lipper_name", "broad_class", "FundClassStatusName"]]
    .merge(ter_df, left_on="RIC_clean", right_index=True, how="left")
    .rename(columns={"RIC_clean": "Instrument",
                     "lipper_name": "LipperName",
                     "FundClassStatusName": "Status"})
)

# --- Add TER to df_uni ---------------------------------------------------
ter_by_class: dict = {}
for cls in BROAD_ORDER:
    ter_by_class[cls] = uni_chars[uni_chars["broad_class"] == cls]["TER"].mean()
    for sub in UNI_SUBGROUPS.get(cls, []):
        ter_by_class[f"  {sub}"] = uni_chars[uni_chars["LipperName"] == sub]["TER"].mean()
ter_by_class["Total"] = uni_chars["TER"].mean()
df_uni["Avg TER (%)"] = pd.Series(ter_by_class)

# --- All-fund panel chars (for Table 3) ----------------------------------
all_ids    = fund_flows_all["Instrument"].unique()
all_meta   = funds_meta[funds_meta["RIC_clean"].isin(all_ids)].copy()
all_chars  = (
    all_meta[["RIC_clean", "lipper_name", "broad_class"]]
    .rename(columns={"RIC_clean": "Instrument", "lipper_name": "LipperName"})
)
panel_stats = fund_flows_all.merge(
    all_chars[["Instrument", "LipperName", "broad_class"]], on="Instrument", how="left")

# --- TABLE 3: return/flow stats (panel only) -----------------------------
def stats_row(g):
    if len(g) == 0:
        return pd.Series({"Ann. Return (\\%)": None, "Return Vol (\\%)": None,
                          "Mean Flow (\\%)": None, "Flow Vol (\\%)": None,
                          "Mean Alpha (\\%)": None})
    return pd.Series({
        "Ann. Return (\\%)": np.log1p(g["Return"]).mean() * 4 * 100,
        "Return Vol (\\%)":  g["Return"].std(ddof=1) * np.sqrt(4) * 100,
        "Mean Flow (\\%)":   g["Flow_Rate"].mean() * 100,
        "Flow Vol (\\%)":    g["Flow_Rate"].std(ddof=1) * 100,
        "Mean Alpha (\\%)":  g["Alpha"].mean() * 4 * 100,
    })

stats_rows = {}
stats_rows["Overall"] = stats_row(fund_flows_all)

for cls in BROAD_ORDER:
    ps_cls = panel_stats[panel_stats["broad_class"] == cls]
    stats_rows[cls] = stats_row(ps_cls)
    for sub in UNI_SUBGROUPS.get(cls, []):
        ps_sub = panel_stats[panel_stats["LipperName"] == sub]
        stats_rows[f"  {sub}"] = stats_row(ps_sub)

df_stats = pd.DataFrame(stats_rows).T
df_stats.index.name = "Asset Class"

print("\n=== Return & Flow Statistics ===")
print(df_stats.to_string())

# --- LaTeX builder for universe table ------------------------------------
def latex_universe(df: pd.DataFrame) -> str:
    int_cols = ["Fund Classes", "Active", "Liquidated", "Merged"]
    cols = int_cols + ["Avg TER (%)"]
    col_spec = "l" + "r" * len(cols)
    def tex_col(c): return c.replace("%", "\\%")
    header = " & ".join(["\\textbf{Asset Class}"] + [f"\\textbf{{{tex_col(c)}}}" for c in cols])

    def is_sub(idx): return str(idx).startswith("  ")
    def label(idx):
        return f"\\quad {str(idx).strip()}" if is_sub(idx) else str(idx)
    def fmt_row(idx, row, bold=False):
        def b(s): return f"\\textbf{{{s}}}" if bold else s
        int_vals = [b(f"{int(row[c]):,}") for c in int_cols]
        ter = row["Avg TER (%)"]
        ter_str = b(f"{ter:.2f}\\%") if pd.notna(ter) else b("--")
        return " & ".join([b(label(idx))] + int_vals + [ter_str])

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Lipper Fund Universe by Broad Asset Class}",
        "  \\label{tab:lipper_universe}",
        "  \\resizebox{\\textwidth}{!}{%",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    {header} \\\\",
        "    \\midrule",
    ]
    body = df.drop("Total")
    body_idx = list(body.index)
    for i, (idx, row) in enumerate(body.iterrows()):
        lines.append(f"    {fmt_row(idx, row)} \\\\")
        if not is_sub(idx) and i < len(body_idx) - 1 and not is_sub(body_idx[i + 1]):
            lines.append("    \\addlinespace")
    lines.append("    \\midrule")
    lines.append(f"    {fmt_row('Total', df.loc['Total'], bold=True)} \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}%",
        "  }",
        "  {\\footnotesize\\textit{Notes:} Fund classes from Lipper (LSEG) global database. "
        "\\textit{Liquidated} includes wound-up share classes; \\textit{Merged} indicates "
        "share classes absorbed into another fund. \\textit{Avg TER} is the mean total expense "
        "ratio as reported by Lipper; -- where not available. "
        "Broad asset classes follow the Lipper Global Classification scheme; "
        "indented rows show selected sub-categories.}",
        "\\end{table}",
    ]
    return "\n".join(lines)

latex_uni = latex_universe(df_uni)

# --- LaTeX builder: Table 3 (availability & TER) -------------------------
# --- LaTeX builder: Table 3 (return, vol, flow, flow vol, alpha) ---------
def latex_stats(df: pd.DataFrame) -> str:
    cols = ["Ann. Return (\\%)", "Return Vol (\\%)",
            "Mean Flow (\\%)", "Flow Vol (\\%)", "Mean Alpha (\\%)"]
    col_spec = "l" + "r" * len(cols)
    header = " & ".join(["\\textbf{Asset Class}"] +
                         [f"\\textbf{{{c}}}" for c in cols])

    def is_sub(idx):
        return str(idx).startswith("  ")

    def label(idx):
        return f"\\quad {str(idx).strip()}" if is_sub(idx) else str(idx)

    def fv(v):
        return f"{v:.2f}\\%" if pd.notna(v) and v is not None else "--"

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Fund Panel: Return and Flow Statistics by Broad Asset Class}",
        "  \\label{tab:fund_stats}",
        "  \\resizebox{\\textwidth}{!}{%",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    {header} \\\\",
        "    \\midrule",
    ]
    ov = df.loc["Overall"]
    lines.append("    " + " & ".join(
        ["\\textbf{Overall}"] + [f"\\textbf{{{fv(v)}}}" for v in ov.values]
    ) + " \\\\")
    lines.append("    \\midrule")
    for idx, row in df.drop("Overall").iterrows():
        sub = is_sub(idx)
        row_label = label(idx)
        vals = " & ".join([row_label] + [fv(v) for v in row.values])
        lines.append(f"    {vals} \\\\")
        if not sub and idx != list(df.drop("Overall").index)[-1]:
            next_idx = list(df.drop("Overall").index)[
                list(df.drop("Overall").index).index(idx) + 1]
            if not is_sub(next_idx):
                lines.append("    \\addlinespace")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}%",
        "  }",
        "  {\\footnotesize\\textit{Notes:} Quarterly panel across all Lipper broad asset classes "
        "(66,580 fund-quarter observations). "
        "Ann.\\ Return and Return Vol are annualised ($\\times 4$ and $\\times\\sqrt{4}$). "
        "\\textit{Flow} is the net quarterly flow rate as \\% of lagged TNA. "
        "Alpha is the rolling 8-quarter OLS alpha vs.\\ S\\&P~500 and US bond market, "
        "annualised ($\\times 4$). Indented rows show selected sub-categories.}",
        "\\end{table}",
    ]
    return "\n".join(lines)

latex_stats_str = latex_stats(df_stats)

# --- Append to summary_tables.txt ----------------------------------------
with open(out_path, "a") as f:
    f.write("\n\n% TABLE 2: Lipper Universe by Broad Asset Class\n\n")
    f.write(latex_uni)
    f.write("\n\n% TABLE 3: Fund Panel -- Return & Flow Statistics by Broad Asset Class\n\n")
    f.write(latex_stats_str)
    f.write("\n")

print(f"\nLipper tables appended to: {out_path}")

# ---------------------------------------------------------------------------
# 10. Cumulative return plot
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Align all series to a common start (first date all PM series are available)
# and compute cumulative wealth index (starting at 1.0)

def cum_ret(r: pd.Series) -> pd.Series:
    """Cumulative wealth index: (1+r1)*(1+r2)*... starting at 1.0 on first date."""
    r = r.dropna()
    return (1 + r).cumprod()

# Gather series per panel
panels = {
    "Private Markets (Hamilton Lane, EUR)": {
        col: cum_ret(hl_ret_eur[col]) for col in hl_ret_eur.columns
    },
    "EUR Benchmarks": {
        "EURO STOXX 600 TR":  cum_ret(bbg_ret["EURO STOXX 600 TR"]),
        "Euro Agg Bond TR":   cum_ret(bbg_ret["Euro Agg Bond TR"]),
        "Euribor 3M (EUR003)": cum_ret(rf_eur),
    },
    "US Benchmarks (EUR)": {
        "S&P 500 (EUR)":         cum_ret(sp500_eur),
        "US Bond Market (EUR)":  cum_ret(bond_eur),
        "Euribor 3M (EUR003)":   cum_ret(rf_eur),
    },
}

# Colour palettes per panel
COLORS = {
    "Private Markets (Hamilton Lane, EUR)": ["#1f4e79", "#2e75b6", "#9dc3e6"],
    "EUR Benchmarks":                        ["#843c0c", "#c55a11", "#aaaaaa"],
    "US Benchmarks (USD→EUR)":               ["#375623", "#70ad47", "#aaaaaa"],
}
LINESTYLES = ["-", "-", "--"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, (title, series_dict), colors in zip(axes, panels.items(), COLORS.values()):
    for (label, s), color, ls in zip(series_dict.items(), colors, LINESTYLES):
        ax.plot(s.index, s.values, label=label, color=color,
                linewidth=1.6, linestyle=ls)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.1f}×")
    )
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=7.5, framealpha=0.8)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

plt.tight_layout()

png_path = BASE / "cumulative_returns.png"
fig.savefig(png_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Cumulative return plot saved to: {png_path}")

# ---------------------------------------------------------------------------
# 11. Active & terminated funds per quarter (full Lipper universe)
# ---------------------------------------------------------------------------

print("Loading full returns universe for active/terminated chart …")
ret_full = pd.read_csv(BASE / "LSEG_merged/returns_merged.csv",
                       parse_dates=["Date"])
ret_full = ret_full[ret_full["Date"] >= "2000-01-01"]
ret_full["Quarter"] = ret_full["Date"].dt.to_period("Q").dt.to_timestamp("Q")

# Active: unique funds with an observation in each quarter
active_q = (
    ret_full.groupby("Quarter")["Instrument"].nunique()
    .reset_index().rename(columns={"Instrument": "Active"})
)

# Terminated: last observed quarter per fund, cross-referenced with status
last_obs = ret_full.groupby("Instrument")["Date"].max().reset_index()
last_obs["Quarter"] = last_obs["Date"].dt.to_period("Q").dt.to_timestamp("Q")
last_obs = last_obs.merge(
    funds_meta[["RIC_clean", "FundClassStatusName"]],
    left_on="Instrument", right_on="RIC_clean", how="left")
terminated = (
    last_obs[last_obs["FundClassStatusName"].isin(["Liquidated", "Merged"])]
    .groupby("Quarter").size()
    .reset_index().rename(columns={0: "Terminated"})
)

chart_df = active_q.merge(terminated, on="Quarter", how="left").fillna(0)
chart_df = chart_df.sort_values("Quarter")
chart_df = chart_df[chart_df["Quarter"] < chart_df["Quarter"].max()]  # drop last quarter (artifact)

fig2, ax1 = plt.subplots(figsize=(11, 4))

# Active funds — left axis
ax1.fill_between(chart_df["Quarter"], chart_df["Active"],
                 alpha=0.15, color="#2e75b6")
ax1.plot(chart_df["Quarter"], chart_df["Active"],
         color="#1f4e79", linewidth=1.6, label="Active funds")
ax1.set_ylabel("Active Funds", fontsize=9, color="#1f4e79")
ax1.set_ylim(0, chart_df["Active"].max() * 1.08)
ax1.tick_params(axis="y", labelcolor="#1f4e79", labelsize=8)

# Terminated — right axis
ax2 = ax1.twinx()
ax2.bar(chart_df["Quarter"], chart_df["Terminated"],
        width=60, color="#c00000", alpha=0.7, label="Terminated (quarter)")
ax2.set_ylabel("Terminated per Quarter", fontsize=9, color="#c00000")
ax2.tick_params(axis="y", labelcolor="#c00000", labelsize=8)
ax2.set_ylim(0, chart_df["Terminated"].max() * 4)

ax1.set_title("Lipper Fund Universe: Active and Terminated Fund Classes per Quarter",
              fontsize=10)
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.tick_params(axis="x", rotation=45, labelsize=8)
ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
for spine in ["top"]:
    ax1.spines[spine].set_visible(False)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

plt.tight_layout()

png2_path = BASE / "active_funds_per_quarter.png"
fig2.savefig(png2_path, dpi=200, bbox_inches="tight")
plt.close(fig2)
print(f"Active funds plot saved to: {png2_path}")

# ---------------------------------------------------------------------------
# 12. Private markets cum returns with macro regime background shading
# ---------------------------------------------------------------------------

from matplotlib.patches import Patch

print("Loading macro regimes for regime-shaded plot …")
_macro_raw = pd.read_csv(BASE / "LSEG_merged/macro_indicators.csv",
                         index_col=0, parse_dates=True)
_macro_raw.index.name = "Date"
_macro_raw = _macro_raw.reset_index().dropna(subset=["growth", "inflation"])
_regime_map = {
    "GrowthUp/InflationDown":   "Goldilocks",
    "GrowthUp/InflationUp":     "Overheating",
    "GrowthDown/InflationDown": "Downturn",
    "GrowthDown/InflationUp":   "Stagflation",
}
_macro_raw["macro_regime"] = _macro_raw["regime"].map(_regime_map)
macro_reg = (
    _macro_raw.dropna(subset=["macro_regime"])
    [["Date", "macro_regime", "growth", "inflation"]]
    .set_index("Date").sort_index()
)

REGIME_COLORS = {
    "Goldilocks":  "#4caf50",   # green
    "Overheating": "#ff9800",   # orange
    "Stagflation": "#e53935",   # red
    "Downturn":    "#1e88e5",   # blue
}

# Build contiguous regime blocks (merge consecutive same-regime quarters)
reg_dates   = macro_reg.index
reg_series  = macro_reg["macro_regime"]

blocks: list[tuple] = []   # (start, end, regime)
for i, (date, regime) in enumerate(reg_series.items()):
    # quarter start ≈ date minus ~3 months
    qstart = date - pd.DateOffset(months=3)
    if blocks and blocks[-1][2] == regime:
        blocks[-1] = (blocks[-1][0], date, regime)
    else:
        blocks.append([qstart, date, regime])

fig3, ax3 = plt.subplots(figsize=(12, 5))

# Shade regime bands (only within plot range)
plot_start = hl_ret_eur.index.min()
for bstart, bend, regime in blocks:
    if bend < plot_start or bstart > hl_ret_eur.index.max():
        continue
    ax3.axvspan(max(bstart, plot_start), bend,
                alpha=0.20, color=REGIME_COLORS.get(regime, "#ffffff"), lw=0)

# Plot Hamilton Lane EUR cumulative returns
PM_COLORS = ["#1f4e79", "#2e75b6", "#5b9bd5"]
for col, color in zip(hl_ret_eur.columns, PM_COLORS):
    s = cum_ret(hl_ret_eur[col])
    ax3.plot(s.index, s.values, label=col, color=color, linewidth=1.8)

ax3.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
ax3.set_title("Private Markets: Cumulative Returns by Macro Regime (EUR, Q1 2000 = 1.0)",
              fontsize=10)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}×"))
ax3.xaxis.set_major_locator(mdates.YearLocator(2))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax3.tick_params(axis="x", rotation=45, labelsize=8)
ax3.tick_params(axis="y", labelsize=8)
ax3.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
for spine in ["top", "right"]:
    ax3.spines[spine].set_visible(False)

# Combined legend: series + regime patches
lines_h, labels_h = ax3.get_legend_handles_labels()
regime_patches = [Patch(facecolor=REGIME_COLORS[r], alpha=0.8, label=r)
                  for r in ["Goldilocks", "Overheating", "Stagflation", "Downturn"]]
ax3.legend(handles=lines_h + regime_patches,
           fontsize=7.5, loc="upper left", framealpha=0.85,
           ncol=2)

plt.tight_layout()
png3_path = BASE / "pm_returns_regimes.png"
fig3.savefig(png3_path, dpi=200, bbox_inches="tight")
plt.close(fig3)
print(f"PM regime chart saved to: {png3_path}")

# ---------------------------------------------------------------------------
# 13. Markov transition matrix (TABLE 5)
# ---------------------------------------------------------------------------

# Use full macro_indicators.csv history for transition matrix estimation
macro_full = macro_reg.copy()   # already loaded above with full history

cur_r  = macro_full["macro_regime"]
nxt_r  = cur_r.shift(-1)
valid  = nxt_r.notna()
tm_raw = pd.crosstab(cur_r[valid], nxt_r[valid], normalize="index")

REGIME_ORDER = ["Goldilocks", "Overheating", "Downturn", "Stagflation"]
tm = tm_raw.reindex(index=REGIME_ORDER, columns=REGIME_ORDER).fillna(0.0)

# Regime observation counts (for footnote / header)
reg_counts = macro_full["macro_regime"].value_counts().reindex(REGIME_ORDER).fillna(0).astype(int)

# Persistence statistics
persistence: dict = {}
for reg in REGIME_ORDER:
    stay = float(tm.loc[reg, reg])
    avg_dur = 1.0 / (1.0 - stay) if stay < 1.0 else float("inf")
    persistence[reg] = {"stay": stay, "avg_dur": avg_dur}

print("\n=== Transition Matrix ===")
print(tm.round(3))
print("\n=== Persistence ===")
for r, v in persistence.items():
    print(f"  {r:15s}: stay={v['stay']:.1%}  avg_dur={v['avg_dur']:.1f} qtrs")

def latex_transition_matrix(tm, persistence, counts, regime_order) -> str:
    n = len(regime_order)
    col_spec = "l" + "r" * n + "r"   # extra col for N obs
    header = " & ".join(
        ["\\textbf{From $\\downarrow$ / To $\\rightarrow$}"] +
        [f"\\textbf{{{r}}}" for r in regime_order] +
        ["\\textbf{N}"]
    )

    def fmt_cell(val, is_diag):
        s = f"{val:.3f}"
        return f"\\textbf{{{s}}}" if is_diag else s

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Macro Regime Transition Probabilities (First-Order Markov Chain)}",
        "  \\label{tab:transition_matrix}",
        "  \\resizebox{\\textwidth}{!}{%",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    {header} \\\\",
        "    \\midrule",
    ]

    for reg in regime_order:
        row_vals = [fmt_cell(float(tm.loc[reg, col]), col == reg)
                    for col in regime_order]
        n_obs = f"{int(counts.get(reg, 0)):,}"
        row_str = " & ".join([reg] + row_vals + [n_obs])
        lines.append(f"    {row_str} \\\\")

    lines.append("    \\midrule")
    stay_vals = [f"{persistence[r]['stay']:.1%}" for r in regime_order]
    dur_vals  = [f"{persistence[r]['avg_dur']:.1f}" for r in regime_order]
    lines.append("    " + " & ".join(
        ["\\textit{Stay prob.}"] + stay_vals + [""]) + " \\\\")
    lines.append("    " + " & ".join(
        ["\\textit{Avg. duration (qtrs)}"] + dur_vals + [""]) + " \\\\")

    # Regime distribution row
    total = int(counts.sum())
    share_vals = [f"{int(counts.get(r, 0)) / total:.1%}" for r in regime_order]
    lines.append("    " + " & ".join(
        ["\\textit{Freq. (\\% qtrs)}"] + share_vals + [f"{total:,}"]) + " \\\\")

    date_start = macro_full.index.min().year
    date_end   = macro_full.index.max().year
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}%",
        "  }",
        f"  {{\\footnotesize\\textit{{Notes:}} Transition probabilities estimated from quarterly "
        f"macro regime data ({date_start}--{date_end}, $N={total}$ quarter observations). "
        "Regimes follow \\citet{Ilmanen2014} and are defined by a median split on two "
        "composite z-scored indicators: Growth $= \\frac{1}{2}z(\\text{CFNAI}_{12m}) + "
        "\\frac{1}{2}z(\\text{IP surprise})$ and Inflation $= \\frac{1}{2}z(\\text{CPI YoY}) + "
        "\\frac{1}{2}z(\\text{CPI surprise})$, where surprises are realised minus "
        "Survey of Professional Forecasters 1-year-ahead forecasts and z-scores are "
        "computed on an expanding window to avoid look-ahead bias. "
        "The four cells are: Goldilocks (Growth$\\uparrow$, Inflation$\\downarrow$), "
        "Overheating (Growth$\\uparrow$, Inflation$\\uparrow$), "
        "Downturn (Growth$\\downarrow$, Inflation$\\downarrow$), "
        "Stagflation (Growth$\\downarrow$, Inflation$\\uparrow$). "
        "Diagonal entries (bold) are same-state persistence probabilities; "
        "\\textit{Avg.\\ duration} = $1/(1-p_{{ii}})$. "
        "Data sources: FRED (CFNAI, INDPRO, CPI); Philadelphia Fed Survey of Professional Forecasters.}",
        "\\end{table}",
    ]
    return "\n".join(lines)

latex_tm = latex_transition_matrix(tm, persistence, reg_counts, REGIME_ORDER)

with open(out_path, "a") as f:
    f.write("\n\n% TABLE 4: Markov Regime Transition Matrix\n\n")
    f.write(latex_tm)
    f.write("\n")

print(f"\nTransition matrix table appended to: {out_path}")

# ---------------------------------------------------------------------------
# 14. Return unsmoothing: GLM diagnostics table + comparison plot
# ---------------------------------------------------------------------------

print("Computing GLM unsmoothing diagnostics …")
import statsmodels.tsa.api as tsa_api

unsmoothed_usd = pd.read_csv(BASE / "Hamilton_Lane_unsmoothed_returns.csv",
                              index_col=0, parse_dates=True)

# Convert unsmoothed USD -> EUR
unsmoothed_eur = pd.DataFrame(index=unsmoothed_usd.index)
for col in unsmoothed_usd.columns:
    unsmoothed_eur[col] = usd_to_eur(unsmoothed_usd[col], eurusd_q)

# Minimal GLM re-fit to recover theta / k
def glm_fit(r_obs, k_max=3):
    r = r_obs.dropna().astype(float)
    mu = r.mean()
    x = r - mu
    full_idx = pd.date_range(x.index.min(), x.index.max(), freq="QE-DEC")
    x = x.reindex(full_idx).interpolate()
    x.index.freq = "QE-DEC"
    best = {"k": 0, "bic": np.inf, "theta": np.array([1.0])}
    for k in range(k_max + 1):
        try:
            fit = tsa_api.ARIMA(x, order=(0, 0, k), trend="n").fit(method="innovations")
            if fit.bic < best["bic"]:
                best["bic"] = fit.bic
                best["k"] = k
                b = np.asarray(fit.maparams, dtype=float) if k > 0 else np.array([])
                t0 = 1.0 / (1.0 + b.sum()) if k > 0 else 1.0
                best["theta"] = np.concatenate(([t0], t0 * b)) if k > 0 else np.array([1.0])
        except Exception:
            pass
    return best["k"], best["theta"]

SHORT_NAMES = {
    "Hamilton Lane Private Credit Index":  "HL Private Credit",
    "Hamilton Lane Private Equity Index":  "HL Private Equity",
    "Hamilton Lane Private Markets Index": "HL Private Markets",
}

unsm_rows = {}
for col in hl_ret_usd.columns:
    if col not in unsmoothed_usd.columns:
        continue
    r_obs  = hl_ret_usd[col].dropna()
    r_unsm = unsmoothed_usd[col].dropna()
    idx    = r_obs.index.intersection(r_unsm.index)
    r_obs  = r_obs.loc[idx]
    r_unsm = r_unsm.loc[idx]
    k, theta = glm_fit(r_obs)
    unsm_rows[SHORT_NAMES.get(col, col)] = {
        "k":       k,
        "t0":      theta[0] if len(theta) > 0 else np.nan,
        "t1":      theta[1] if len(theta) > 1 else np.nan,
        "t2":      theta[2] if len(theta) > 2 else np.nan,
        "ac1_raw":  r_obs.autocorr(1),
        "ac1_unsm": r_unsm.autocorr(1),
        "ac2_raw":  r_obs.autocorr(2),
        "ac2_unsm": r_unsm.autocorr(2),
        "vol_raw":  r_obs.std() * np.sqrt(4) * 100,
        "vol_unsm": r_unsm.std() * np.sqrt(4) * 100,
        "vol_ratio":r_unsm.std() / r_obs.std() if r_obs.std() > 0 else np.nan,
    }

df_unsm = pd.DataFrame(unsm_rows).T
print("\n=== Unsmoothing Diagnostics ===")
print(df_unsm.round(3).to_string())

# --- LaTeX table ---------------------------------------------------------
def latex_unsmoothing(df) -> str:
    col_spec = "l" + "r" * 10
    header = (
        "\\textbf{Series} & \\textbf{$k$} & "
        "\\textbf{$\\theta_0$} & \\textbf{$\\theta_1$} & \\textbf{$\\theta_2$} & "
        "\\textbf{AC(1) Raw} & \\textbf{AC(1) Unsm.} & "
        "\\textbf{AC(2) Raw} & \\textbf{AC(2) Unsm.} & "
        "\\textbf{Ann.\\,Vol Raw} & \\textbf{Ann.\\,Vol Unsm.}"
    )

    def fv(v, fmt=".3f"):
        return f"{v:{fmt}}" if pd.notna(v) else "--"

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{GLM Return Unsmoothing: Parameters and Diagnostics "
        "(Getmansky, Lo \\& Makarov, 2004)}",
        "  \\label{tab:unsmoothing}",
        "  \\resizebox{\\textwidth}{!}{%",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    {header} \\\\",
        "    \\midrule",
    ]
    for name, row in df.iterrows():
        t2 = fv(row["t2"]) if row["k"] >= 2 else "--"
        vals = " & ".join([
            str(name),
            str(int(row["k"])),
            fv(row["t0"]),
            fv(row["t1"]) if row["k"] >= 1 else "--",
            t2,
            fv(row["ac1_raw"]),
            fv(row["ac1_unsm"]),
            fv(row["ac2_raw"]),
            fv(row["ac2_unsm"]),
            f"{row['vol_raw']:.2f}\\%",
            f"{row['vol_unsm']:.2f}\\%",
        ])
        lines.append(f"    {vals} \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}%",
        "  }",
        "  {\\footnotesize\\textit{Notes:} GLM Getmansky--Lo--Makarov (2004) unsmoothing "
        "applied to Hamilton Lane USD quarterly returns. $k$ is the MA lag order selected "
        "by BIC (maximum $k=3$). $\\theta_j$ are the smoothing weights (sum to 1); "
        "$\\theta_0$ is the contemporaneous weight. AC($i$) is the $i$th-order autocorrelation. "
        "Ann.\\ Vol is annualised quarterly standard deviation ($\\times\\sqrt{4}$). "
        "Returns are in USD; stats computed over the overlapping sample.}",
        "\\end{table}",
    ]
    return "\n".join(lines)

latex_unsm = latex_unsmoothing(df_unsm)
with open(out_path, "a") as f:
    f.write("\n\n% TABLE 5: GLM Return Unsmoothing Diagnostics\n\n")
    f.write(latex_unsm)
    f.write("\n")
print(f"Unsmoothing table appended to: {out_path}")

# --- Plot: observed vs unsmoothed cumulative returns (EUR) ---------------
plot_cols = [c for c in hl_ret_eur.columns if c in unsmoothed_eur.columns]
fig4, axes4 = plt.subplots(1, len(plot_cols), figsize=(15, 5), sharey=True)

OBS_COLOR  = "#1f4e79"
UNSM_COLOR = "#c55a11"

for ax, col in zip(axes4, plot_cols):
    r_obs  = hl_ret_eur[col].dropna()
    r_unsm = unsmoothed_eur[col].dropna()
    idx    = r_obs.index.intersection(r_unsm.index)
    cum_obs  = (1 + r_obs.loc[idx]).cumprod()
    cum_unsm = (1 + r_unsm.loc[idx]).cumprod()

    ax.plot(cum_obs.index,  cum_obs.values,  color=OBS_COLOR,  linewidth=1.8,
            label="Observed")
    ax.plot(cum_unsm.index, cum_unsm.values, color=UNSM_COLOR, linewidth=1.8,
            linestyle="--", label="Unsmoothed")
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax.set_title(SHORT_NAMES.get(col, col), fontsize=12, fontweight="bold", pad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}×"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

plt.tight_layout()
png4_path = BASE / "unsmoothing_comparison.png"
fig4.savefig(png4_path, dpi=200, bbox_inches="tight")
plt.close(fig4)
print(f"Unsmoothing comparison plot saved to: {png4_path}")
