"""
ELTIF Cash Buffer Lookup Tool
==============================
A focused practitioner tool for ELTIF 2.0 fund managers to look up
their optimal cash buffer weight from pre-computed optimisation results.

Run with:  streamlit run streamlit_dashboard.py
"""

# PLAN
# Sidebar: gate mode (strict/economic), credit capacity scenario, haircut,
#          credit spread, gamma (risk tolerance), redemption gate (economic only)
# Main:    three metric cards — w*, P(emergency sale), return drag (bps)
# Below:   CRRA utility curve vs cash weight with w* marked
# Bottom:  sensitivity heatmap (haircut x spread), all-scenario comparison table

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Paths and constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OPT_DIR     = BASE_DIR / "optimization_runs"
REG_TAG     = "_bond_hy"

HAIRCUTS    = [5, 10, 20, 30]       # integer percent
SPREADS     = [100, 200, 300, 500]  # integer bps
GATE_PCTS   = [10, 20, 50]          # integer percent (economic mode gates)

PM_INDEX_LABELS = {
    "composite": "Composite (all PM asset classes)",
    "equity":    "Private Equity",
    "credit":    "Private Credit / Debt",
}

SCENARIO_LABELS = {
    "ample":   "Ample  (20% of fund size)",
    "tight":   "Tight  (5% of fund size)",
    "reg_max": "Reg. Max  (50% of fund size)",
}
SCENARIO_DISPLAY = {
    "ample":   "Ample (20% of TNA)",
    "tight":   "Tight (5% of TNA)",
    "reg_max": "Reg. Max (50% of TNA)",
}

PRIMARY_BLUE  = "#1B4F72"
ACCENT_TEAL   = "#148F77"
ACCENT_AMBER  = "#D4AC0D"
ACCENT_RED    = "#C0392B"
LIGHT_GREY_BG = "#F0F4F8"

# ─────────────────────────────────────────────────────────────────────────────
# File-name helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pm_tag(pm_index: str) -> str:
    return f"_pm_{pm_index}" if pm_index != "composite" else ""


def frontier_path_strict(haircut_int: int, spread_int: int, pm_index: str = "composite") -> Path:
    return OPT_DIR / (
        f"eltif_optimization_frontier{REG_TAG}{_pm_tag(pm_index)}"
        f"_h{haircut_int}_s{spread_int}_strict.csv"
    )


def optimal_path_strict(haircut_int: int, spread_int: int, pm_index: str = "composite") -> Path:
    return OPT_DIR / (
        f"eltif_optimization_optimal{REG_TAG}{_pm_tag(pm_index)}"
        f"_h{haircut_int}_s{spread_int}_strict.csv"
    )


def frontier_path_economic(gate_int: int, haircut_int: int, spread_int: int, pm_index: str = "composite") -> Path:
    return OPT_DIR / (
        f"eltif_optimization_frontier{REG_TAG}{_pm_tag(pm_index)}"
        f"_g{gate_int}_h{haircut_int}_s{spread_int}_economic.csv"
    )


def optimal_path_economic(gate_int: int, haircut_int: int, spread_int: int, pm_index: str = "composite") -> Path:
    return OPT_DIR / (
        f"eltif_optimization_optimal{REG_TAG}{_pm_tag(pm_index)}"
        f"_g{gate_int}_h{haircut_int}_s{spread_int}_economic.csv"
    )


def grid_summary_path_strict(pm_index: str = "composite") -> Path:
    return OPT_DIR / f"grid_summary{REG_TAG}{_pm_tag(pm_index)}_strict.csv"


def grid_summary_path_economic(gate_int: int, pm_index: str = "composite") -> Path:
    return OPT_DIR / f"grid_summary{REG_TAG}{_pm_tag(pm_index)}_g{gate_int}_economic.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Cached data loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def load_frontier(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False, ttl=3600)
def load_optimal(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False, ttl=3600)
def load_grid_summary(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ELTIF Cash Buffer Tool",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal custom CSS: tighten metric card spacing, set a clean background tone
st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #D6E4F0;
        border-radius: 8px;
        padding: 16px 20px 12px 20px;
    }
    [data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; color: #555; }
    .block-container { padding-top: 1.5rem; }

    /* Sidebar: smaller font for radio/selectbox option labels */
    [data-testid="stSidebar"] [data-testid="stRadio"] label p,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {
        font-size: 0.78rem !important;
        line-height: 1.3 !important;
    }
    /* Sidebar: smaller font for selectbox selected value */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] span {
        font-size: 0.78rem !important;
    }
    /* Sidebar: smaller font for slider labels */
    [data-testid="stSidebar"] [data-testid="stSlider"] label p {
        font-size: 0.78rem !important;
    }
    /* Sidebar: tighten subheader size */
    [data-testid="stSidebar"] h3 {
        font-size: 0.95rem !important;
        margin-bottom: 0.2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — fund manager inputs
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Fund Parameters")
    st.caption(
        "Set your fund's current parameters below. "
        "The tool looks up your pre-computed optimal cash allocation instantly."
    )

    st.divider()

    # ── Gate mode ─────────────────────────────────────────────────────────────
    st.subheader("Gate Mode")
    gate_mode = st.radio(
        "Redemption gate type",
        options=["Strict (ELTIF 2.0 Regulatory)", "Economic (Fund Manager Choice)"],
        index=0,
        help=(
            "**Strict (ELTIF 2.0):** The gate is anchored to your declared cash target — "
            "max quarterly outflow = 50% of your cash weight. "
            "This is the regulatory default under ELTIF 2.0. "
            "The optimum may be at the boundary (minimum cash) because a lower cash target "
            "mechanically tightens the gate.\n\n"
            "**Economic:** A fixed gate applies regardless of cash weight (e.g. max 20% of "
            "fund size per quarter). Use this to find a genuine interior optimum."
        ),
        label_visibility="collapsed",
    )
    is_strict = gate_mode.startswith("Strict")

    if not is_strict:
        redemption_gate = st.selectbox(
            "Maximum quarterly redemption",
            options=GATE_PCTS,
            index=1,
            format_func=lambda g: f"{g}% of fund size per quarter",
            help=(
                "Under the economic gate, investors can redeem up to this percentage of "
                "the fund's total assets each quarter, regardless of cash held. "
                "The optimiser finds the cash weight that maximises risk-adjusted return "
                "given this fixed gate."
            ),
        )
    else:
        redemption_gate = None

    # gate_int is defined here so it is always in scope regardless of branch.
    # On the strict path it remains None and is never evaluated (ternary guard).
    gate_int = redemption_gate  # int or None

    st.divider()

    # ── Credit capacity scenario ───────────────────────────────────────────────
    st.subheader("Credit Facility Size")
    credit_scenario_key = st.radio(
        "How large is your revolving credit line?",
        options=list(SCENARIO_LABELS.keys()),
        format_func=lambda k: SCENARIO_LABELS[k],
        index=0,
        label_visibility="collapsed",
        help=(
            "The revolving credit line acts as a second-tier liquidity buffer. "
            "It is drawn only when the cash pool is exhausted.\n\n"
            "- **Ample (20%):** credit can cover up to 20% of fund size — "
            "fund rarely needs to sell PM assets.\n"
            "- **Tight (5%):** limited credit headroom — "
            "more reliance on cash and PM asset sales.\n"
            "- **Reg. Max (50%):** maximum permitted under ELTIF 2.0 — "
            "provides the strongest backstop but increases cost of capital."
        ),
    )

    st.divider()

    # ── Haircut rate ──────────────────────────────────────────────────────────
    st.subheader("Forced Sale Haircut")
    haircut_sel = st.selectbox(
        "Expected price discount on emergency asset sale",
        options=HAIRCUTS,
        index=1,
        format_func=lambda h: f"{h}%",
        help=(
            "If all cash and credit are exhausted, the fund must liquidate private market "
            "assets at short notice. This is the discount below fair value you would expect "
            "to accept under those conditions.\n\n"
            "Typical range: 5% (liquid credit, mild stress) to 30% (illiquid equity, severe stress)."
        ),
    )

    st.divider()

    # ── Credit spread ─────────────────────────────────────────────────────────
    st.subheader("Borrowing Cost")
    spread_sel = st.selectbox(
        "Credit facility spread over base rate",
        options=SPREADS,
        index=2,
        format_func=lambda s: f"{s} bps",
        help=(
            "The annualised spread you pay over the 3-month Euribor on your revolving "
            "credit facility. This is the 'insurance premium' for having credit available.\n\n"
            "100 bps = strong relationship bank pricing; 500 bps = stressed market / "
            "first-time borrower."
        ),
    )

    st.divider()

    # ── Risk tolerance (gamma) ─────────────────────────────────────────────────
    st.subheader("Risk Tolerance")
    gamma_sel = st.slider(
        "Risk tolerance",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.25,
        format="%.2f",
        help=(
            "Controls the trade-off between expected returns and downside risk.\n\n"
            "- **1.0 (Aggressive):** maximises expected log return — accepts higher "
            "probability of emergency asset sales.\n"
            "- **2.0 (Standard):** benchmark setting used in most macro-finance research. "
            "Balances return and risk.\n"
            "- **5.0 (Conservative):** strongly penalises any scenario involving "
            "emergency asset sales, even at significant cost in forgone return."
        ),
    )
    # Display the risk label
    if gamma_sel <= 1.5:
        risk_label = "Aggressive"
        risk_color = ACCENT_RED
    elif gamma_sel <= 2.5:
        risk_label = "Moderate"
        risk_color = ACCENT_TEAL
    elif gamma_sel <= 3.5:
        risk_label = "Cautious"
        risk_color = ACCENT_AMBER
    else:
        risk_label = "Conservative"
        risk_color = PRIMARY_BLUE

    st.markdown(
        f"<span style='color:{risk_color};font-weight:600;'>Profile: {risk_label}</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── PM asset class ────────────────────────────────────────────────────────
    st.subheader("Private Market Asset Class")
    available_pm = [
        k for k in PM_INDEX_LABELS
        if (
            grid_summary_path_strict(k) if is_strict
            else grid_summary_path_economic(redemption_gate or 20, k)
        ).exists()
    ]
    if not available_pm:
        available_pm = ["composite"]
    pm_index_sel = st.selectbox(
        "Hamilton Lane index used for PM returns",
        options=available_pm,
        format_func=lambda k: PM_INDEX_LABELS.get(k, k),
        help=(
            "Determines which Hamilton Lane private market index is used as the "
            "PM return series in the simulation.\n\n"
            "- **Composite:** blended across all PM strategies (default)\n"
            "- **Private Equity:** equity-focused PE/VC return profile\n"
            "- **Private Credit/Debt:** senior/mezzanine debt return profile\n\n"
            "Options shown are those for which optimisation results exist."
        ),
    )

    st.divider()

    # ── Fund size ─────────────────────────────────────────────────────────────
    st.subheader("Fund Size")
    fund_size_m = st.number_input(
        "Total net assets (EUR M)",
        min_value=10,
        max_value=5000,
        value=100,
        step=10,
        help=(
            "Used to convert percentage allocations into euro amounts throughout the tool. "
            "The optimal cash weight (w*) itself is independent of fund size — the "
            "optimisation was run at a fixed EUR 100M baseline. Only the EUR amounts "
            "shown alongside percentages scale with this input."
        ),
    )
    fund_size_eur = fund_size_m * 1_000_000

    st.divider()
    st.caption(
        "Calibrated on high-yield bond ELTIF fund flows (Goldstein et al. 2017 regression). "
        "10-year optimisation horizon. Monte Carlo paths per grid point: see "
        "`--n_paths` argument used when generating the CSV files "
        "(optimization_grid_analysis.py default: 1,000)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Resolve file paths from user inputs
# ─────────────────────────────────────────────────────────────────────────────
if is_strict:
    f_path = frontier_path_strict(haircut_sel, spread_sel, pm_index_sel)
    o_path = optimal_path_strict(haircut_sel, spread_sel, pm_index_sel)
    grid_path = grid_summary_path_strict(pm_index_sel)
    gate_label = "ELTIF 2.0 Strict Gate"
else:
    f_path = frontier_path_economic(gate_int, haircut_sel, spread_sel, pm_index_sel)
    o_path = optimal_path_economic(gate_int, haircut_sel, spread_sel, pm_index_sel)
    grid_path = grid_summary_path_economic(gate_int, pm_index_sel)
    gate_label = f"Economic Gate ({gate_int}% / quarter)"

# ─────────────────────────────────────────────────────────────────────────────
# Check files exist before proceeding
# ─────────────────────────────────────────────────────────────────────────────
files_missing = not f_path.exists() or not o_path.exists()

# ─────────────────────────────────────────────────────────────────────────────
# Main page header
# ─────────────────────────────────────────────────────────────────────────────
st.title("ELTIF Cash Buffer Optimisation Tool")
st.caption(
    "Look up the optimal cash allocation for your ELTIF fund. "
    "Adjust the parameters in the sidebar and the recommendation updates instantly."
)

# Parameter summary bar
_gate_val  = "Strict (ELTIF 2.0)" if is_strict else f"Economic ({gate_int}%/qtr)"
_param_bar = [
    ("Gate Mode",          _gate_val),
    ("Credit Facility",    SCENARIO_DISPLAY[credit_scenario_key]),
    ("Forced Sale Haircut", f"{haircut_sel}%"),
    ("Borrowing Cost",     f"{spread_sel} bps"),
    ("Risk Tolerance",     f"γ = {gamma_sel:.2f} ({risk_label})"),
]
param_cols = st.columns(len(_param_bar))
for col, (label, value) in zip(param_cols, _param_bar):
    col.markdown(
        f"<div style='background:#F8FAFC;border:1px solid #D6E4F0;border-radius:6px;"
        f"padding:8px 10px;text-align:center'>"
        f"<div style='font-size:0.70rem;color:#666;margin-bottom:2px'>{label}</div>"
        f"<div style='font-size:0.82rem;font-weight:600;color:#1B4F72;line-height:1.3'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Handle missing file gracefully
# ─────────────────────────────────────────────────────────────────────────────
if files_missing:
    st.error(
        f"Optimisation results not found for this parameter combination. "
        f"Expected files:\n\n"
        f"- `{f_path.name}`\n"
        f"- `{o_path.name}`\n\n"
        f"Run the optimisation grid first:\n"
        f"```\npython optimization_grid_analysis.py --reg_tag _bond_hy \\\n"
        f"    --haircuts \"5,10,20,30\" --spreads \"100,200,300,500\" \\\n"
        f"    --gate_mode {'strict' if is_strict else 'economic'}"
        + (f" --redemption_gate_pct {gate_int/100:.2f}" if not is_strict else "")
        + "\n```"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
df_frontier = load_frontier(str(f_path))
df_optimal  = load_optimal(str(o_path))

# ── Column safety check ───────────────────────────────────────────────────────
required_frontier = [
    "scenario", "cash_weight", "p_fl_any_path",
    "expected_log_growth_ann", "expected_shortfall_fl_cond",
]
required_optimal = [
    "scenario", "gamma", "optimal_cash_weight",
    "expected_log_growth_ann", "p_fl_any_path",
]
# Build the CRRA column name: gamma_sel is a float from the slider, e.g. 2.0 → "crra_g2.0"
crra_col = f"crra_g{gamma_sel}"

missing_f = [c for c in required_frontier if c not in df_frontier.columns]
missing_o = [c for c in required_optimal  if c not in df_optimal.columns]
if missing_f or missing_o:
    st.error(
        f"Unexpected column layout in result files.\n"
        f"Frontier missing: {missing_f}\n"
        f"Optimal missing: {missing_o}"
    )
    st.stop()

if crra_col not in df_frontier.columns:
    st.error(
        f"CRRA column `{crra_col}` not found in frontier data. "
        f"Available CRRA columns: "
        f"{[c for c in df_frontier.columns if c.startswith('crra_g')]}"
    )
    st.stop()

# ── Extract the row for the selected scenario and gamma ───────────────────────
df_scen_frontier = df_frontier[df_frontier["scenario"] == credit_scenario_key].copy()
df_scen_optimal  = df_optimal[df_optimal["scenario"]  == credit_scenario_key].copy()

row_opt = df_scen_optimal[
    (df_scen_optimal["gamma"] - gamma_sel).abs() < 1e-6
]

if row_opt.empty:
    # Nearest gamma fallback
    nearest_gamma = float(
        df_scen_optimal.iloc[
            (df_scen_optimal["gamma"] - gamma_sel).abs().argsort().iloc[0]
        ]["gamma"]
    )
    row_opt = df_scen_optimal[
        (df_scen_optimal["gamma"] - nearest_gamma).abs() < 1e-6
    ]
    st.warning(
        f"Exact gamma {gamma_sel} not found — showing nearest available: {nearest_gamma}"
    )

if row_opt.empty:
    st.error(
        f"No optimisation result found for scenario '{credit_scenario_key}' "
        f"and risk tolerance {gamma_sel}."
    )
    st.stop()

row_opt = row_opt.iloc[0]

# ── Core result values ────────────────────────────────────────────────────────
w_star          = float(row_opt["optimal_cash_weight"])
p_fl            = float(row_opt["p_fl_any_path"])
log_growth_ann  = float(row_opt["expected_log_growth_ann"])

# Return drag: annualised log return at minimum cash weight (5%) vs at w*.
# Use abs() tolerance rather than exact float equality to avoid silent NaN if
# the grid starts at 0.04999... due to floating-point representation.
min_w_row = df_scen_frontier[
    (df_scen_frontier["cash_weight"] - 0.05).abs() < 1e-5
]
if not min_w_row.empty:
    base_log_growth = float(min_w_row.iloc[0]["expected_log_growth_ann"])
    return_drag_bps = (base_log_growth - log_growth_ann) * 10_000
else:
    # Fallback: use the lowest available cash weight as baseline
    min_w_row = df_scen_frontier.nsmallest(1, "cash_weight")
    if not min_w_row.empty:
        base_log_growth = float(min_w_row.iloc[0]["expected_log_growth_ann"])
        return_drag_bps = (base_log_growth - log_growth_ann) * 10_000
    else:
        return_drag_bps = float("nan")

# Expected haircut cost at w* — look up the frontier row matching w*
wstar_row = df_scen_frontier[
    (df_scen_frontier["cash_weight"] - w_star).abs() < 1e-5
]
if not wstar_row.empty:
    es_cond = float(wstar_row.iloc[0]["expected_shortfall_fl_cond"])
    haircut_cost_pct = es_cond * (haircut_sel / 100) / fund_size_eur * 100
else:
    es_cond          = float("nan")
    haircut_cost_pct = float("nan")

# ─────────────────────────────────────────────────────────────────────────────
# Main result — three headline metric cards
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Recommended Allocation")

col_w, col_credit, col_cost = st.columns(3)

with col_w:
    st.metric(
        label="Recommended Cash Allocation",
        value=f"{w_star:.0%}",
        help=(
            "The cash weight that maximises expected risk-adjusted return "
            "over a 10-year horizon for your selected parameters. "
            "Hold this share of fund assets in cash or near-cash instruments."
        ),
    )
    cash_eur = w_star * fund_size_eur
    st.caption(
        f"Hold **EUR {cash_eur/1e6:.1f}M** as cash or near-cash instruments "
        f"(out of EUR {fund_size_m}M total fund size)."
    )

with col_credit:
    # Credit capacity in EUR
    credit_cap_frac = {"ample": 0.20, "tight": 0.05, "reg_max": 0.50}[credit_scenario_key]
    credit_eur = credit_cap_frac * fund_size_eur
    st.metric(
        label="Credit Facility Size",
        value=f"EUR {credit_eur/1e6:.1f}M",
        help=(
            "The revolving credit facility size for your selected credit scenario. "
            "This acts as the second-tier buffer after cash is exhausted."
        ),
    )
    st.caption(
        f"Maintain a **EUR {credit_eur/1e6:.1f}M** revolving credit facility "
        f"({credit_cap_frac:.0%} of fund size under the {SCENARIO_DISPLAY[credit_scenario_key]} scenario)."
    )

with col_cost:
    if not np.isnan(return_drag_bps):
        cost_display = f"{return_drag_bps:.0f} bps/yr"
        cost_eur_m = return_drag_bps / 10_000 * fund_size_eur / 1e6
    else:
        cost_display = "N/A"
        cost_eur_m = float("nan")
    st.metric(
        label="Annual Cost of Holding Buffer",
        value=cost_display,
        help=(
            "Basis points per year of forgone return from holding cash instead of "
            "fully investing in private market assets. "
            "Calculated as the return at minimum 5% cash minus the return at the optimal weight."
        ),
    )
    if not np.isnan(cost_eur_m):
        st.caption(
            f"This buffer costs approximately **EUR {cost_eur_m:.2f}M per year** "
            f"in forgone private market returns (at EUR {fund_size_m}M fund size)."
        )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Second row — emergency sale risk and haircut cost
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Risk Metrics at Recommended Allocation")

col_risk, col_hc, col_lg = st.columns(3)

with col_risk:
    # Colour-code the P(FL) gauge
    if p_fl < 0.02:
        risk_icon = "LOW"
        risk_bg   = "#D5F5E3"
        risk_border = "#27AE60"
        risk_text_col = "#1E8449"
    elif p_fl < 0.05:
        risk_icon = "MODERATE"
        risk_bg   = "#FEF9E7"
        risk_border = "#F1C40F"
        risk_text_col = "#9A7D0A"
    else:
        risk_icon = "ELEVATED"
        risk_bg   = "#FADBD8"
        risk_border = "#C0392B"
        risk_text_col = "#922B21"

    st.markdown(
        f"""
        <div style="
            background:{risk_bg};
            border:1.5px solid {risk_border};
            border-radius:8px;
            padding:16px 20px 12px 20px;
        ">
            <div style="font-size:0.85rem;color:#555;margin-bottom:4px;">
                Probability of Emergency Asset Sale
            </div>
            <div style="font-size:2.2rem;font-weight:700;color:{risk_text_col};">
                {p_fl:.1%}
            </div>
            <div style="font-size:0.8rem;color:{risk_text_col};font-weight:600;margin-top:2px;">
                {risk_icon}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Probability that the fund is forced to liquidate private market assets "
        "at least once over a 10-year horizon. "
        "Below 2% = low risk; 2-5% = moderate; above 5% = elevated."
    )

with col_hc:
    if not np.isnan(haircut_cost_pct):
        hc_display = f"{haircut_cost_pct:.3f}% of TNA"
        hc_eur = haircut_cost_pct / 100 * fund_size_eur
    else:
        hc_display = "N/A"
        hc_eur = float("nan")
    st.metric(
        label="Expected Fire-Sale Loss (paths with FL)",
        value=hc_display,
        help=(
            "Average cumulative fire-sale loss over the 10-year fund life, "
            "computed only for the paths on which at least one forced liquidation "
            "occurred. Calculated as: (total EUR value of PM assets force-sold "
            "across all quarters of those paths) x (haircut rate) / (fund size). "
            "This is a per-path total, not a per-event figure — paths with multiple "
            "forced-sale quarters will have higher totals."
        ),
    )
    if not np.isnan(hc_eur):
        st.caption(
            f"On paths where emergency sales occur, cumulative fire-sale losses "
            f"average approximately **EUR {hc_eur/1e6:.2f}M** over 10 years "
            f"(at {haircut_sel}% haircut, scaled to EUR {fund_size_m}M fund size)."
        )

with col_lg:
    st.metric(
        label="Expected Annualised Log Return at w*",
        value=f"{log_growth_ann:.2%}",
        help=(
            "The expected annual log return on the fund at the optimal cash weight. "
            "This is the fund's annual growth rate (net of the cash drag and credit costs) "
            "averaged across 10,000 simulated 10-year paths."
        ),
    )
    st.caption(
        f"Expected annual portfolio return at the recommended "
        f"{w_star:.0%} cash allocation across all simulated market environments."
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# CRRA utility curve — the optimisation landscape
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Why This Is the Optimal Cash Level")
st.caption(
    "The curve shows the risk-adjusted return score (CRRA utility) at every possible cash "
    "weight, given your risk tolerance setting. The peak is your optimal allocation. "
    "Left of the peak = under-buffered (higher emergency-sale risk). "
    "Right of the peak = over-buffered (too much return drag, not enough extra safety)."
)

df_curve = df_scen_frontier.sort_values("cash_weight").copy()

fig_curve = go.Figure()

# Shade the area under the curve
fig_curve.add_trace(go.Scatter(
    x=(df_curve["cash_weight"] * 100).tolist(),
    y=df_curve[crra_col].tolist(),
    fill="tozeroy",
    fillcolor="rgba(27, 79, 114, 0.08)",
    line=dict(color=PRIMARY_BLUE, width=2.5),
    mode="lines",
    name="Risk-adjusted return score",
    hovertemplate=(
        "Cash weight: %{x:.0f}%<br>"
        "Score: %{y:.5f}<br>"
        "<extra></extra>"
    ),
))

# Mark w*
w_star_pct = w_star * 100
crra_at_wstar_series = df_curve[
    (df_curve["cash_weight"] - w_star).abs() < 1e-5
][crra_col]
crra_at_wstar = float(crra_at_wstar_series.iloc[0]) if not crra_at_wstar_series.empty else None

if crra_at_wstar is not None:
    fig_curve.add_trace(go.Scatter(
        x=[w_star_pct],
        y=[crra_at_wstar],
        mode="markers",
        marker=dict(
            color=ACCENT_TEAL,
            size=14,
            symbol="diamond",
            line=dict(color="white", width=2),
        ),
        name=f"Optimal: {w_star:.0%}",
        hovertemplate=(
            f"<b>Optimal cash weight: {w_star:.0%}</b><br>"
            f"Score: {crra_at_wstar:.5f}<extra></extra>"
        ),
    ))
    fig_curve.add_vline(
        x=w_star_pct,
        line=dict(color=ACCENT_TEAL, width=2, dash="dash"),
        annotation_text=f"Optimal: {w_star:.0%}",
        annotation_position="top right",
        annotation_font=dict(size=12, color=ACCENT_TEAL),
    )

fig_curve.update_layout(
    xaxis=dict(
        title="Cash allocation (% of fund size)",
        ticksuffix="%",
        gridcolor="#EBF5FB",
    ),
    yaxis=dict(
        title="Risk-adjusted return score",
        gridcolor="#EBF5FB",
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=420,
    margin=dict(t=20, b=60, l=60, r=40),
    legend=dict(orientation="h", y=-0.18, x=0),
    hovermode="x unified",
    title=dict(
        text=(
            f"Risk-adjusted return score vs cash allocation  "
            f"| {SCENARIO_DISPLAY[credit_scenario_key]}  "
            f"| Risk tolerance {gamma_sel}  "
            f"| Haircut {haircut_sel}%  "
            f"| {spread_sel} bps spread"
        ),
        font=dict(size=13),
        x=0,
    ),
)
st.plotly_chart(fig_curve, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# All-scenario comparison table
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("How Other Credit Scenarios Compare")
st.caption(
    "Same haircut, spread, and risk tolerance — but different credit facility sizes. "
    "The highlighted row is your selected scenario."
)

comparison_rows = []
for sc_key in ["ample", "tight", "reg_max"]:
    sc_opt_row = df_optimal[
        (df_optimal["scenario"] == sc_key) &
        ((df_optimal["gamma"] - gamma_sel).abs() < 1e-6)
    ]
    if sc_opt_row.empty:
        continue
    sc_row = sc_opt_row.iloc[0]
    sc_w   = float(sc_row["optimal_cash_weight"])
    sc_pfl = float(sc_row["p_fl_any_path"])
    sc_lg  = float(sc_row["expected_log_growth_ann"])

    # return drag for this scenario — use tolerance match, not exact equality
    sc_frontier = df_frontier[df_frontier["scenario"] == sc_key]
    sc_min_row  = sc_frontier[(sc_frontier["cash_weight"] - 0.05).abs() < 1e-5]
    if sc_min_row.empty:
        sc_min_row = sc_frontier.nsmallest(1, "cash_weight")
    if not sc_min_row.empty:
        sc_base = float(sc_min_row.iloc[0]["expected_log_growth_ann"])
        sc_drag = (sc_base - sc_lg) * 10_000
    else:
        sc_drag = float("nan")

    comparison_rows.append({
        "Credit Facility": SCENARIO_DISPLAY[sc_key],
        "Recommended Cash": f"{sc_w:.0%}",
        "Cash (EUR M)": f"EUR {sc_w * fund_size_eur / 1e6:.1f}M",
        "P(Emergency Sale)": f"{sc_pfl:.1%}",
        "Annual Cost (bps)": f"{sc_drag:.0f}" if not np.isnan(sc_drag) else "—",
        "Exp. Return": f"{sc_lg:.2%}",
        "_is_selected": sc_key == credit_scenario_key,
    })

if comparison_rows:
    df_cmp = pd.DataFrame(comparison_rows)

    # Build highlighted HTML table
    header_cols = [c for c in df_cmp.columns if not c.startswith("_")]
    html_rows = []
    for _, r in df_cmp.iterrows():
        row_style = (
            f"background:#D6EAF8;font-weight:600;"
            if r["_is_selected"] else ""
        )
        cells = "".join(
            f"<td style='padding:8px 14px;border-bottom:1px solid #e0e0e0;{row_style}'>"
            f"{r[c]}</td>"
            for c in header_cols
        )
        html_rows.append(f"<tr>{cells}</tr>")

    header_html = "".join(
        f"<th style='padding:8px 14px;background:{PRIMARY_BLUE};color:white;"
        f"text-align:left;font-weight:600;'>{c}</th>"
        for c in header_cols
    )

    table_html = f"""
    <table style='border-collapse:collapse;width:100%;font-size:0.9rem;'>
      <thead><tr>{header_html}</tr></thead>
      <tbody>{''.join(html_rows)}</tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption(
        "Highlighted row = your selected scenario. "
        "P(Emergency Sale) = probability of at least one forced private market "
        "liquidation over the 10-year fund life."
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity heatmap — collapsed by default
# ─────────────────────────────────────────────────────────────────────────────
with st.expander(
    "Parameter Sensitivity: How the Recommendation Changes Across Scenarios",
    expanded=False,
):
    st.caption(
        "Each cell shows the recommended cash allocation for a different combination "
        "of forced-sale haircut (rows) and borrowing cost (columns). "
        "Your current selection is highlighted. "
        "Use this to check how sensitive your answer is to parameter uncertainty."
    )

    grid_missing = not grid_path.exists()
    if grid_missing:
        st.info(
            f"Sensitivity grid not found: `{grid_path.name}`. "
            f"Run the full grid optimisation to populate this view."
        )
    else:
        df_grid = load_grid_summary(str(grid_path))

        # Metric selector
        metric_options = {
            "Recommended Cash Allocation (w*)": "optimal_cash_weight",
            "Annual Cost of Buffer (bps)": "return_drag_bps",
            "Fire-Sale Loss per Event (% TNA)": "haircut_cost_pct_tna",
        }
        metric_label = st.selectbox(
            "Show in heatmap",
            list(metric_options.keys()),
            index=0,
            key="heatmap_metric",
        )
        metric_col = metric_options[metric_label]

        # Filter grid to selected scenario and gamma
        df_grid_sel = df_grid[
            (df_grid["scenario"] == credit_scenario_key) &
            ((df_grid["gamma"] - gamma_sel).abs() < 1e-6)
        ].copy()

        # Check if haircut_pct column exists (grid summary has it pre-computed)
        if "haircut_pct" not in df_grid_sel.columns or "spread_bps" not in df_grid_sel.columns:
            st.warning(
                "Grid summary file does not contain `haircut_pct` / `spread_bps` columns. "
                "Re-run `optimization_grid_analysis.py` to regenerate."
            )
        elif df_grid_sel.empty:
            st.info(
                f"No grid data for scenario '{credit_scenario_key}' "
                f"and risk tolerance {gamma_sel}."
            )
        else:
            # Build pivot table
            haircut_vals = sorted(df_grid_sel["haircut_pct"].unique())
            spread_vals  = sorted(df_grid_sel["spread_bps"].unique())

            z_matrix   = []
            text_matrix = []

            for h in haircut_vals:
                z_row   = []
                txt_row = []
                for s in spread_vals:
                    cell = df_grid_sel[
                        (df_grid_sel["haircut_pct"] == h) &
                        (df_grid_sel["spread_bps"]  == s)
                    ]
                    if cell.empty or metric_col not in cell.columns:
                        z_row.append(None)
                        txt_row.append("—")
                    else:
                        val = float(cell.iloc[0][metric_col])
                        # haircut_cost_pct_tna in the grid summary is pre-baked
                        # at the default EUR 100M simulation TNA.  Re-scale to
                        # the fund manager's actual fund size so the heatmap is
                        # consistent with the metric card above.
                        if metric_col == "haircut_cost_pct_tna":
                            val = val * (100_000_000 / fund_size_eur)
                        z_row.append(val)
                        if metric_col == "optimal_cash_weight":
                            txt_row.append(f"{val:.0%}")
                        elif metric_col == "return_drag_bps":
                            txt_row.append(f"{val:.0f}")
                        else:
                            txt_row.append(f"{val:.2f}%")
                z_matrix.append(z_row)
                text_matrix.append(txt_row)

            # Highlight current selection
            sel_h_idx = haircut_vals.index(haircut_sel) if haircut_sel in haircut_vals else None
            sel_s_idx = spread_vals.index(spread_sel)  if spread_sel  in spread_vals  else None

            colorscale = "Blues" if metric_col == "optimal_cash_weight" else "YlOrRd"

            fig_heat = go.Figure(go.Heatmap(
                z=z_matrix,
                x=[f"{s} bps" for s in spread_vals],
                y=[f"{h}% haircut" for h in haircut_vals],
                text=text_matrix,
                texttemplate="%{text}",
                textfont=dict(size=13),
                colorscale=colorscale,
                colorbar=dict(title=metric_label, thickness=16),
                hoverongaps=False,
                hovertemplate=(
                    "Haircut: %{y}<br>"
                    "Spread: %{x}<br>"
                    f"{metric_label}: %{{text}}"
                    "<extra></extra>"
                ),
            ))

            # Add a rectangle around the currently selected cell
            if sel_h_idx is not None and sel_s_idx is not None:
                fig_heat.add_shape(
                    type="rect",
                    x0=sel_s_idx - 0.5, x1=sel_s_idx + 0.5,
                    y0=sel_h_idx - 0.5, y1=sel_h_idx + 0.5,
                    line=dict(color=ACCENT_TEAL, width=3),
                )
                fig_heat.add_annotation(
                    x=sel_s_idx,
                    y=sel_h_idx,
                    text="Your selection",
                    showarrow=True,
                    arrowhead=2,
                    ax=40,
                    ay=-30,
                    font=dict(color=ACCENT_TEAL, size=11),
                    arrowcolor=ACCENT_TEAL,
                )

            fig_heat.update_layout(
                title=dict(
                    text=(
                        f"{metric_label} by haircut and borrowing cost  "
                        f"| {SCENARIO_DISPLAY[credit_scenario_key]}  "
                        f"| Risk tolerance {gamma_sel}"
                    ),
                    font=dict(size=13),
                    x=0,
                ),
                xaxis_title="Credit facility borrowing cost",
                yaxis_title="Forced-sale haircut rate",
                height=400,
                margin=dict(t=50, b=60, l=120, r=40),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption(
                f"Teal border = your current selection ({haircut_sel}% haircut, {spread_sel} bps). "
                f"Scenario: {SCENARIO_DISPLAY[credit_scenario_key]}. "
                f"Risk tolerance: {gamma_sel}."
            )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Glossary / help expander
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Glossary and methodology notes", expanded=False):
    st.markdown(
        """
        **Recommended Cash Allocation (w*)**
        The cash weight that maximises the expected risk-adjusted return over a 10-year
        fund life, given your selected parameters. Computed by grid-searching cash
        weights from 5% to 50% in 1% increments, simulating N Monte Carlo paths per
        grid point (N set via --n_paths when the grid was generated), and selecting
        the weight with the highest CRRA utility.

        **Credit Facility Size**
        A revolving credit line the fund can draw when cash is exhausted but before
        being forced to sell private market assets. The three scenarios (Ample/Tight/Reg. Max)
        correspond to credit line sizes of 20%, 5%, and 50% of total fund assets.

        **Annual Cost of Holding Buffer (bps)**
        The reduction in expected annualised log return from holding the recommended cash
        allocation versus holding only the minimum 5% cash. One basis point = 0.01% per year.

        **Probability of Emergency Asset Sale**
        The share of simulation paths on which the fund was forced to liquidate private
        market assets at least once over the 10-year horizon. This happens only when
        both the cash pool and the full credit line are exhausted simultaneously.

        **Risk Tolerance (1.0 – 5.0)**
        Parameterises how much the fund penalises downside outcomes. At 1.0 (log utility),
        the fund maximises expected log return. Higher values increasingly penalise paths
        with emergency asset sales, shifting the optimum towards more cash.

        **Forced Sale Haircut**
        The discount below fair value assumed when private market assets must be sold at
        short notice. This loss is borne by all remaining investors, not just those redeeming.

        **Gate Mode**
        - *Strict (ELTIF 2.0):* the maximum quarterly outflow equals 50% of the fund's
          declared cash weight. A 10% cash target implies a 5% per quarter outflow gate.
          The optimum under strict mode may lie at the minimum (5% cash) because a lower
          cash target also mechanically tightens the gate.
        - *Economic:* a fixed quarterly outflow gate applies regardless of cash weight.
          This recovers an interior optimum and is the more informative landscape for
          understanding the genuine return-risk trade-off.

        **Expected Fire-Sale Loss (paths with FL)**
        The average cumulative EUR loss from forced PM asset sales, measured only on
        simulation paths where at least one forced liquidation event occurred. This is
        a per-path 10-year total, not a per-quarter figure. Paths with multiple
        forced-sale quarters contribute their full cumulative total to this average.
        Scaled by your selected haircut rate and fund size.

        **Calibration**
        Fund flow sensitivities are calibrated on high-yield bond mutual fund data from
        LSEG/Lipper (2002–2026, ~2,600 fund-quarter observations) using the
        Goldstein, Jiang & Ng (2017) regime-conditional regression. Private market returns
        are drawn from Hamilton Lane composite indices (EUR-converted, unsmoothed via
        Getmansky, Lo & Makarov 2004). Macro regimes follow a first-order Markov chain
        estimated from FRED GDP and CPI data.

        **Fund size and scaling note**
        The optimal cash weight (w*) is independent of fund size — the optimisation was
        run at a fixed EUR 100M baseline TNA. Only the EUR amounts displayed alongside
        percentages scale with the fund size you enter in the sidebar. The sensitivity
        heatmap fire-sale loss percentages are also re-scaled to your fund size.
        """,
        unsafe_allow_html=False,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.caption(
    "Master Thesis — WU Wien  |  ELTIF 2.0 Cash Buffer Optimisation  |  "
    "Run `thesis-auditor` on `app.py` to verify."
)
