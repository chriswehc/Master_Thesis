"""
ELTIF Simulation Results — Streamlit Dashboard
================================================
Run with:  streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ELTIF Simulation Dashboard",
    page_icon="📊",
    layout="wide",
)

REGIME_COLORS = {
    "Goldilocks":  "#2ecc71",
    "Overheating": "#e67e22",
    "Downturn":    "#e74c3c",
    "Stagflation": "#9b59b6",
}

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading simulation results …")
def load_results():
    df = pd.read_csv("eltif_results_revolving_credit.csv")
    return df


@st.cache_data(show_spinner="Computing metrics …")
def compute_metrics(df):
    m = {}
    m["credit_usage_probability"]       = (df["Shortfall_Flag"] >= 1).mean()
    m["forced_liquidation_probability"] = (df["Shortfall_Flag"] == 2).mean()

    sf = df[df["Shortfall_Flag"] >= 1]["Shortfall"]
    m["expected_shortfall"] = sf.mean() if len(sf) > 0 else 0
    m["max_shortfall"]      = sf.max()  if len(sf) > 0 else 0

    cd = df[df["Credit_Drawn"] > 0]["Credit_Drawn"]
    m["avg_credit_drawn"] = cd.mean() if len(cd) > 0 else 0
    m["max_credit_drawn"] = cd.max()  if len(cd) > 0 else 0

    m["avg_credit_outstanding"] = df["Credit_Outstanding"].mean()
    m["max_credit_outstanding"] = df["Credit_Outstanding"].max()
    m["total_credit_interest"]          = df["Credit_Interest"].sum()
    m["avg_credit_interest_per_quarter"]= df["Credit_Interest"].mean()

    final = df.groupby("path")["TNA"].last()
    m["final_tna_mean"]   = final.mean()
    m["final_tna_median"] = final.median()
    m["final_tna_p10"]    = final.quantile(0.10)
    m["final_tna_p90"]    = final.quantile(0.90)

    m["avg_pm_return"] = df["PM_Return"].mean() * 4
    m["avg_flow_rate"] = df["Flow_Rate"].mean()

    m["credit_usage_by_regime"] = (
        df.groupby("Regime")["Shortfall_Flag"]
        .apply(lambda x: (x >= 1).mean())
        .to_dict()
    )
    return m


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
try:
    df_all = load_results()
except FileNotFoundError:
    st.error(
        "❌ `eltif_results_revolving_credit.csv` not found. "
        "Run `Core_fund_simulation.py` first to generate it."
    )
    st.stop()

all_paths   = sorted(df_all["path"].unique())
all_regimes = sorted(df_all["Regime"].dropna().unique())
n_paths     = len(all_paths)
n_quarters  = df_all["Quarter"].max() + 1

# ─────────────────────────────────────────────
# Sidebar — filters
# ─────────────────────────────────────────────
st.sidebar.title("🔧 Filters")
st.sidebar.markdown("---")

st.sidebar.subheader("Paths shown in fan charts")
n_display = st.sidebar.slider(
    "Number of paths to display",
    min_value=10,
    max_value=min(n_paths, 10000),
    value=min(n_paths, 200),
    step=10,
    help="Fewer paths = faster rendering",
)
display_paths = all_paths[:n_display]

st.sidebar.subheader("Regime filter (metrics & histograms)")
selected_regimes = st.sidebar.multiselect(
    "Macro regimes",
    options=all_regimes,
    default=all_regimes,
)

st.sidebar.subheader("Quarter range")
q_min, q_max = st.sidebar.slider(
    "Quarter window",
    min_value=0,
    max_value=int(n_quarters - 1),
    value=(0, int(n_quarters - 1)),
)

# Apply filters
df_filtered = df_all[
    (df_all["Regime"].isin(selected_regimes)) &
    (df_all["Quarter"] >= q_min) &
    (df_all["Quarter"] <= q_max)
]

metrics = compute_metrics(df_filtered)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("📊 ELTIF Simulation — Revolving Credit Line")
st.markdown(
    f"**{n_paths:,} Monte Carlo paths · {n_quarters} quarters (20 years) · "
    f"Regimes: {', '.join(selected_regimes)}**"
)
st.markdown("---")

# ─────────────────────────────────────────────
# KPI cards — row 1: liquidity stress
# ─────────────────────────────────────────────
st.subheader("Liquidity Stress Events")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Credit Usage Probability",      f"{metrics['credit_usage_probability']:.2%}")
c2.metric("Forced Liquidation Probability",f"{metrics['forced_liquidation_probability']:.2%}")
c3.metric("Expected Shortfall",            f"€{metrics['expected_shortfall']:,.0f}")
c4.metric("Max Shortfall",                 f"€{metrics['max_shortfall']:,.0f}")

# ─────────────────────────────────────────────
# KPI cards — row 2: credit line
# ─────────────────────────────────────────────
st.subheader("Credit Line Statistics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Credit Drawn / Event",  f"€{metrics['avg_credit_drawn']:,.0f}")
c2.metric("Max Credit Drawn",          f"€{metrics['max_credit_drawn']:,.0f}")
c3.metric("Avg Outstanding Balance",   f"€{metrics['avg_credit_outstanding']:,.0f}")
c4.metric("Max Outstanding Balance",   f"€{metrics['max_credit_outstanding']:,.0f}")

# ─────────────────────────────────────────────
# KPI cards — row 3: cost & performance
# ─────────────────────────────────────────────
st.subheader("Cost & Performance")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Credit Interest",       f"€{metrics['total_credit_interest']:,.0f}")
c2.metric("Avg Interest / Quarter",      f"€{metrics['avg_credit_interest_per_quarter']:,.0f}")
c3.metric("Avg PM Return (ann.)",        f"{metrics['avg_pm_return']:.2%}")
c4.metric("Avg Flow Rate",               f"{metrics['avg_flow_rate']:.2%}")
c5.metric("Median Final TNA",            f"€{metrics['final_tna_median']/1e6:.1f}M")

st.markdown("---")

# ─────────────────────────────────────────────
# Helper: pre-compute per-quarter percentiles
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def quarter_percentiles(df, col):
    grp = df.groupby("Quarter")[col]
    return pd.DataFrame({
        "p10":    grp.quantile(0.10),
        "p25":    grp.quantile(0.25),
        "median": grp.median(),
        "p75":    grp.quantile(0.75),
        "p90":    grp.quantile(0.90),
    })


# ─────────────────────────────────────────────
# Plots — Row 1
# ─────────────────────────────────────────────
col_left, col_mid, col_right = st.columns(3)

# ── 1. TNA Evolution ──────────────────────────
with col_left:
    st.subheader("TNA Evolution")
    fig = go.Figure()

    df_display = df_all[df_all["path"].isin(display_paths)]
    for path, g in df_display.groupby("path"):
        fig.add_trace(go.Scatter(
            x=g["Quarter"], y=g["TNA"] / 1e6,
            mode="lines",
            line=dict(color="royalblue", width=0.6),
            opacity=0.15,
            showlegend=False,
            hoverinfo="skip",
        ))

    pct = quarter_percentiles(df_all, "TNA")
    for label, col_name, dash in [
        ("Median", "median", "solid"),
        ("p10 / p90", "p10", "dash"),
        ("p90", "p90", "dash"),
    ]:
        if col_name in ("p10", "p90"):
            fig.add_trace(go.Scatter(
                x=pct.index, y=pct[col_name] / 1e6,
                mode="lines",
                line=dict(color="navy", width=2, dash=dash),
                name=col_name,
                showlegend=(col_name == "p10"),
                legendgroup="pct",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=pct.index, y=pct[col_name] / 1e6,
                mode="lines",
                line=dict(color="navy", width=2.5),
                name="Median",
            ))

    fig.update_layout(
        xaxis_title="Quarter", yaxis_title="TNA (€M)",
        height=350, margin=dict(t=10, b=40),
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig, width='stretch')

# ── 2. Credit Usage Probability by Regime ─────
with col_mid:
    st.subheader("Credit Usage by Regime")
    regime_probs = (
        df_filtered.groupby("Regime")["Shortfall_Flag"]
        .apply(lambda x: (x >= 1).mean())
        .reset_index()
        .rename(columns={"Shortfall_Flag": "Probability"})
    )
    fig = px.bar(
        regime_probs,
        x="Regime", y="Probability",
        color="Regime",
        color_discrete_map=REGIME_COLORS,
        text_auto=".1%",
    )
    fig.update_layout(
        showlegend=False,
        yaxis_tickformat=".0%",
        xaxis_title="",
        yaxis_title="Probability",
        height=350,
        margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig, width='stretch')

# ── 3. Flow Rate Distribution ─────────────────
with col_right:
    st.subheader("Flow Rate Distribution")
    flows = df_filtered["Flow_Rate"].dropna().clip(-0.5, 0.5) * 100
    fig = px.histogram(
        flows, nbins=60,
        color_discrete_sequence=["steelblue"],
        labels={"value": "Flow Rate (%)"},
    )
    fig.add_vline(x=0, line_color="red", line_dash="dash", line_width=2)
    fig.update_layout(
        xaxis_title="Flow Rate (%)", yaxis_title="Count",
        showlegend=False, height=350, margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────
# Plots — Row 2
# ─────────────────────────────────────────────
col_left, col_mid, col_right = st.columns(3)

# ── 4. Credit Outstanding ─────────────────────
with col_left:
    st.subheader("Credit Outstanding")
    fig = go.Figure()

    for path, g in df_display.groupby("path"):
        g_q = g[(g["Quarter"] >= q_min) & (g["Quarter"] <= q_max)]
        fig.add_trace(go.Scatter(
            x=g_q["Quarter"], y=g_q["Credit_Outstanding"] / 1e6,
            mode="lines",
            line=dict(color="red", width=0.6),
            opacity=0.15,
            showlegend=False,
            hoverinfo="skip",
        ))

    pct_c = quarter_percentiles(
        df_all[(df_all["Quarter"] >= q_min) & (df_all["Quarter"] <= q_max)],
        "Credit_Outstanding"
    )
    fig.add_trace(go.Scatter(
        x=pct_c.index, y=pct_c["median"] / 1e6,
        mode="lines", line=dict(color="darkred", width=4), name="Median",
    ))
    fig.add_trace(go.Scatter(
        x=pct_c.index, y=pct_c["p90"] / 1e6,
        mode="lines", line=dict(color="darkred", width=4, dash="dash"), name="p90",
    ))

    fig.update_layout(
        xaxis_title="Quarter", yaxis_title="Outstanding (€M)",
        height=350, margin=dict(t=10, b=40),
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig, width='stretch')

# ── 5. Credit Interest Distribution ───────────
with col_mid:
    st.subheader("Credit Interest Distribution")
    interest = df_filtered[df_filtered["Credit_Interest"] > 0]["Credit_Interest"]
    if len(interest) > 0:
        fig = px.histogram(
            interest, nbins=40,
            color_discrete_sequence=["orange"],
            labels={"value": "Interest Paid (€)"},
        )
        fig.update_layout(
            xaxis_title="Interest Paid (€)", yaxis_title="Count",
            showlegend=False, height=350, margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No credit interest paid in selected filter.")

# ── 6. Shortfall Size Distribution ───────────
with col_right:
    st.subheader("Shortfall Size Distribution")
    shortfalls = df_filtered[df_filtered["Shortfall"] > 0]["Shortfall"] / 1e6
    if len(shortfalls) > 0:
        fig = px.histogram(
            shortfalls, nbins=40,
            color_discrete_sequence=["crimson"],
            labels={"value": "Shortfall (€M)"},
        )
        fig.update_layout(
            xaxis_title="Shortfall (€M)", yaxis_title="Count",
            showlegend=False, height=350, margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No shortfalls in selected filter.")

# ─────────────────────────────────────────────
# Plots — Row 3
# ─────────────────────────────────────────────
col_left, col_mid, col_right = st.columns(3)

# ── 7. Final TNA Distribution ─────────────────
with col_left:
    st.subheader("Final TNA Distribution")
    final_tna = df_all.groupby("path")["TNA"].last() / 1e6
    med = final_tna.median()
    fig = px.histogram(
        final_tna, nbins=60,
        color_discrete_sequence=["seagreen"],
        labels={"value": "Final TNA (€M)"},
    )
    fig.add_vline(
        x=med, line_color="red", line_dash="dash", line_width=2,
        annotation_text=f"Median €{med:.0f}M",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="Final TNA (€M)", yaxis_title="Count",
        showlegend=False, height=350, margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig, width='stretch')

# ── 8. Flow-Performance Relationship ──────────
with col_mid:
    st.subheader("Flow–Performance Relationship")
    sample = df_filtered.dropna(subset=["Alpha", "Flow_Rate"]).sample(
        min(8000, len(df_filtered)), random_state=42
    )
    fig = px.scatter(
        sample,
        x=sample["Alpha"] * 100,
        y=sample["Flow_Rate"] * 100,
        color="Regime",
        color_discrete_map=REGIME_COLORS,
        opacity=0.35,
        labels={"x": "Alpha (%)", "y": "Flow Rate (%)", "color": "Regime"},
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        xaxis_title="Alpha (%)", yaxis_title="Flow Rate (%)",
        height=350, margin=dict(t=10, b=40),
        legend=dict(orientation="h", y=-0.30),
    )
    st.plotly_chart(fig, width='stretch')

# ── 9. Cumulative Credit Cost ─────────────────
with col_right:
    st.subheader("Cumulative Credit Cost")
    fig = go.Figure()

    for path, g in df_display.groupby("path"):
        g_q = g[(g["Quarter"] >= q_min) & (g["Quarter"] <= q_max)].copy()
        g_q = g_q.sort_values("Quarter")
        fig.add_trace(go.Scatter(
            x=g_q["Quarter"], y=g_q["Credit_Interest"].cumsum() / 1e6,
            mode="lines",
            line=dict(color="darkred", width=3),
            opacity=0.15,
            showlegend=False,
            hoverinfo="skip",
        ))

    # Percentile bands: compute cumulative sum per path then percentile per quarter
    df_q = df_all[(df_all["Quarter"] >= q_min) & (df_all["Quarter"] <= q_max)].copy()
    df_q = df_q.sort_values(["path", "Quarter"])
    df_q["CumCost"] = df_q.groupby("path")["Credit_Interest"].cumsum()
    pct_cost = df_q.groupby("Quarter")["CumCost"].agg(
        median="median", p90=lambda x: x.quantile(0.90)
    )
    fig.add_trace(go.Scatter(
        x=pct_cost.index, y=pct_cost["median"] / 1e6,
        mode="lines", line=dict(color="darkred", width=2.5), name="Median",
    ))
    fig.add_trace(go.Scatter(
        x=pct_cost.index, y=pct_cost["p90"] / 1e6,
        mode="lines", line=dict(color="darkred", width=2, dash="dash"), name="p90",
    ))

    fig.update_layout(
        xaxis_title="Quarter", yaxis_title="Cumulative Interest (€M)",
        height=350, margin=dict(t=10, b=40),
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig, width='stretch')

# ─────────────────────────────────────────────
# Final TNA summary table
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("Final TNA Distribution Summary")
final_tna_all = df_all.groupby("path")["TNA"].last() / 1e6
summary = pd.DataFrame({
    "Statistic": ["Mean", "Median", "p10", "p25", "p75", "p90", "Std Dev"],
    "Value (€M)": [
        f"{final_tna_all.mean():.2f}",
        f"{final_tna_all.median():.2f}",
        f"{final_tna_all.quantile(0.10):.2f}",
        f"{final_tna_all.quantile(0.25):.2f}",
        f"{final_tna_all.quantile(0.75):.2f}",
        f"{final_tna_all.quantile(0.90):.2f}",
        f"{final_tna_all.std():.2f}",
    ],
})
st.dataframe(summary, width='content', hide_index=True)

# ─────────────────────────────────────────────
# Raw data explorer
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("🔍 Raw Data Explorer", expanded=False):
    path_sel = st.selectbox("Select path", options=all_paths, index=0)
    df_path = df_all[df_all["path"] == path_sel].reset_index(drop=True)
    st.dataframe(
        df_path.style.format({
            "TNA": "€{:,.0f}", "Flow": "€{:,.0f}",
            "Credit_Outstanding": "€{:,.0f}", "Credit_Interest": "€{:,.0f}",
            "PM_Return": "{:.4f}", "Alpha": "{:.4f}", "Flow_Rate": "{:.4f}",
        }),
        width='stretch',
        height=400,
    )

st.caption("Master Thesis — WU Wien | ELTIF Monte Carlo Simulation with Revolving Credit Line")
