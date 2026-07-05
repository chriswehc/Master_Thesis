"""
Growth & Inflation Indicators — Ilmanen, Maloney & Ross (2014)
"Exploring Macroeconomic Sensitivities"
==============================================================

Growth Indicator    = ½·z(CFNAI_12m_avg)  + ½·z(IP_surprise)
Inflation Indicator = ½·z(CPI_yoy)         + ½·z(CPI_surprise)

"Surprise" = realised 12m growth − SPF 1-year-ahead forecast
             made at the START of that 12-month window

Sampling:    quarterly overlapping 12-month windows
Z-scoring:   expanding window (no look-ahead bias)
Regimes:     median split → binary Up / Down per indicator
Interaction: 4-cell Growth×Inflation classification

Dependencies
────────────
    pip install pandas numpy requests pandas_datareader openpyxl matplotlib
"""
# %%

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Config ────────────────────────────────────────────────────────────────────
START    = "1980-01-01"
END      = "2025-12-31"
MIN_OBS  = 30         # expanding z-score minimum window


# %%

# =============================================================================
# BLOCK 1 — FRED  (CFNAI / INDPRO / CPI)
# =============================================================================

print("=" * 60)
print("BLOCK 1: Downloading FRED data …")
print("=" * 60)

fred = web.DataReader(["CFNAI", "INDPRO", "CPIAUCSL"], "fred", START, END)
fred.columns   = ["cfnai", "indpro", "cpi"]
fred.index     = pd.to_datetime(fred.index)
fred           = fred.resample("MS").last()   # enforce month-start frequency

print(f"  CFNAI:  {fred['cfnai'].dropna().index[0].date()} → {fred['cfnai'].dropna().index[-1].date()}")
print(f"  INDPRO: {fred['indpro'].dropna().index[0].date()} → {fred['indpro'].dropna().index[-1].date()}")
print(f"  CPI:    {fred['cpi'].dropna().index[0].date()} → {fred['cpi'].dropna().index[-1].date()}")

# %%
MEAN_LEVEL  = "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/historical-data/meanLevel.xlsx"
MEAN_GROWTH = "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/historical-data/meanGrowth.xlsx"


print("Downloading SPF mean level forecasts …")
headers = {"User-Agent": "Mozilla/5.0"}
r = requests.get(MEAN_LEVEL, timeout=60, headers=headers)
r.raise_for_status()
spf_bytes = r.content   # keep bytes — BytesIO can only be read once

spf_indpro = pd.read_excel(BytesIO(spf_bytes), sheet_name="INDPROD",
                            na_values=["#N/A", "", "na"], engine="openpyxl")
spf_indpro.columns = spf_indpro.columns.str.strip().str.upper()

spf_cpi    = pd.read_excel(BytesIO(spf_bytes), sheet_name="CPI",
                            na_values=["#N/A", "", "na"], engine="openpyxl")
spf_cpi.columns = spf_cpi.columns.str.strip().str.upper()

print("INDPROD columns:", spf_indpro.columns.tolist())
print("CPI columns:",     spf_cpi.columns.tolist())


#%%

# =============================================================================
# BLOCK 2 — Survey of Professional Forecasters (Philadelphia Fed)
#
# Files are Excel with columns: YEAR | QUARTER | <VAR>1 … <VAR>6
# where the numeric suffix = forecast horizon in quarters (1=current Q,
# 2=1Q ahead, …, 5=4Q ahead).  We use h=5 (4 quarters ahead) as the
# 1-year-ahead forecast.
#
# Note: for INDPRO the SPF provides *level* forecasts (index values).
#       for CPI the SPF also provides *level* forecasts.
# =============================================================================




def spf_h_to_quarterly(df: pd.DataFrame, var_prefix: str, h: int) -> pd.Series:
    """
    Extract horizon-h forecast from an SPF level file.
    Returns a Series indexed by quarter-end date (the survey date).

    SPF column naming convention:
        h=1 → current quarter  → column suffix '1'   (e.g. INDPRO1)
        h=2 → 1Q ahead         → column suffix '2'
        ...
        h=5 → 4Q ahead (≈1yr)  → column suffix '5'
    """
    col = f"{var_prefix.upper()}{h}"
    if col not in df.columns:
        # Try alternative naming (e.g. '04' suffix)
        alt = [c for c in df.columns if c.startswith(var_prefix.upper()) and str(h) in c]
        if not alt:
            raise KeyError(f"Column '{col}' not found. Available: {[c for c in df.columns if c.startswith(var_prefix.upper())]}")
        col = alt[0]

    sub = df[["YEAR", "QUARTER", col]].dropna()
    sub["date"] = pd.PeriodIndex(
        year=sub["YEAR"].astype(int),
        quarter=sub["QUARTER"].astype(int),
        freq="Q"
    ).to_timestamp("Q")          # last day of that quarter
    return sub.set_index("date")[col].sort_index().rename(f"{var_prefix}_h{h}")


# h=5 = 4 quarters ahead ≈ 1-year-ahead level forecast
spf_ip_fcast  = spf_h_to_quarterly(spf_indpro, "INDPROD", h=5)
spf_cpi_fcast = spf_h_to_quarterly(spf_cpi,    "CPI",     h=5)

print(f"  SPF INDPRO: {spf_ip_fcast.dropna().index[0].date()} → {spf_ip_fcast.dropna().index[-1].date()}")
print(f"  SPF CPI:    {spf_cpi_fcast.dropna().index[0].date()} → {spf_cpi_fcast.dropna().index[-1].date()}")


# %%

# =============================================================================
# BLOCK 3 — Build quarterly 12-month window series
# =============================================================================

print("\n" + "=" * 60)
print("BLOCK 3: Constructing quarterly 12-month window series …")
print("=" * 60)

# ── 3a. CFNAI annual average ──────────────────────────────────────────────────
# "Annual average of monthly CFNAI over the trailing 12-month window"
cfnai_12m = (
    fred["cfnai"]
    .rolling(12, min_periods=10)
    .mean()
    .resample("Q").last()
)

# ── 3b. Realised IP 12-month growth rate ─────────────────────────────────────
ip_qtr  = fred["indpro"].resample("Q").last()
ip_yoy  = (ip_qtr / ip_qtr.shift(4) - 1) * 100   # trailing 4-quarter % growth

# ── 3c. IP surprise ──────────────────────────────────────────────────────────
#
# At survey date t (quarter t), SPF forecasts INDPRO level at t+4.
# At realisation date t+4, surprise = actual_yoy - forecasted_yoy.
#
# forecasted_yoy[t+4] = (SPF_forecast_made_at_t / actual_level_at_t - 1) * 100
#
# Implementation: shift spf_ip_fcast forward by 4 quarters so the forecast
# lands at the realisation date, then divide by the actual level 4Q ago.
#
spf_ip_at_realisation = spf_ip_fcast.shift(4)    # forecast for t, now at t+4
ip_actual_base        = ip_qtr.shift(4)           # actual level 4Q ago (= t)
ip_fcast_yoy          = (spf_ip_at_realisation / ip_actual_base - 1) * 100
ip_surprise           = ip_yoy - ip_fcast_yoy

# ── 3d. CPI YoY ──────────────────────────────────────────────────────────────
cpi_qtr = fred["cpi"].resample("Q").last()
cpi_yoy = (cpi_qtr / cpi_qtr.shift(4) - 1) * 100

# ── 3e. CPI surprise ─────────────────────────────────────────────────────────
spf_cpi_at_realisation = spf_cpi_fcast.shift(4)
cpi_actual_base        = cpi_qtr.shift(4)
cpi_fcast_yoy          = (spf_cpi_at_realisation / cpi_actual_base - 1) * 100
cpi_surprise           = cpi_yoy - cpi_fcast_yoy

# ── Align all series on a common quarterly index ─────────────────────────────
raw = pd.DataFrame({
    "cfnai_12m":   cfnai_12m,
    "ip_yoy":      ip_yoy,
    "ip_surprise": ip_surprise,
    "cpi_yoy":     cpi_yoy,
    "cpi_surprise": cpi_surprise,
}).dropna(how="all")

print(f"  Raw series shape: {raw.shape}  (quarters × series)")
print(f"  Date range:       {raw.index[0].date()} → {raw.index[-1].date()}")
print("\n  Missing-value counts:")
print(raw.isnull().sum().to_string())


# =============================================================================
# BLOCK 4 — Expanding-window z-scores
# =============================================================================

print("\n" + "=" * 60)
print("BLOCK 4: Computing expanding z-scores …")
print("=" * 60)

def expanding_zscore(s: pd.Series, min_obs: int = MIN_OBS) -> pd.Series:
    """Z-score using expanding (historical) mean and std — no look-ahead bias."""
    mu  = s.expanding(min_periods=min_obs).mean()
    sig = s.expanding(min_periods=min_obs).std()
    return ((s - mu) / sig).rename(f"z_{s.name}")


z = pd.DataFrame({
    "z_cfnai":       expanding_zscore(raw["cfnai_12m"]),
    "z_ip_surprise": expanding_zscore(raw["ip_surprise"]),
    "z_cpi_yoy":     expanding_zscore(raw["cpi_yoy"]),
    "z_cpi_surprise": expanding_zscore(raw["cpi_surprise"]),
})

print("  Z-score descriptive stats:")
print(z.describe().round(3).to_string())


# =============================================================================
# BLOCK 5 — Composite indicators + regime classification
# =============================================================================

print("\n" + "=" * 60)
print("BLOCK 5: Building composite indicators and regimes …")
print("=" * 60)

out = z.copy()
out["growth"]    = z[["z_cfnai",       "z_ip_surprise"]].mean(axis=1)
out["inflation"] = z[["z_cpi_yoy",     "z_cpi_surprise"]].mean(axis=1)

# Merge in raw levels for reference
out = out.join(raw[["cfnai_12m","ip_yoy","ip_surprise","cpi_yoy","cpi_surprise"]])

# ── Binary regimes (median split → equal cell counts) ────────────────────────
# Drop rows where both indicators are NaN before computing medians
valid = out[["growth","inflation"]].dropna()
med_g = valid["growth"].median()
med_i = valid["inflation"].median()

out["growth_up"]    = (out["growth"]    >= med_g).astype(float)
out["inflation_up"] = (out["inflation"] >= med_i).astype(float)

# 4-cell interaction label
def regime_label(row):
    if pd.isna(row["growth_up"]) or pd.isna(row["inflation_up"]):
        return np.nan
    g = "GrowthUp"    if row["growth_up"]    else "GrowthDown"
    i = "InflationUp" if row["inflation_up"] else "InflationDown"
    return f"{g}/{i}"

out["regime"] = out.apply(regime_label, axis=1)

print(f"  Growth    median: {med_g:.3f}")
print(f"  Inflation median: {med_i:.3f}")
print("\n  Regime counts (should be ~equal, ~25% each):")
print(out["regime"].value_counts().to_string())

# Save full output
# out = out.loc["2000":]

out.to_csv(os.path.join(_HERE, "macro_indicators.csv"))
print(f"\n  Saved → {os.path.join(_HERE, 'macro_indicators.csv')}")


# =============================================================================
# BLOCK 6 — Diagnostic plots
# =============================================================================

print("\n" + "=" * 60)
print("BLOCK 6: Generating plots …")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Ilmanen, Maloney & Ross (2014) — Macro Indicators", fontsize=13, fontweight="bold")

colors = {"GrowthUp/InflationUp":     "#d62728",
          "GrowthUp/InflationDown":   "#2ca02c",
          "GrowthDown/InflationUp":   "#ff7f0e",
          "GrowthDown/InflationDown": "#1f77b4"}

# Panel 1 — Growth indicator
ax = axes[0]
ax.plot(out.index, out["growth"], color="steelblue", lw=1.2)
ax.axhline(med_g, color="black", lw=0.8, ls="--", alpha=0.5)
ax.fill_between(out.index, out["growth"], med_g,
                where=(out["growth"] >= med_g), alpha=0.3, color="steelblue", label="Growth Up")
ax.fill_between(out.index, out["growth"], med_g,
                where=(out["growth"] < med_g),  alpha=0.3, color="salmon",    label="Growth Down")
ax.set_ylabel("Z-score", fontsize=9)
ax.set_title("Growth Indicator  = ½·z(CFNAI₁₂ₘ) + ½·z(IP Surprise)", fontsize=9)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

# Panel 2 — Inflation indicator
ax = axes[1]
ax.plot(out.index, out["inflation"], color="firebrick", lw=1.2)
ax.axhline(med_i, color="black", lw=0.8, ls="--", alpha=0.5)
ax.fill_between(out.index, out["inflation"], med_i,
                where=(out["inflation"] >= med_i), alpha=0.3, color="firebrick", label="Inflation Up")
ax.fill_between(out.index, out["inflation"], med_i,
                where=(out["inflation"] < med_i),  alpha=0.3, color="lightblue", label="Inflation Down")
ax.set_ylabel("Z-score", fontsize=9)
ax.set_title("Inflation Indicator = ½·z(CPI YoY) + ½·z(CPI Surprise)", fontsize=9)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

# Panel 3 — 4-cell regime
ax = axes[2]
regime_num = {"GrowthUp/InflationUp": 4, "GrowthUp/InflationDown": 3,
              "GrowthDown/InflationDown": 2, "GrowthDown/InflationUp": 1}
regime_series = out["regime"].map(regime_num)
for reg, val in regime_num.items():
    mask = regime_series == val
    ax.fill_between(out.index, 0, mask.astype(float) * val,
                    where=mask, color=colors[reg], alpha=0.7, label=reg)
ax.set_ylabel("Regime", fontsize=9)
ax.set_title("Growth × Inflation Interaction (4-cell)", fontsize=9)
ax.set_yticks(list(regime_num.values()))
ax.set_yticklabels(list(regime_num.keys()), fontsize=7)
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.grid(alpha=0.3)

ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=45, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(_HERE, "macro_indicators.png"), dpi=150, bbox_inches="tight")
print(f"  Saved → {os.path.join(_HERE, 'macro_indicators.png')}")
plt.show()

print("\n✓ All done.")
print(f"\nFinal indicator table (last 8 quarters):")
print(out[["growth","inflation","regime"]].dropna().tail(8).round(3).to_string())



# %%
