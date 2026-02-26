# %%
import pandas as pd
import yfinance as yf
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from IPython.display import display
from datetime import date
from pathlib import Path
import openpyxl
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from scipy.stats import multivariate_t


# %%
# Set file path and sheet index
FILE = Path("PM_Indices.xlsx")
SHEET_Hamilton_Lane = 2


# %%
def load_table(file: Path, sheet=0) -> pd.DataFrame:
    # 1) Read from Excel; treat blanks as NA (incl. empty strings / whitespace)
    df = pd.read_excel(
        file,
        sheet_name=sheet,
        engine="openpyxl",
        na_values=["", " ", "  ", "\t", "NA", "N/A", "#N/A"],
        keep_default_na=True,
    )

    # 2) Standardize column names
    df.columns = df.columns.astype(str).str.strip()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")])

    # Parse Date safely (handles "21.01.1997" and true Excel date cells)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="raise")

    # 4) Convert all other columns to numeric
    value_cols = [c for c in df.columns if c != "Date"]
    df[value_cols] = (
        df[value_cols]
        .replace({r"\s+": None}, regex=True)   # whitespace-only -> NA
        .replace({",": "."}, regex=True)       # decimal comma -> dot
        .apply(pd.to_numeric, errors="coerce") # invalid -> NA
    )
    # 5) Index 
    df = df.set_index("Date")  # if you prefer time-series indexing

    return df

# %%
Hamilton_Lane_df = load_table(FILE, SHEET_Hamilton_Lane)

# %% [markdown]
# ## Unsmoothing the return series

# %%
def fit_glm_theta_mle(r_obs: pd.Series, k: int):
    """
    Estimate GLM smoothing weights theta (sum to 1) using MLE of an MA(k).

    Paper mapping:
      X_t = R^o_t - mu  is MA(k): X_t = θ0 η_t + ... + θk η_{t-k}, sum θ = 1   [oai_citation:4‡Gemansky_Lo_Makarov.pdf](sediment://file_000000006e7071f496ce156d3cb29f38)

    Practical approach:
      1) demean X_t
      2) fit standard MA(k): X_t = eps_t + b1 eps_{t-1} + ... + bk eps_{t-k}
      3) transform to θ weights that sum to 1 (paper’s suggested transformation)  [oai_citation:5‡Gemansky_Lo_Makarov.pdf](sediment://file_000000006e7071f496ce156d3cb29f38)
    """
    import statsmodels.api as sm

    r_obs = r_obs.dropna().astype(float)
    mu_hat = r_obs.mean()
    x = r_obs - mu_hat  # X_t

    # MLE MA(k). enforce_invertibility True by default in statsmodels ARIMA.
    fit = sm.tsa.ARIMA(x, order=(0, 0, k), trend="n").fit(
        method="innovations"
    )

    # statsmodels returns MA params b1..bk for: x_t = eps_t + b1 eps_{t-1} + ... + bk eps_{t-k}
    b = np.asarray(fit.maparams, dtype=float)  # length k

    # Transform standard MA params -> θ weights with sum θ = 1
    # This matches the paper’s scaling argument / normalization transformation.  [oai_citation:6‡Gemansky_Lo_Makarov.pdf](sediment://file_000000006e7071f496ce156d3cb29f38)
    theta0 = 1.0 / (1.0 + b.sum())
    theta = np.concatenate(([theta0], theta0 * b))

    # sanity checks
    if not np.isfinite(theta).all():
        raise ValueError("Non-finite theta estimates. Try a smaller k or check data quality.")
    if abs(theta.sum() - 1.0) > 1e-6:
        raise ValueError("Theta weights do not sum to 1 (unexpected).")

    return {
        "theta": theta,         # θ0..θk
        "mu_hat": mu_hat,       # sample mean of observed returns
        "fit": fit,             # statsmodels result object
        "x": x                  # demeaned series used in fit
    }


def unsmooth_from_theta(x: pd.Series, theta: np.ndarray) -> pd.Series:
    """
    Invert the smoothing filter to recover eta_t:
      eta_t = (x_t - sum_{j=1..k} theta_j eta_{t-j}) / theta_0
    """
    x = x.dropna().astype(float)
    k = len(theta) - 1
    theta0 = float(theta[0])

    if theta0 == 0:
        raise ValueError("theta0 is zero; cannot invert.")

    eta = np.zeros(len(x), dtype=float)
    for t in range(len(x)):
        acc = x.iloc[t]
        for j in range(1, k + 1):
            if t - j >= 0:
                acc -= theta[j] * eta[t - j]
        eta[t] = acc / theta0

    return pd.Series(eta, index=x.index, name="eta_hat")


def choose_k_by_bic(r_obs: pd.Series, k_max: int = 4) -> pd.DataFrame:
    """
    Fit k = 0..k_max and return an IC table to choose k.
    """
    rows = []
    for k in range(k_max + 1):
        out = fit_glm_theta_mle(r_obs, k=k)
        fit = out["fit"]
        rows.append({
            "k": k,
            "bic": fit.bic,
            "aic": fit.aic,
            "theta0": out["theta"][0],
            "theta": out["theta"]
        })
    return pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)

# %%
def run_unsmoothing_panel(ret_df: pd.DataFrame, k_max: int = 4, freq: str = "QE-DEC"):
    summary_rows = []
    unsmoothed_cols = {}

    for col in ret_df.columns:
        # 1. Strip NaNs and ensure DatetimeIndex
        r = ret_df[col].dropna().copy()
        if len(r) < 20: 
            continue
        r.index = pd.to_datetime(r.index)

        # 2. Reindex to a clean, continuous frequency
        # This solves the statsmodels "ValueWarning"
        full_idx = pd.date_range(start=r.index.min(), end=r.index.max(), freq=freq)
        r = r.reindex(full_idx)

        # 3. Handle NaNs created by reindexing (gaps in the timeline)
        # We interpolate so the MA(k) model has continuous data points
        r = r.interpolate(method='linear')
        
        # 4. Explicitly assign the freq to the index
        r.index.freq = freq

        # --- Remaining Logic ---
        # 1) Choose k by BIC
        ic = choose_k_by_bic(r, k_max=k_max)
        best_k = int(ic.loc[0, "k"])

        # 2) Fit theta at best_k
        out = fit_glm_theta_mle(r, k=best_k)
        theta = out["theta"]
        mu_hat = out["mu_hat"]
        x = out["x"]
        fit = out["fit"]

        # 3) Unsmooth
        eta_hat = unsmooth_from_theta(x, theta)
        r_unsmoothed = (eta_hat + mu_hat).rename(col)

        unsmoothed_cols[col] = r_unsmoothed

        # Diagnostics
        ac1_obs = r.autocorr(lag=1) if len(r) > 2 else np.nan
        ac1_unsm = r_unsmoothed.autocorr(lag=1) if len(r_unsmoothed) > 2 else np.nan

        summary = {
            "series": col,
            "best_k": best_k,
            "aic": fit.aic,
            "bic": fit.bic,
            "n_obs": len(r),
            "vol_obs": r.std(),
            "vol_unsm": r_unsmoothed.std(),
            "ac1_obs": ac1_obs,
            "ac1_unsm": ac1_unsm,
        }

        for j, tj in enumerate(theta):
            summary[f"theta{j}"] = tj

        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows).set_index("series").sort_values("bic")
    unsmoothed_df = pd.concat(unsmoothed_cols.values(), axis=1).sort_index()

    return summary_df, unsmoothed_df

# %%
def vola_smoothing_analysis (df_observed,df_unsmoothed):
    rows = []
    for col in df_observed.columns:
        obs = df_observed[col].dropna()
        uns = unsmoothed_df[col].dropna()

        idx = obs.index.intersection(uns.index)
        obs = obs.loc[idx]
        uns = uns.loc[idx]

        rows.append({
            "series": col,
            "Observed SD (ann.)": obs.std() * np.sqrt(4),
            "Unsmoothed SD (ann.)": uns.std() * np.sqrt(4),
            "Vol Ratio": (uns.std() * np.sqrt(4)) / (obs.std() * np.sqrt(4)) if obs.std() != 0 else np.nan,
            "n": len(idx),
        })

    vol_df = pd.DataFrame(rows).set_index("series").sort_values("Vol Ratio", ascending=False)
    return vol_df


# %%
Hamilton_Lane_ret_df = Hamilton_Lane_df.pct_change().dropna()

# Set start and end date of the analysis

start_date = Hamilton_Lane_df.index[0]
end_date = Hamilton_Lane_df.index[-1]


corr = Hamilton_Lane_ret_df.corr()

corr_style = (
    corr.round(2)
        .style
        .background_gradient(axis=None)   # heatmap-like coloring
        .format("{:.2f}")
)

display(corr_style)

summary_df, unsmoothed_df = run_unsmoothing_panel(Hamilton_Lane_ret_df, k_max=2, freq="QE-DEC")

display(summary_df)

display(unsmoothed_df)

vola_smoothing_output = vola_smoothing_analysis(Hamilton_Lane_ret_df,unsmoothed_df)

display(vola_smoothing_output)

unsmoothed_df.to_csv("Hamilton_Lane_unsmoothed_returns.csv")

# %% [markdown]
# ### Plotting observed vs unsmoothed return series

# %%
obs = Hamilton_Lane_ret_df.sort_index()
unsm = unsmoothed_df.sort_index()

for col in obs.columns:
    if col not in unsm.columns:
        print(f"Skipping {col}: not found in unsmoothed_df")
        continue

    obs_s = obs[col].dropna()
    uns_s = unsm[col].dropna()

    # align on common dates
    idx = obs_s.index.intersection(uns_s.index)
    if len(idx) < 10:
        print(f"Skipping {col}: not enough overlapping data ({len(idx)} points)")
        continue

    obs_s = obs_s.loc[idx]
    uns_s = uns_s.loc[idx]

    # cumulative growth of $1
    cum_obs = (1 + obs_s).cumprod()
    cum_uns = (1 + uns_s).cumprod()

    # plot
    ax = cum_obs.plot(figsize=(12, 6), linewidth=2, label="Observed")
    cum_uns.plot(ax=ax, linewidth=2, label="Unsmoothed")

    ax.set_title(f"Cumulative Returns: Observed vs Unsmoothed\n{col}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()
