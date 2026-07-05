# %%
import lseg.data as ld
import pandas as pd
import numpy as np
from datetime import datetime
from lseg.data import discovery

ld.open_session()


# %%
status_codes = ['381', '364', '365']

schemes = [
    "*Bond Emerging Markets Global HC",
    "Bond Convertibles US",
    "Bond Emerging Mkts Global LC",
    "Bond Global High Yield USD",
    "Bond Global USD",
    "Bond USD",
    "Bond USD Corporates",
    "Bond USD Government",
    "Bond USD Government Short Term",
    "Bond USD High Yield",
    "Bond USD Inflation Linked",
    "Bond USD Medium Term",
    "Bond USD Mortgages",
    "Bond USD Municipal",
    "Bond USD Municipal High Yield",
    "Bond USD Municipal MT",
    "Bond USD Municipal ST",
    "Bond USD Short Term",
    "Bond USD Tax Exempt Money Market",
    "Equity Asia Pacific ex Japan",
    "Equity Canada",
    "Equity China",
    "Equity Emerging Mkts Global",
    "Equity Emerging Mkts Latin Am",
    "Equity Europe",
    "Equity Frontier Markets",
    "Equity Global",
    "Equity Global Sm&Mid Cap",
    "Equity Global ex US",
    "Equity Global ex US Sm&Mid Cap",
    "Equity India",
    "Equity Israel Sm&Mid Cap",
    "Equity Sector Communication Services",
    "Equity Sector Consumer Discretionary",
    "Equity Sector Consumer Staples",
    "Equity Sector Energy",
    "Equity Sector Financials",
    "Equity Sector Gold&Prec Metals",
    "Equity Sector Healthcare",
    "Equity Sector Information Tech",
    "Equity Sector Real Est Global",
    "Equity Sector Real Est US",
    "Equity Sector Utilities",
    "Equity Theme - Alternative Energy",
    "Equity Theme - Infrastructure",
    "Equity Theme - Natural Resources",
    "Equity US",
    "Equity US Income",
    "Equity US Sm&Mid Cap",
    "*Mixed Asset USD Conservative",
    "Mixed Asset USD Aggressive",
    "Mixed Asset USD Bal - Global",
    "Mixed Asset USD Bal - US",
    "Mixed Asset USD Flex - Global",
    "Mixed Asset USD Flex - US",
    "Money Market USD",
]

all_funds = []

for status_code in status_codes:
    for scheme in schemes:
        skip = 0
        batch = 1000

        while (skip + batch) <= 10000:
            res = ld.content.search.Definition(
                view=ld.content.search.Views.FUND_QUOTES,
                filter=(
                    f"IssuerDomicileCountry eq 'USA' "
                    f"and AssetCategory eq 'OPF' "
                    f"and FundClassStatus eq '{status_code}' "
                    f"and FundClassCurrency eq 'USD' "
                    f"and IssueLipperGlobalSchemeName eq '{scheme}'"
                ),
                select=(
                    "DocumentTitle, RIC, FundClassLipperID, FundEntityLipperId, "
                    "FundClassCurrency, IssueLipperGlobalSchemeName, "
                    "FundClassStatus, FundClassStatusName"
                ),
                top=batch,
                skip=skip
            ).get_data()

            df_batch = res.data.df
            if df_batch.empty:
                break

            all_funds.append(df_batch)
            skip += batch
            print(f"✓ {status_code} | {scheme}: {skip} so far...")

            if len(df_batch) < batch:
                break

funds_df = pd.concat(all_funds).reset_index(drop=True)
funds_df = funds_df.drop_duplicates(subset='RIC')
print(f"\nTotal universe: {len(funds_df)} funds")
print(funds_df['FundClassStatusName'].value_counts())
funds_df.to_csv('fund_data/us_open_end_funds_full.csv', index=False)

# %%
funds_df = pd.read_csv('fund_data/us_open_end_funds_full.csv')
funds_filtered_2 = (
    funds_df[funds_df['RIC'].str.startswith('LP')]
    .dropna(subset=['FundEntityLipperId', 'IssueLipperGlobalSchemeName'])
    .copy()
)

funds_filtered_2['RIC_clean'] = funds_filtered_2['RIC'].str.replace(r'\^.*$', '', regex=True)

funds_filtered_2 = (
    funds_filtered_2
    .sort_values('RIC')
    .groupby('FundEntityLipperId')
    .head(2)
    .reset_index(drop=True)
)


print(f"Unique fund entities: {funds_filtered_2['FundEntityLipperId'].nunique()}")

# Check for duplicate RIC_clean
dupes = funds_filtered_2[funds_filtered_2.duplicated(subset='RIC_clean', keep=False)]
print(f"Duplicate RIC_clean entries: {len(dupes)}")


funds_filtered_2 = (
    funds_filtered_2
    .sort_values('RIC')  # sort so clean RIC (no suffix) is preferred over ^suffix
    .drop_duplicates(subset='RIC_clean', keep='first')
    .reset_index(drop=True)
)

rics = list(funds_filtered_2['RIC_clean'].dropna().unique())
print(f"RICs after dedup: {len(rics)}")
print(f"Unique fund entities: {funds_filtered_2['FundEntityLipperId'].nunique()}")


rics_df = (
    funds_filtered_2
    .dropna(subset=['RIC_clean'])
    .drop_duplicates(subset=['RIC_clean'])
    .reset_index(drop=True)
)

rics_df.to_csv("fund_data/Clean_funds_with_asset_class.csv")

# %%

BATCH_SIZE = 50
SAVE_EVERY = 500

# ============================================================
# DOWNLOAD TNA (QUARTERLY)
# ============================================================

all_tna = []

for i in range(0, len(rics), BATCH_SIZE):
    batch = rics[i:i+BATCH_SIZE]
    batch_num = i // BATCH_SIZE

    try:
        df = ld.get_history(
            universe=batch,
            fields=["TR.FundTotalNetAssets"],
            interval="quarterly",
            start="1990-01-01",
            end="2025-12-31"
        )
        all_tna.append(df)
    except Exception as e:
        print(f"  TNA Batch {batch_num} failed: {e}")
        continue

    if (i + BATCH_SIZE) % SAVE_EVERY == 0 or (i + BATCH_SIZE) >= len(rics):
        checkpoint_num = (i + BATCH_SIZE) // SAVE_EVERY
        pd.concat(all_tna[-SAVE_EVERY//BATCH_SIZE:]).to_csv(f'fund_data/tna_checkpoint_{checkpoint_num}.csv')
        print(f"✓ TNA checkpoint {checkpoint_num} ({i+BATCH_SIZE}/{len(rics)} RICs)")

    if batch_num % 10 == 0:
        print(f"TNA Progress: {i}/{len(rics)} ({i/len(rics)*100:.1f}%)")

tna_raw = pd.concat(all_tna)
tna_raw.to_csv('tna_raw_full.csv')
print(f"TNA complete: {len(tna_raw)} rows")



# ============================================================
# RECOVERY: reload from checkpoints if session crashes
# ============================================================

def reload_from_checkpoints(prefix):
    from pathlib import Path
    files = sorted(Path('.').glob(f'{prefix}_*.csv'))
    if not files:
        print(f"No checkpoint files found for {prefix}")
        return None
    dfs = [pd.read_csv(f, index_col=0, parse_dates=True) for f in files]
    print(f"Reloaded {len(files)} files for {prefix}")
    return pd.concat(dfs)

# Uncomment to reload after crash:
# tna_raw = reload_from_checkpoints('tna_checkpoint')
# nav_raw = reload_from_checkpoints('nav_checkpoint')

# %%

BATCH_SIZE = 50

for i in range(0, len(rics), BATCH_SIZE):
    batch = rics[i:i+BATCH_SIZE]
    batch_num = i // BATCH_SIZE
    batch_file = f'fund_data/return_batch_{batch_num}.csv'

    try:
        pd.read_csv(batch_file, nrows=1)
        continue  # skip if exists
    except:
        pass

    try:
        df = ld.get_data(
            batch,
            ["TR.FundRollingPerformance(RollTimeFrame=SI,Interval=Q,Curn=NATIVE).date",
             "TR.FundRollingPerformance(RollTimeFrame=SI,Interval=Q,Curn=NATIVE)"]
        )
        df.to_csv(batch_file)
    except Exception as e:
        print(f"  Return Batch {batch_num} failed: {e}")
        continue

    if batch_num % 10 == 0:
        print(f"Return Progress: {i}/{len(rics)} ({i/len(rics)*100:.1f}%)")

print("Returns complete")

# %%



