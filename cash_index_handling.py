import pandas as pd
import math
import numpy as np
from datetime import date
import openpyxl
from pathlib import Path
from Private_Markets_handling  import load_table



# Set file path and sheet index
FILE_LA = Path("Liquid_Assets_2.xlsx")
SHEET_Cash = 3


Cash_df = load_table(FILE_LA, SHEET_Cash)

Cash_df["EUR003_Index"] = (Cash_df["EUR003_Index"] /100 / 360)
cash_quarter_ret_df = (1+ Cash_df["EUR003_Index"]).resample("QE").prod() -1

cash_quarter_ret_df.to_csv("Cash_quarterly_returns.csv")


print(cash_quarter_ret_df)