# =========================================
# v6 PREPROCESSING SCRIPT with SEMANTIC LAG NAMES
# =========================================
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar  # adjust if needed

# -----------------------------
# 1. LOAD RAW DATA
# -----------------------------
PATH_RAW = "/content/drive/MyDrive/final_dataset_with_lags.csv"
df = pd.read_csv(PATH_RAW, parse_dates=["UTCExternalDate"])
df = df.sort_values("UTCExternalDate").reset_index(drop=True)
print(" Loaded raw dataset:", df.shape)

# -----------------------------
# 2. TIME FEATURES
# -----------------------------
df["hour"] = df["UTCExternalDate"].dt.hour
df["dayofweek"] = df["UTCExternalDate"].dt.dayofweek
df["month"] = df["UTCExternalDate"].dt.month
df["quarter"] = df["UTCExternalDate"].dt.quarter
df["weekofyear"] = df["UTCExternalDate"].dt.isocalendar().week.astype(int)
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

# Optional: holidays (adjust country/calendar if needed)
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=df["UTCExternalDate"].min(), end=df["UTCExternalDate"].max())
df["is_holiday"] = df["UTCExternalDate"].isin(holidays).astype(int)

# -----------------------------
# 3. LAG FEATURES (past traffic)
# -----------------------------
TARGET = "NbPaxTotal"
SEAT_COL = "NbOfSeats"
LAGS = [1, 3, 7, 14]  # days
ROLL_WINDOWS = [7, 14, 30]

for lag in LAGS:
    # original lag
    df[f"{TARGET}_lag_{lag}"] = df[TARGET].shift(lag)
    df[f"{SEAT_COL}_lag_{lag}"] = df[SEAT_COL].shift(lag)
    # differences
    df[f"{TARGET}_diff_lag_{lag}"] = df[TARGET] - df[f"{TARGET}_lag_{lag}"]

    # semantic renaming
    if lag == 1:
        df.rename(columns={
            f"{TARGET}_lag_{lag}": f"{TARGET}_prev_day",
            f"{SEAT_COL}_lag_{lag}": f"{SEAT_COL}_prev_day",
            f"{TARGET}_diff_lag_{lag}": f"{TARGET}_change_since_prev_day"
        }, inplace=True)
    elif lag == 3:
        df.rename(columns={
            f"{TARGET}_lag_{lag}": f"{TARGET}_prev_3days",
            f"{SEAT_COL}_lag_{lag}": f"{SEAT_COL}_prev_3days",
            f"{TARGET}_diff_lag_{lag}": f"{TARGET}_change_since_3days_ago"
        }, inplace=True)
    elif lag == 7:
        df.rename(columns={
            f"{TARGET}_lag_{lag}": f"{TARGET}_prev_week",
            f"{SEAT_COL}_lag_{lag}": f"{SEAT_COL}_prev_week",
            f"{TARGET}_diff_lag_{lag}": f"{TARGET}_change_since_prev_week"
        }, inplace=True)
    elif lag == 14:
        df.rename(columns={
            f"{TARGET}_lag_{lag}": f"{TARGET}_prev_2weeks",
            f"{SEAT_COL}_lag_{lag}": f"{SEAT_COL}_prev_2weeks",
            f"{TARGET}_diff_lag_{lag}": f"{TARGET}_change_since_2weeks_ago"
        }, inplace=True)

# rolling features with semantic names
for window in ROLL_WINDOWS:
    df[f"{TARGET}_rolling_mean_{window}"] = df[TARGET].shift(1).rolling(window).mean()
    df[f"{TARGET}_rolling_std_{window}"] = df[TARGET].shift(1).rolling(window).std()
    df[f"{SEAT_COL}_rolling_mean_{window}"] = df[SEAT_COL].shift(1).rolling(window).mean()
    df[f"{SEAT_COL}_rolling_std_{window}"] = df[SEAT_COL].shift(1).rolling(window).std()

    df.rename(columns={
        f"{TARGET}_rolling_mean_{window}": f"{TARGET}_rolling_mean_{window}d",
        f"{TARGET}_rolling_std_{window}": f"{TARGET}_rolling_std_{window}d",
        f"{SEAT_COL}_rolling_mean_{window}": f"{SEAT_COL}_rolling_mean_{window}d",
        f"{SEAT_COL}_rolling_std_{window}": f"{SEAT_COL}_rolling_std_{window}d",
    }, inplace=True)

# -----------------------------
# 4. HANDLE CATEGORICALS
# -----------------------------
cat_cols = [
    "AirportOrigin",
    "airlineOACICode",
    "Direction",
    "ScheduleType",
    "route"
]

for col in cat_cols:
    df[col] = df[col].astype("category")

# -----------------------------
# 5. DROP CONSTANT COLUMNS
# -----------------------------
const_cols = [c for c in df.columns if df[c].nunique() <= 1]
if const_cols:
    print("🗑 Dropping constant columns:", const_cols)
    df.drop(columns=const_cols, inplace=True)

# -----------------------------
# 6. HANDLE MISSING VALUES
# -----------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# -----------------------------
# 7. SAVE FINAL PREPROCESSED CSV
# -----------------------------
PATH_PRE = "/content/drive/MyDrive/final_dataset_preprocessed_v6_semantic.csv"
df.to_csv(PATH_PRE, index=False)
print(" Preprocessed dataset saved:", PATH_PRE)

# -----------------------------
# 8. DISPLAY SHAPE & BASIC STATS
# -----------------------------
print(" Final shape:", df.shape)
print("\n Column info:")
print(df.info())
print("\n Numeric columns statistics:")
print(df[numeric_cols].describe().T)
print("\n Categorical columns unique values:")
for col in cat_cols:
    if col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")
