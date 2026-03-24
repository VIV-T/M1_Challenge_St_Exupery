# ================================
#  FULL DATA PROFILING PIPELINE (STRUCTURED)
# ================================

# ============================================
print("\n STEP 1: IMPORTING LIBRARIES")
print("What: Importing required Python libraries")
print("Why: Needed for data manipulation, numerical computation, and BigQuery access")
print("How: Using pandas, numpy, and google.cloud.bigquery\n")

import os
import pandas as pd
import numpy as np
from google.cloud import bigquery
from IPython.display import display
from google.colab import files


# ============================================
print("\n STEP 2: CONFIGURATION")
print("What: Define project parameters and dataset configuration")
print("Why: Centralizes configuration for easy modification")
print("How: Set project ID, dataset, table, credentials, and sample size\n")

PROJECT_ID = "va-sdh-adl-staging"
DATASET_ID = "aero_insa"
TABLE_ID = "mouvements_aero_insa"
SERVICE_ACCOUNT_KEY_PATH = "/content/va-sdh-adl-staging.json"

SAMPLE_SIZE = 100000


# ============================================
print("\n STEP 3: AUTHENTICATION & CONNECTION")
print("What: Authenticate and connect to BigQuery")
print("Why: Required to query remote dataset securely")
print("How: Using service account JSON credentials\n")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_PATH
client = bigquery.Client(project=PROJECT_ID)

table_ref = f"`{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"


# ============================================
print("\n STEP 4: DATA LOADING")
print("What: Load a sample of data from BigQuery")
print("Why: Enables fast exploration without loading full dataset")
print("How: SQL query with LIMIT and safe API fallback\n")

query = f"""
SELECT *
FROM {table_ref}
LIMIT {SAMPLE_SIZE}
"""

print(f" Loading sample ({SAMPLE_SIZE} rows)...")
df = client.query(query).to_dataframe(create_bqstorage_client=False)
print(" Data loaded.\n")


# ============================================
print("\n STEP 5: BASIC DATA OVERVIEW")
print("What: Inspect dataset shape and memory usage")
print("Why: Understand dataset scale and computational cost")
print("How: Using pandas shape and memory_usage\n")

print(f"Shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

print("\nColumn types:")
display(df.dtypes.value_counts())


# ============================================
print("\n STEP 6: PREVIEW DATA")
print("What: Display first rows of dataset")
print("Why: Quickly understand structure and content")
print("How: Using df.head()\n")

display(df.head(10))


# ============================================
print("\n STEP 7: COLUMN TYPE DETECTION")
print("What: Identify numerical, categorical, datetime, boolean columns")
print("Why: Required for feature engineering and modeling strategy")
print("How: Using pandas select_dtypes\n")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

print(f"Numerical: {len(num_cols)}")
print(f"Categorical: {len(cat_cols)}")
print(f"Datetime: {len(datetime_cols)}")
print(f"Boolean: {len(bool_cols)}")


# ============================================
print("\n STEP 8: MISSING VALUES ANALYSIS")
print("What: Analyze missing data distribution")
print("Why: Critical for data cleaning and imputation strategies")
print("How: Compute missing counts and percentages\n")

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct
}).sort_values(by="missing_pct", ascending=False)

display(missing_df.head(30))


# ============================================
print("\n STEP 9: NUMERICAL STATISTICS")
print("What: Compute statistical summary of numerical features")
print("Why: Understand distributions, skewness, and scaling needs")
print("How: Using describe(), skew(), kurtosis()\n")

if len(num_cols) > 0:
    num_stats = df[num_cols].describe().T
    num_stats["skew"] = df[num_cols].skew()
    num_stats["kurtosis"] = df[num_cols].kurtosis()
    display(num_stats.head(30))


# ============================================
print("\n STEP 10: CORRELATION ANALYSIS")
print("What: Identify relationships between numerical variables")
print("Why: Detect redundancy and useful predictive signals")
print("How: Correlation matrix and sorting\n")

if len(num_cols) > 1:
    corr_matrix = df[num_cols].corr()
    corr_unstacked = corr_matrix.abs().unstack().sort_values(ascending=False)
    corr_unstacked = corr_unstacked[corr_unstacked < 1]
    display(corr_unstacked.head(20))


# ============================================
print("\n STEP 11: DUPLICATE DETECTION")
print("What: Identify duplicate rows")
print("Why: Duplicates can bias model training")
print("How: Using df.duplicated()\n")

duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")


# ============================================
print("\n STEP 12: OUTLIER DETECTION")
print("What: Detect extreme values in numerical features")
print("Why: Outliers can distort model training")
print("How: Using IQR method\n")

outlier_summary = {}

for col in num_cols[:30]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    outlier_summary[col] = len(outliers)

outlier_df = pd.DataFrame.from_dict(outlier_summary, orient='index', columns=["outlier_count"])
display(outlier_df.sort_values(by="outlier_count", ascending=False).head(20))


# ============================================
print("\n STEP 13: DATETIME ANALYSIS")
print("What: Analyze time coverage of dataset")
print("Why: Critical for time-based modeling and validation")
print("How: Extract min/max per datetime column\n")

for col in datetime_cols:
    print(f"{col}: {df[col].min()} → {df[col].max()}")


# ============================================
print("\n STEP 14: COLUMN SUMMARY TABLE")
print("What: Generate detailed per-column statistics")
print("Why: Central tool for feature engineering decisions")
print("How: Loop over columns with safe numeric handling\n")

summary_data = []

for col in df.columns:
    col_data = df[col]
    
    summary = {
        "column": col,
        "dtype": str(col_data.dtype),
        "n_unique": col_data.nunique(),
        "missing_count": col_data.isnull().sum(),
        "missing_pct": col_data.isnull().mean() * 100,
    }

    is_numeric = pd.api.types.is_numeric_dtype(col_data)
    is_bool = pd.api.types.is_bool_dtype(col_data)

    if is_numeric and not is_bool:
        try:
            clean_col = pd.to_numeric(col_data, errors='coerce')
            summary.update({
                "mean": clean_col.mean(),
                "std": clean_col.std(),
                "min": clean_col.min(),
                "25%": clean_col.quantile(0.25),
                "50%": clean_col.quantile(0.5),
                "75%": clean_col.quantile(0.75),
                "max": clean_col.max(),
                "skew": clean_col.skew(),
            })
        except:
            summary.update({k: None for k in ["mean","std","min","25%","50%","75%","max","skew"]})
    else:
        summary.update({k: None for k in ["mean","std","min","25%","50%","75%","max","skew"]})

    try:
        if col_data.dtype == "object" or col_data.dtype.name == "category":
            summary["top_values"] = str(col_data.value_counts().head(3).to_dict())
        else:
            summary["top_values"] = None
    except:
        summary["top_values"] = None

    summary_data.append(summary)

summary_df = pd.DataFrame(summary_data).sort_values(by="missing_pct", ascending=False)
display(summary_df.head(50))


# ============================================
print("\n STEP 15: DATASET SUMMARY")
print("What: Compute global dataset metrics")
print("Why: Provides high-level understanding for reporting")
print("How: Aggregate statistics\n")

dataset_summary = {
    "n_rows": len(df),
    "n_columns": df.shape[1],
    "total_missing_cells": df.isnull().sum().sum(),
    "missing_percentage_total": df.isnull().sum().sum() / (df.shape[0]*df.shape[1]) * 100,
    "n_numeric_cols": len(num_cols),
    "n_categorical_cols": len(cat_cols),
    "n_datetime_cols": len(datetime_cols),
    "n_duplicate_rows": duplicates,
}

dataset_summary_df = pd.DataFrame(dataset_summary.items(), columns=["metric", "value"])
display(dataset_summary_df)


# ============================================
print("\n STEP 16: FEATURE ENGINEERING FLAGS")
print("What: Flag problematic features")
print("Why: Identify risks for modeling (missing, low variance, high cardinality)")
print("How: Rule-based checks\n")

flags = []

for col in df.columns:
    n_unique = df[col].nunique()
    missing_pct = df[col].isnull().mean() * 100
    
    flags.append({
        "column": col,
        "high_missing (>30%)": missing_pct > 30,
        "low_variance (<=2 unique)": n_unique <= 2,
        "high_cardinality (>100)": n_unique > 100,
    })

flags_df = pd.DataFrame(flags)
display(flags_df.head(50))


# ============================================
print("\n STEP 17: SAVE DATA AND PROFILING OUTPUTS LOCALLY")
print("What: Persist raw sample and profiling tables to local CSV files")
print("Why: Enables offline inspection, reproducibility, and download")
print("How: Use pandas to_csv() method and provide download links in Colab\n")

# Save the raw loaded sample
df.to_csv("raw_dataset_sample.csv", index=False)
print(" Raw dataset saved locally as raw_dataset_sample.csv")

# Save summary tables
summary_df.to_csv("column_summary.csv", index=False)
dataset_summary_df.to_csv("dataset_summary.csv", index=False)
flags_df.to_csv("feature_flags.csv", index=False)
print(" Profiling outputs saved locally: column_summary.csv, dataset_summary.csv, feature_flags.csv")

# Optional: Provide download links for convenience in Colab
from google.colab import files
print("\nDownloading files...")
files.download("raw_dataset_sample.csv")
files.download("column_summary.csv")
files.download("dataset_summary.csv")
files.download("feature_flags.csv")
print(" Downloads initiated")
