import pandas as pd
import numpy as np

"""
Functions used to preprocess the weather data.
"""

def remove_missing_values(csv_path : str) -> pd.DataFrame:
    """
    Remove columns with missing values.
    First the columns whose are empty. (100% NA).
    Secondly the columns with more than 25% of missings values, except 3 columns
    """

    # remove columns with 100% of missing values
    df = pd.read_csv(csv_path)
    df_no_empty = df.dropna(axis=1, how='all')
    removed_columns = set(df.columns) - set(df_no_empty.columns)
    print(f"{len(removed_columns)} deleted columns (100% empty).")

    # remove columns with > 25 % of missing values
    threshold = int(len(df) * 0.75)
    df_clean = df_no_empty.dropna(axis=1, thresh=threshold)
    removed_columns = set(df_no_empty.columns) - set(df_clean.columns)
    print(f"{len(removed_columns)} deleted columns (>25% empty).")

    return df_clean


def remove_bad_quality(df : pd.DataFrame) -> pd.DataFrame:
    """
    Each data point/column is assigned a quality code (e.g., T;QT):
        9: Filtered data (the data has passed the initial filters/checks)
        0: Protected data (the data has been definitively validated by the climatologist)
        1: validated data (the data has been validated by an automated check or by the climatologist)
        2: questionable data currently being verified (the data has been flagged as questionable by an automated check)
        
        We simply retain the valid data (with a quality code of 0, 1 or 9) and remove the columns with the quality code.
    """

    # Identify all data columns that have an associated “Q” column
    quality_columns = [c for c in df.columns if 'Q' + c in df.columns]

    for col in quality_columns:
        col_q = 'Q' + col
        
        # Define the mask for “bad” data (data that is not 0, 1, or 9)
        # Add fillna(2) to treat missing Q codes as “doubtful” as a precaution
        invalid = ~df[col_q].fillna(2).isin([0, 1, 9])
        
        # Clear the column if the quality is poor
        df.loc[invalid, col] = np.nan

    # Completely remove all columns starting with Q
    removed_columns = [c for c in df.columns if c.startswith('Q')]
    df_clean = df.drop(columns=removed_columns)

    return df_clean


def remove_unnecesserary_columns(df : pd.DataFrame) -> pd.DataFrame :
    """
    Remove the columns that are not relevant.
    """

    columns_to_remove = [
        "NUM_POSTE",
        "NOM_USUEL",
        "LAT",
        "LON",
        "ALTI",
        "HTN",
        "HTX",
        "HXI",
        "HXY",
        "HFXI3S",
        "HUN",
        "HUX",
        "PMER",
        "PSTAT",
        "PMERMIN",
        "CL",
        "VV",
        "DVV200",
        "WW",
        "W1",
        "W2",
        "STATUS_FXI3S",
        "STATUS_DXI3S"
    ]

    df_clean = df.drop(columns=columns_to_remove)

    return df_clean