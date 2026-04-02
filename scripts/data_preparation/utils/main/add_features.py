"""
Explainations:
This python script allow us to add new features to our main dataset, based on the existing ones. 
The new features are created by calling the different functions defined in the "add_features.py" file, and can be easily modified by changing the parameters of these functions.


There are 7 different types of features created in this script:
- Date related features: creation of new features based on the date (day, month, hour, day of the week, etc.) and cyclical encoding of these features.
- Lag features: creation of lag features based on different groupings (ex: par avion, par route, etc.) and differents lags (ex: 1 day, 1 week, 1 month, 6 months, 1 year).
- Rolling features: creation of rolling features based on different aggregations and temporal windows (ex: mean of the last 30 days, max of the last 90 days, etc.).
- Trend features: creation of trend features based on the ratio between a short window and a long window.
- Lagged rolling features: creation of features based on rolling statistics, but lagged in the time (ex: mean of the last 14 days from 1 year ago). => Allow to capture historical tendencies while avoiding data leakage.
- Cross feature : computation of cross feature. Allow the model to understand that some features may influence each other.
- Momentum features: Allow the model to understand 'the market evolution' meaning how a trend growth or decrease (with which ratio). It compares the same amount between diverses temporal diemension/scale. 



- Global ones ? No ! Using global stats may lead to data leakage. This is what we need to avoid absolutly ! 
        We can't use future data to make predictions !!!
"""


### Imports
import pandas as pd
import numpy as np 
from typing import cast
import numpy.typing as npt
import gc

import logging
logger = logging.getLogger(__name__)

### Configuration
TARGET = "NbPaxTotal"

# Column configurations
COLUMN_LIST_BASE = [
    "SysTerminal",
    "IdAircraftType",
    "FlightNumberNormalized", 
    "airlineOACICode",
    ]

# Statistics to calculate for the lag features
STATISTICS_LIST = ['mean', 'min', 'max', 'std', 'median']

# Lags configuration
CUSTOM_LAGS = {
    "1year": pd.DateOffset(years=1),
    "6months": pd.DateOffset(months=6),
    "3months": pd.DateOffset(months=3),
    "1month": pd.DateOffset(months=1),
    "1week": pd.DateOffset(weeks=1)
}

# Rolling configuration - values = number of days
ROLLING_CONFIG = {
    "week": 7,
    "month": 30,
    "quarter": 91,
    "semester": 182,
    "year": 365
}

# Trend features configuration
TREND_CONFIG = {
    ("7D", "14D"),  # Trend based on the ratio between the mean of the last 7 days and the mean of the last 14 days
    ("14D", "30D"), 
    ("30D", "91D"),  
    ("91D", "182D"), 
    ("182D", "365D") 
}

# Rolling lags windows configuration
ROLLING_LAGS_CONFIG = {
    "lag365_win28": {"lag": "365D", "window": "28D"}, # D-365  +/- 14 days
    "lag182_win20": {"lag": "182D", "window": "20D"}, # D-182  +/- 10 days
    "lag91_win20": {"lag": "91D", "window": "20D"}, # D-91  +/- 10 days
    "lag30_win14": {"lag": "30D", "window": "14D"}, # D-30  +/- 7 days
    "lag7_win6": {"lag": "7D", "window": "6D"} # D-7  +/- 3 days
}


### Functions definition
def date_columns_creation(df : pd.DataFrame) -> pd.DataFrame:
        """
        The function to add temporal features.
        It allow to slice the ScheduledDatetime feature into another set of features.

        Add of cyclical encoding using cosinus and sinus functions. It allows us to have a continuity between time cycle (example: 12 pm is close to 1 am)
        """
        df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime'])
        # Creation of date related features
        df["Year"] = pd.to_datetime(df['LTScheduledDatetime']).dt.year
        df['Semester'] = np.where(df['LTScheduledDatetime'].dt.month <= 6, 1, 2)
        df['Quarter'] = df['LTScheduledDatetime'].dt.quarter
        df["Month"] = pd.to_datetime(df['LTScheduledDatetime']).dt.month
        df["Day"] = pd.to_datetime(df['LTScheduledDatetime']).dt.day
        df["Hour"] = pd.to_datetime(df['LTScheduledDatetime']).dt.hour
        df["Minute"] = pd.to_datetime(df['LTScheduledDatetime']).dt.minute
        df["DayOfWeek"] = pd.to_datetime(df['LTScheduledDatetime']).dt.dayofweek
        df['Hour_Of_Week'] = df['LTScheduledDatetime'].dt.dayofweek * 24 + df['LTScheduledDatetime'].dt.hour

        # Cyclical encoding
        # the cyclical encoding for hours is here on a base of 60 minutes, but we can also do it on a base of 24h, or even 168h (hour of the week)
        for col, period in [('Minute', 60), ('Hour', 24), ('Month', 12), ('DayOfWeek', 7), ('Hour_Of_Week', 168)]:
            df[f'sin_{col}'] = np.sin(2 * np.pi * df[col] / period)
            df[f'cos_{col}'] = np.cos(2 * np.pi * df[col] / period)
            df = df.drop(columns=[col])

        return df



def add_lag_features(df: pd.DataFrame, group_cols: list, lags: dict) -> pd.DataFrame:
    """
    Lags features are brut values of previous date (1 day, 1 month, ...) and a group argument (ex: IdAircraftType).
    Example: Value calculated = 1 month before, the average NbPaxTotal group by IdAircraftType.
    No need of STATISTIC_LIST here => 'Retrieve the raw value for this specific group at that exact moment in the past.'
    
    Params:
    - df: DataFrame with features 'LTScheduledDatetime' & TARGET.
    - group_cols: features used for grouping and aggregation.
    - lags: Dict CUSTOM_LAGS
    """
    ### Small view & light computation (RAM friendly)
    # Naming convention
    prefix = "_".join(group_cols) if group_cols else "global"

    # Smaller version of the df
    merge_keys = group_cols + ['LTScheduledDatetime']
    temp_base = df[merge_keys + [TARGET]].copy()

    # Type cleaning
    for col in group_cols:
        temp_base[col] = temp_base[col].astype(str)
    
    temp_base['LTScheduledDatetime'] = pd.to_datetime(
        temp_base['LTScheduledDatetime'], utc=True
    ).dt.tz_localize(None).dt.floor('min')
    merge_keys = group_cols + ['LTScheduledDatetime']

    # stat comutation
    aggregated_base = temp_base.groupby(merge_keys)[TARGET].mean().reset_index()

    # Immediate RAM cleaning
    del temp_base
    gc.collect()


    ### Lag columns generation using mapping to avoid memory issues
    # Create dictionary mappings for each lag feature
    lag_mappings = {}
    
    for lag_name, offset in lags.items():
        new_col_name = f"{prefix}_lag_{lag_name}_raw"
        
        # Create lag dataframe
        lag_df = aggregated_base[merge_keys + [TARGET]].copy()
        lag_df['LTScheduledDatetime'] = lag_df['LTScheduledDatetime'] + offset
        lag_df = lag_df.rename(columns={TARGET: new_col_name})
        lag_df[new_col_name] = lag_df[new_col_name].astype('float32')
        
        # Create a mapping dictionary (more memory-efficient than merge)
        lag_df_dict = lag_df.set_index(merge_keys)[new_col_name].to_dict()
        lag_mappings[new_col_name] = lag_df_dict
        
        del lag_df
        gc.collect()

    ### Apply lag features using mapping (avoids Cartesian product)
    for new_col_name, lag_dict in lag_mappings.items():
        # Create tuple keys for lookup
        if group_cols:
            df[new_col_name] = df.apply(
                lambda row: lag_dict.get(
                    tuple([row[col] for col in group_cols] + [pd.to_datetime(row['LTScheduledDatetime']).floor('min')]),
                    np.nan
                ),
                axis=1
            ).astype('float32')
        else:
            df[new_col_name] = df.apply(
                lambda row: lag_dict.get(
                    (pd.to_datetime(row['LTScheduledDatetime']).floor('min'),),
                    np.nan
                ),
                axis=1
            ).astype('float32')
    
    # RAM cleaning
    del aggregated_base
    del lag_mappings
    gc.collect()

    return df



def add_rolling_features(df: pd.DataFrame, group_cols: list, windows: dict) -> pd.DataFrame:
    """
    Rolling features calculation using slicing windows. Recent mobile statistics.
    
    Params:
    - df: DataFrame with features 'LTScheduledDatetime' & TARGET.
    - group_cols: features used for grouping and aggregation.
    - windows: Dict ROLLING_CONFIG
    """   
    # Smaller version of the df
    merge_keys = group_cols + ['LTScheduledDatetime']
    df_indexed = df[merge_keys + [TARGET]].copy()

    # naming convention
    prefix_base = "_".join(group_cols) if group_cols else "global"
    
    for name, window_size in windows.items():
        # Distinction for the global case
        if group_cols:
            rolling_obj = df_indexed.groupby(group_cols)[TARGET]
        else:
            rolling_obj = df_indexed[TARGET]
        
        # Shift to avoid data leakage (using data at 't' - handle by "closed" = "left" - and 't-1' to make prediction for 't')
        shifted_target = rolling_obj.shift(1)
        
        # Computation object
        rolling_group = shifted_target.rolling(
            window=window_size, 
            closed='left', 
            min_periods=1
        )
        
        # name
        prefix = f"{prefix_base}_rolling_{name}"
        
        # Main loop: stats computation
        # linear complexity -> RAM friendly
        for stat in STATISTICS_LIST:
            col_name = f"{prefix}_{stat}"
            
            # Computation
            res = rolling_group.agg(stat)
            
            # using float32 (RAM) -> df is sorted 
            df[col_name] = res.values.astype('float32')
            
            # RAM cleaning
            del res
            gc.collect()

    # RAM cleaning
    del df_indexed
    gc.collect()
    
    return df



def add_trend_features(df: pd.DataFrame, group_cols: list, short_win: str = "7D", long_win: str = "30D") -> pd.DataFrame:
    """
    Calculation of trend ratio (Trend = short_win / long_win) for every stats.

    Params:
    - df: DataFrame with features 'LTScheduledDatetime' & TARGET.
    - group_cols: features used for grouping and aggregation.
    - short_win: the length of the shortest window
    - long_win: the length of the longest window

    """
    # Create a copy and add an index column to track original positions
    df_copy = df.copy()
    df_copy['_row_idx'] = range(len(df_copy))
    
    # Sort keys
    sort_keys = group_cols + ['LTScheduledDatetime']
    
    # Prepare data for calculation  
    df_calc = df_copy[group_cols + ['LTScheduledDatetime', TARGET, '_row_idx']].copy()
    df_calc = df_calc.sort_values(by=sort_keys).reset_index(drop=True)
    
    # Naming convention
    prefix_base = "_".join(group_cols) if group_cols else "global"
    
    # Store all trend features
    trend_features = {}
    
    if group_cols:
        # Calculate trends for each group
        for _, group in df_calc.groupby(group_cols, sort=False):
            group = group.set_index('LTScheduledDatetime')
            shifted = group[TARGET].shift(1)
            
            short_roll = shifted.rolling(window=short_win, closed='left', min_periods=1)
            long_roll = shifted.rolling(window=long_win, closed='left', min_periods=1)
            
            for stat in STATISTICS_LIST:
                col_name = f"{prefix_base}_trend_{short_win}_vs_{long_win}_{stat}"
                
                s_values = short_roll.agg(stat).values
                l_values = long_roll.agg(stat).values
                
                ratio = np.divide(
                    s_values, 
                    l_values, 
                    out=np.ones_like(s_values, dtype='float32'), 
                    where=(l_values > 0) & (l_values != np.nan)
                )
                
                # Store with row indices as keys
                for idx, val in zip(group['_row_idx'].values, ratio):
                    if col_name not in trend_features:
                        trend_features[col_name] = {}
                    trend_features[col_name][idx] = val
    else:
        df_calc = df_calc.set_index('LTScheduledDatetime')
        shifted = df_calc[TARGET].shift(1)
        
        short_roll = shifted.rolling(window=short_win, closed='left', min_periods=1)
        long_roll = shifted.rolling(window=long_win, closed='left', min_periods=1)
        
        for stat in STATISTICS_LIST:
            col_name = f"{prefix_base}_trend_{short_win}_vs_{long_win}_{stat}"
            
            s_values = short_roll.agg(stat).values
            l_values = long_roll.agg(stat).values
            
            ratio = np.divide(
                s_values, 
                l_values, 
                out=np.ones_like(s_values, dtype='float32'), 
                where=(l_values > 0) & (l_values != np.nan)
            )
            
            for idx, val in zip(df_calc['_row_idx'].values, ratio):
                if col_name not in trend_features:
                    trend_features[col_name] = {}
                trend_features[col_name][idx] = val
    
    # Add trend features to the original dataframe
    for col_name, values_dict in trend_features.items():
        df[col_name] = [values_dict.get(i, np.nan) for i in range(len(df))]
    
    # RAM cleaning
    del df_copy, df_calc
    gc.collect()
    
    return df



def add_lagged_rolling_features(df: pd.DataFrame, group_cols: list, lag: str, window: str, new_col_name: str) -> pd.DataFrame:
    """
    Compute historical rolling stats (lagged and windowed) using mapping to avoid memory issues.
    """
    # Prepare base data
    merge_keys = group_cols + ['LTScheduledDatetime']
    temp_df = df[merge_keys + [TARGET]].copy()
    temp_df['LTScheduledDatetime'] = pd.to_datetime(temp_df['LTScheduledDatetime']).dt.floor('min')
    
    # Type cleaning for group columns
    for col in group_cols:
        temp_df[col] = temp_df[col].astype(str)
    
    # Initial aggregation (1 row = 1 datetime per group)
    base_grouped = temp_df.groupby(merge_keys)[TARGET].mean().reset_index()
    base_grouped = base_grouped.sort_values(by=merge_keys).reset_index(drop=True)
    
    # Temporal index for rolling
    base_indexed = base_grouped.set_index('LTScheduledDatetime')
    
    # Global case distinction - creation of the rolling object
    if group_cols:
        rolling_obj = base_indexed.groupby(group_cols)[TARGET]
    else:
        rolling_obj = base_indexed[TARGET]
        
    # Naming convention
    prefix = new_col_name if new_col_name else ("_".join(group_cols) if group_cols else "global")
    
    # Compute stats using mapping approach
    stat_mappings = {}
    
    for stat in STATISTICS_LIST:
        # Compute centered rolling stats
        stat_series = rolling_obj.rolling(window=window, min_periods=1, center=True).agg(stat)
        
        col_name = f"{prefix}_{stat}"
        current_stat_df = stat_series.reset_index()
        
        # Apply lag to datetime
        current_stat_df['LTScheduledDatetime'] = current_stat_df['LTScheduledDatetime'] + pd.to_timedelta(lag)
        
        # Convert to float32 and rename
        current_stat_df[TARGET] = current_stat_df[TARGET].astype('float32')
        current_stat_df.rename(columns={TARGET: col_name}, inplace=True)
        
        # Create mapping dictionary
        stat_dict = current_stat_df.set_index(merge_keys)[col_name].to_dict()
        stat_mappings[col_name] = stat_dict
        
        del stat_series
        del current_stat_df
        gc.collect()
    
    # Apply lagged rolling features using mapping (avoids Cartesian product)
    df = df.copy()
    df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime']).dt.floor('min')
    for col in group_cols:
        df[col] = df[col].astype(str)
    
    # Drop existing columns to avoid duplicates
    cols_to_drop = [c for c in stat_mappings.keys() if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # Apply features using mapping for each stat
    for col_name, stat_dict in stat_mappings.items():
        if group_cols:
            df[col_name] = df.apply(
                lambda row: stat_dict.get(
                    tuple([row[col] for col in group_cols] + [pd.to_datetime(row['LTScheduledDatetime']).floor('min')]),
                    np.nan
                ),
                axis=1
            ).astype('float32')
        else:
            df[col_name] = df.apply(
                lambda row: stat_dict.get(
                    (pd.to_datetime(row['LTScheduledDatetime']).floor('min'),),
                    np.nan
                ),
                axis=1
            ).astype('float32')
    
    del temp_df, base_grouped, base_indexed
    del stat_mappings
    gc.collect()
    
    return df



def add_interaction_features(df: pd.DataFrame, base_col: str, feature_pattern: str, suffix: str = "x") -> pd.DataFrame:
    """
    To compute the interaction between features --> to create cross_features : 2 distinct features may influence each other. 

    Params:
    - df: DataFrame with features 'LTScheduledDatetime' & TARGET.
    - base_col: the base column used to compute the interaction (numeric * numeric)
    - feature_pattern : a 'str' pattern to select features to mix with the base_col (here 'mean')
    - suffix : a 'str' indication for the feature name construction.
    """
    # columns identification to compute the interaction.
    target_cols = [c for c in df.columns if feature_pattern in str(c) and c != base_col]
    
    if not target_cols:
        return df

    new_interactions = {}
    
    # Extract the base_col, using flatten to avoid error. (ensure 1D vector)
    base_vals = cast(npt.NDArray[np.float32], df[base_col].values.astype(np.float32)).flatten()

    for col in target_cols:
        new_col_name = f"INT_{base_col}_{suffix}_{col}"
        
        # targets col extraction to 1D vectors (using the pattern)
        feat_vals = cast(npt.NDArray[np.float32], df[col].values.astype(np.float32))
        
        # Security about size of the vectors
        if feat_vals.ndim > 1:
            feat_vals = feat_vals.flatten()
        
        # Interaction computation (simple multiplication)
        interaction_values = base_vals * feat_vals
        
        # Adding a new feature in the dataframe
        df[new_col_name] = interaction_values.astype('float32')
    
    return df


def add_momentum_features(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Momentum features allow the model to understand 'the market evolution'.
    It compares the same amount between different temporal dimensions/scales.
    Example: ratio of (mean of last 7 days) / (mean of last 30 days)
    """
    # Naming convention
    prefix_base = "_".join(group_cols) if group_cols else "global"
    
    # Only create momentum if we have rolling features
    pattern = f"{prefix_base}_rolling_"
    rolling_cols = [c for c in df.columns if pattern in c and "_mean" in c]
    
    if len(rolling_cols) < 2:
        return df
    
    # Extract window names from column names
    windows = []
    for col in rolling_cols:
        # Extract window name: "prefix_rolling_WINDOW_mean"
        parts = col.split('_')
        for i, part in enumerate(parts):
            if part == 'rolling' and i + 1 < len(parts):
                windows.append(parts[i + 1])
                break
    
    windows = list(set(windows))
    
    # Create momentum features comparing pairs of windows
    momentum_config = [
        ("week", "month"),
        ("month", "quarter"),
        ("quarter", "semester"),
        ("semester", "year")
    ]
    
    for short_win, long_win in momentum_config:
        short_col = f"{prefix_base}_rolling_{short_win}_mean"
        long_col = f"{prefix_base}_rolling_{long_win}_mean"
        
        if short_col in df.columns and long_col in df.columns:
            new_col_name = f"{prefix_base}_momentum_{short_win}_vs_{long_win}"
            df[new_col_name] = (df[short_col] / df[long_col].replace(0, np.nan)).astype('float32')
    
    return df



def add_features(df):
    """
    Main function to add all features to the dataset.
    This orchestrates all feature engineering functions.
    """
    
    # 1. Date related features
    logger.info("Add of date related features")
    df = date_columns_creation(df)
    
    # 2. Lag, rolling, trend, lagged rolling, and interaction features for each grouping level
    ITERATION_GROUPS = [[]] + [[col] for col in COLUMN_LIST_BASE]
    for group in ITERATION_GROUPS:
        group_desc = "_".join(group) if group else "global"
        logger.info(f"Processing features for group: {group_desc}")

            
        # Sort, chronological order
        df = df.sort_values(by=group + ['LTScheduledDatetime']).reset_index(drop=True)

        ### Lag features
        df = add_lag_features(df, group_cols=group, lags=CUSTOM_LAGS)

        ### Rolling features (recent rolling stats)
        df = add_rolling_features(df, group_cols=group, windows=ROLLING_CONFIG)

        ### Lagged rolling features (historic rolling stats)
        for config_name, config in ROLLING_LAGS_CONFIG.items():
            current_new_col_name = f"{group_desc}_{config_name}"
            df = add_lagged_rolling_features(
                df=df,
                group_cols=group,
                lag=config["lag"],
                window=config["window"],
                new_col_name=current_new_col_name
            )
        
        ### Momentum features
        df = add_momentum_features(df=df, group_cols=group)

    logger.info("Dedicated features added (Global + Categorical)")

    # 3. Interaction features (cross features) - use rolling mean features
    df = add_interaction_features(df=df, base_col="NbOfSeats", feature_pattern="_mean")
    logger.info("Interaction features added")
    
    # RAM cleaning
    gc.collect()
    
    return df





### Test purposes
# if __name__=="__main__":
#     data = pd.read_csv('data/main.csv')
#     data = data.loc[:1000, :]
#     df = add_features(df=data)
#     df.to_csv('test.csv', encoding='utf-8')