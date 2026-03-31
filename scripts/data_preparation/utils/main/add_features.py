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
from datetime import date
from typing import cast
import numpy.typing as npt
import gc



### Configuration
TARGET = "NbPaxTotal"

# Column configurations
COLUMN_LIST_BASE = [
    "FlightNumberNormalized", 
    "airlineOACICode",
    "IdAircraftType",
    "SysTerminal", 
    # Direction 
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

# Rolling configuration
ROLLING_CONFIG = {
    "week": "7D",
    "month": "30D",
    "quarter": "91D",
    "semester": "182D",
    "year": "365D"
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
    Lags features are statistics calculated using a lag (1 day, 1 month, ...) and a group argument (ex: IdAircraftType).
    Example: Value calculated = 1 month before, the average NbPaxTotal group by IdAircraftType.

    Params:
    - df: DataFrame with features 'LTScheduledDatetime' & TARGET.
    - group_cols: features used for grouping and aggregation.
    - lags: Dict CUSTOM_LAGS
    """
    # Type cleaning
    for col in group_cols:
        df[col] = df[col].astype(str)
    df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime'], utc=True).dt.tz_localize(None).dt.floor('min')

    # Statistics calculation using STATISTICS_LIST
    for lag_name, offset in lags.items():
        temp_df = (
            df.groupby(group_cols + ['LTScheduledDatetime'])
            .agg({TARGET: STATISTICS_LIST})
            .reset_index()
        )

        # Avoid pandas multi-index.
        temp_df.columns = group_cols + ['LTScheduledDatetime'] + STATISTICS_LIST

        # Lag application
        temp_df['LTScheduledDatetime'] = temp_df['LTScheduledDatetime'] + offset
        
        # Re-aggregate, using min, the calculated statistics. Using 'mean' allow us to smooth the aggregation of the calculated statistics.
        temp_df = (
            temp_df.groupby(group_cols + ['LTScheduledDatetime'])
            .agg({s: 'mean' for s in STATISTICS_LIST})
            .reset_index()
        )

        # Rename the column associated to calculated stats. 
        rename_dict = {
            s: f"{'_'.join(group_cols)}_lag_{lag_name}_{s}" 
            for s in STATISTICS_LIST
        }
        temp_df = temp_df.rename(columns=rename_dict)

        # Merge with the main df : allow to retrieve the calculated stats in the main df.
        df = pd.merge(
            df, 
            temp_df, 
            on=group_cols + ['LTScheduledDatetime'], 
            how='left'
        )
        
        # RAM cleaning
        del temp_df
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
    # Sort, chornological order
    df = df.sort_values(by=group_cols + ['LTScheduledDatetime']).reset_index(drop=True)
    
    # Tempral index + df.copy
    df_indexed = df.set_index('LTScheduledDatetime').sort_index()
    
    
    for name, window_size in windows.items():
        # Group and slicing windows creation
        # Most important argument
        # closed='left' is crucial to avoid Data Leakage (use of unknow future feature = cheating)
        rolling_group = (
            df_indexed.groupby(group_cols)[TARGET]
            .shift(1)   # the data of yesterday have to be used for tiomorrow's predictions (not today's data that aren't available yet)
            .rolling(window=window_size, closed='left', min_periods=1)
        )
        
        # Stats computation using .agg() method
        temp_rolling_stats = rolling_group.agg(STATISTICS_LIST).reset_index()
        
        # renamming of columns
        prefix = f"{'_'.join(group_cols)}_rolling_{name}"
        
        for stat in STATISTICS_LIST:
            col_name = f"{prefix}_{stat}"
            # Use of .value to keep the same order of the original sorted df 
            df[col_name] = temp_rolling_stats[stat].values
            
            # float32 = RAM optimization
            df[col_name] = df[col_name].astype('float32')

        # RAM cleaning
        del temp_rolling_stats
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
    # Sort + temporal indexation
    df = df.sort_values(by=group_cols + ['LTScheduledDatetime']).reset_index(drop=True)
    df_indexed = df.set_index('LTScheduledDatetime').sort_index()
        
    # Aggration calculation (short and long window)
    # same as add_rolling_features : close = 'left' + .shift(1) --> avoid to use unknown data and Data leakage.
    short_rolling = (
        df_indexed.groupby(group_cols)[TARGET]
        .shift(1)
        .rolling(window=short_win, closed='left', min_periods=1)
        .agg(STATISTICS_LIST)
        .reset_index()
    )
    
    long_rolling = (
        df_indexed.groupby(group_cols)[TARGET]
        .shift(1)
        .rolling(window=long_win, closed='left', min_periods=1)
        .agg(STATISTICS_LIST)
        .reset_index()
    )
    
    # New features creation
    new_features = {}
    prefix = f"{'_'.join(group_cols)}_trend_{short_win}_vs_{long_win}"
    
    for stat in STATISTICS_LIST:
        s_values = short_rolling[stat].values
        l_values = long_rolling[stat].values
        
        col_name = f"{prefix}_{stat}"
        
        # Ratio calculation
        new_features[col_name] = np.divide(
            s_values, 
            l_values, 
            out=np.ones_like(s_values, dtype='float32'), 
            where=l_values > 0
        ).astype('float32')

    # Add features to the main df using merging 
    new_cols_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)

    # cleaning
    del short_rolling, long_rolling, new_features, new_cols_df
    gc.collect()
    
    return df



def add_lagged_rolling_features(df: pd.DataFrame, group_cols: list, lag: str, window: str, new_col_name: str) -> pd.DataFrame:
    """
    Calculation of mobile stats. Lag those stats. 
    => Allow to compute historical trend.

    Params:
    - df: DataFrame with features 'LTScheduledDatetime' & TARGET.
    - group_cols: features used for grouping and aggregation.
    - lag: the date lag to center the window
    - window: the number of day of aggregation to compute statistics.
    - new_col_name : the new feature name created.

    """
    # initialization - same as the previous functions
    df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime']).dt.floor('min')
    df = df.sort_values(by=group_cols + ['LTScheduledDatetime']).reset_index(drop=True)
    df_temp = df.set_index('LTScheduledDatetime').sort_index()
    
    
    #Stats computation using aggregation
    # center=True allow to center the window: +/- (window/2) around the lag.
    # note that the stats is compute for each row, the lag is done only after this step.
    rolling_gen = (
        df_temp.groupby(group_cols)[TARGET]
        .rolling(window=window, min_periods=1, center=True)
        .agg(STATISTICS_LIST)
        .reset_index()
    )
    
    # Lag - date lag using the lag param.
    rolling_gen['LTScheduledDatetime'] = rolling_gen['LTScheduledDatetime'] + pd.to_timedelta(lag)
    
    # Security : re-aggregation if dates are fusionning.
    stats_df = (
        rolling_gen.groupby(group_cols + ['LTScheduledDatetime'])[STATISTICS_LIST]
        .mean()
        .reset_index()
    )
    
    # Rename features
    rename_dict = {s: f"{new_col_name}_{s}" for s in STATISTICS_LIST}
    stats_df = stats_df.rename(columns=rename_dict)
    
    # RAM optimization
    for col in rename_dict.values():
        stats_df[col] = stats_df[col].astype('float32')
    
    # Merge with the main df.
    df = pd.merge(df, stats_df, on=group_cols + ['LTScheduledDatetime'], how='left')
    
    # RAM cleaning
    del rolling_gen, stats_df
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
            feat_vals = feat_vals[:, 0]
        else:
            feat_vals = feat_vals.flatten()

        # Computation (vect * vect)
        if base_vals.shape[0] == feat_vals.shape[0]:
            new_interactions[new_col_name] = base_vals * feat_vals
        else:
            print(f"Saut de la colonne {col} : dimensions incompatibles.")

    if not new_interactions:
        return df

    # df creation
    new_df = pd.DataFrame(new_interactions, index=df.index)
    
    # concatenation + index management 
    try:
        # If clean index
        df = pd.concat([df, new_df], axis=1)
    except ValueError:
        # If "duplicate labels", synchronization using 'reset_index'
        print(f"Index dupliqués détectés lors de l'interaction {feature_pattern}. Correction en cours...")
        df = df.reset_index(drop=True)
        new_df.index = df.index
        df = pd.concat([df, new_df], axis=1)
    
    return df



def add_momentum_features(df: pd.DataFrame, short_term_pattern: str, long_term_pattern: str, suffix: str = "div") -> pd.DataFrame:
    """
    Computation of dynamic ratio between 2 existing features.res existantes.
    Ex : short_term_pattern='rolling_7D', long_term_pattern='rolling_30D'

    Params:
    - df: DataFrame with features 'LTScheduledDatetime' & TARGET.
    - short_term_pattern: to identify the corresponding feature
    - long_term_pattern: to identify the corresponding feature
    - suffix : a 'str' indication for the feature name construction.

    """
    # column identification - same col, but with different temporal dimension.
    new_momentum = {}
    
    for stat in STATISTICS_LIST:
        col_short = [c for c in df.columns if short_term_pattern in c and c.endswith(f'_{stat}')]
        col_long = [c for c in df.columns if long_term_pattern in c and c.endswith(f'_{stat}')]
        
        # Iteration over correspondances
        for s_col in col_short:
            # Search for the long_pattern corresponding to the short one
            l_col = s_col.replace(short_term_pattern, long_term_pattern)
            
            if l_col in df.columns:
                new_col_name = f"MOM_{s_col}_{suffix}_{long_term_pattern}"
                
                from typing import cast
                import numpy as np
                import numpy.typing as npt

                # Explicit extractaction to a NDArray (type is important)
                l_values = cast(npt.NDArray[np.float32], df[l_col].values.astype(np.float32))
                s_values = cast(npt.NDArray[np.float32], df[s_col].values.astype(np.float32))

                condition_mask = l_values > 0

                # ratio calculation
                new_momentum[new_col_name] = np.divide(
                    s_values, 
                    l_values, 
                    out=np.ones_like(s_values), 
                    where=condition_mask
                )

    # Concatenation
    if new_momentum:
        new_df = pd.DataFrame(new_momentum, index=df.index)
        df = pd.concat([df, new_df], axis=1)
    else:
        print(f"No correspondances between {short_term_pattern} & {long_term_pattern}")
        
    return df



def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to add new features. 
    Note: Order is critical (Momentum must come after Rolling/Lag).
    """
    # Security : Delete duplicates index
    df = df.reset_index(drop=True)

    ### Date related features
    df = date_columns_creation(df=df)

    for col in COLUMN_LIST_BASE:
        ### Lag features
        df = add_lag_features(df, group_cols=[col], lags=CUSTOM_LAGS)

        ### Rolling features (recent rolling stats)
        df = add_rolling_features(df, group_cols=[col], windows=ROLLING_CONFIG)

        ### Lagged rolling features (historic rolling stats)
        for config_name, config in ROLLING_LAGS_CONFIG.items():
            df = add_lagged_rolling_features(
                df, 
                group_cols=[col], 
                lag=config["lag"], 
                window=config["window"], 
                new_col_name=f"{col}_{config_name}"
            )

        ### Trend features
        for short_win, long_win in TREND_CONFIG:
            df = add_trend_features(df, group_cols=[col], short_win=short_win, long_win=long_win)

        
        ### Momentum features
        # recent momentum
        df = add_momentum_features(df, 
                                   short_term_pattern='rolling_week', 
                                   long_term_pattern='rolling_month', 
                                   suffix="accel")

        # middle term momentum
        df = add_momentum_features(df, 
                                   short_term_pattern='rolling_month', 
                                   long_term_pattern='rolling_quarter', 
                                   suffix="month_ratio")
        
        # long term momentum
        df = add_momentum_features(df, 
                                   short_term_pattern='rolling_quarter', 
                                   long_term_pattern='rolling_year', 
                                   suffix="season_ratio")


    ### 6. Interaction features ("NbSeats" * calculated means)
    df = add_interaction_features(df, base_col="NbOfSeats", feature_pattern="_mean")

    return df





### Test 
# if __name__ == "__main__":
#     df = pd.read_csv("data/main.csv")
#     df = add_features(df)
#     df.to_csv("data/main_preprocessed_new.csv", index=False)
