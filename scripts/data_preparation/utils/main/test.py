import pandas as pd
import numpy as np 
from datetime import date
from typing import cast
import numpy.typing as npt
import gc

TARGET = "NbPaxTotal"

# Configuration des colonnes de base
COLUMN_LIST_BASE = [
    "FlightNumberNormalized", 
    "airlineOACICode",
    "IdAircraftType",
    "SysTerminal", 
    "Direction"
]

STATISTICS_LIST = ['mean', 'min', 'max', 'std', 'median']

ROLLING_CONFIG = {
    "week": "7D",
    "month": "30D",
    "quarter": "91D",
    "semester": "182D",
    "year": "365D"
}

# MODIFICATION : Lags commencent à 2 jours pour la réalité opérationnelle (D-1 pour prédire D+1)
CUSTOM_LAGS = {
    "1year": pd.DateOffset(years=1),
    "6months": pd.DateOffset(months=6),
    "3months": pd.DateOffset(months=3),
    "1month": pd.DateOffset(months=1),
    "2days": pd.DateOffset(days=2) # Le lag 1D est supprimé car non disponible en prod
}

TREND_CONFIG = [
    ("7D", "14D"), 
    ("14D", "30D"), 
    ("30D", "91D"),  
    ("91D", "182D"), 
    ("182D", "365D") 
]

ROLLING_LAGS_CONFIG = {
    "lag365_win28": {"lag": "365D", "window": "28D"},
    "lag182_win20": {"lag": "182D", "window": "20D"},
    "lag91_win20": {"lag": "91D", "window": "20D"},
    "lag30_win14": {"lag": "30D", "window": "14D"},
    "lag7_win6": {"lag": "7D", "window": "6D"} 
}

def date_columns_creation(df: pd.DataFrame) -> pd.DataFrame:
    df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime'])
    df["Year"] = df['LTScheduledDatetime'].dt.year
    df['Semester'] = np.where(df['LTScheduledDatetime'].dt.month <= 6, 1, 2)
    df['Quarter'] = df['LTScheduledDatetime'].dt.quarter
    df["Month"] = df['LTScheduledDatetime'].dt.month
    df["Day"] = df['LTScheduledDatetime'].dt.day
    df["Hour"] = df['LTScheduledDatetime'].dt.hour
    df["Minute"] = df['LTScheduledDatetime'].dt.minute
    df["DayOfWeek"] = df['LTScheduledDatetime'].dt.dayofweek
    df['Hour_Of_Week'] = df['LTScheduledDatetime'].dt.dayofweek * 24 + df['LTScheduledDatetime'].dt.hour

    for col, period in [('Minute', 60), ('Hour', 24), ('Month', 12), ('DayOfWeek', 7), ('Hour_Of_Week', 168)]:
        df[f'sin_{col}'] = np.sin(2 * np.pi * df[col] / period).astype('float32')
        df[f'cos_{col}'] = np.cos(2 * np.pi * df[col] / period).astype('float32')
        df = df.drop(columns=[col])
    return df

def add_lag_features(df: pd.DataFrame, group_cols: list, lags: dict) -> pd.DataFrame:
    for col in group_cols:
        df[col] = df[col].astype(str)
    df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime']).dt.tz_localize(None).dt.floor('min')

    for lag_name, offset in lags.items():
        temp_df = df.groupby(group_cols + ['LTScheduledDatetime']).agg({TARGET: STATISTICS_LIST}).reset_index()
        temp_df.columns = group_cols + ['LTScheduledDatetime'] + STATISTICS_LIST
        
        # Décalage temporel
        temp_df['LTScheduledDatetime'] = temp_df['LTScheduledDatetime'] + offset
        
        temp_df = temp_df.groupby(group_cols + ['LTScheduledDatetime']).agg({s: 'mean' for s in STATISTICS_LIST}).reset_index()
        
        rename_dict = {s: f"{'_'.join(group_cols)}_lag_{lag_name}_{s}" for s in STATISTICS_LIST}
        temp_df = temp_df.rename(columns=rename_dict)
        
        for c in rename_dict.values():
            temp_df[c] = temp_df[c].astype('float32')

        df = pd.merge(df, temp_df, on=group_cols + ['LTScheduledDatetime'], how='left')
        del temp_df
        gc.collect()
    return df

def add_rolling_features(df: pd.DataFrame, group_cols: list, windows: dict) -> pd.DataFrame:
    # Tri indispensable pour la monotonicité
    df = df.sort_values(by=group_cols + ['LTScheduledDatetime']).reset_index(drop=True)
    df_indexed = df.set_index('LTScheduledDatetime')
    
    for name, window_size in windows.items():
        # SHIFT(1) + CLOSED='LEFT' pour la contrainte opérationnelle J-1 -> J+1
        rolling_group = (
            df_indexed.groupby(group_cols)[TARGET]
            .shift(1)
            .groupby(group_cols) # Re-grouper après le shift pour isoler les séries
            .rolling(window=window_size, closed='left', min_periods=1)
            .agg(STATISTICS_LIST)
            .reset_index()
        )
        
        prefix = f"{'_'.join(group_cols)}_rolling_{name}"
        for stat in STATISTICS_LIST:
            col_name = f"{prefix}_{stat}"
            df[col_name] = rolling_group[stat].values.astype('float32')

        del rolling_group
        gc.collect()
    return df

def add_trend_features(df: pd.DataFrame, group_cols: list, short_win: str, long_win: str) -> pd.DataFrame:
    df = df.sort_values(by=group_cols + ['LTScheduledDatetime']).reset_index(drop=True)
    df_indexed = df.set_index('LTScheduledDatetime')
    
    # Intégration du shift de sécurité
    short_roll = df_indexed.groupby(group_cols)[TARGET].shift(1).groupby(group_cols).rolling(window=short_win, closed='left', min_periods=1).mean().reset_index()
    long_roll = df_indexed.groupby(group_cols)[TARGET].shift(1).groupby(group_cols).rolling(window=long_win, closed='left', min_periods=1).mean().reset_index()
    
    col_name = f"{'_'.join(group_cols)}_trend_{short_win}_vs_{long_win}_mean"
    
    s_vals = short_roll[TARGET].values.astype('float32')
    l_vals = long_roll[TARGET].values.astype('float32')
    
    df[col_name] = np.divide(s_vals, l_vals, out=np.ones_like(s_vals), where=l_vals > 0).astype('float32')
    
    del short_roll, long_roll
    gc.collect()
    return df

def add_lagged_rolling_features(df: pd.DataFrame, group_cols: list, lag: str, window: str, new_col_name: str) -> pd.DataFrame:
    df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime']).dt.floor('min')
    df_temp = df.sort_values(by=group_cols + ['LTScheduledDatetime']).set_index('LTScheduledDatetime')
    
    rolling_gen = (
        df_temp.groupby(group_cols)[TARGET]
        .rolling(window=window, min_periods=1, center=True)
        .agg(STATISTICS_LIST)
        .reset_index()
    )
    
    rolling_gen['LTScheduledDatetime'] = rolling_gen['LTScheduledDatetime'] + pd.to_timedelta(lag)
    
    stats_df = rolling_gen.groupby(group_cols + ['LTScheduledDatetime']).mean().reset_index()
    rename_dict = {s: f"{new_col_name}_{s}" for s in STATISTICS_LIST}
    stats_df = stats_df.rename(columns=rename_dict)
    
    for col in rename_dict.values():
        stats_df[col] = stats_df[col].astype('float32')
    
    df = pd.merge(df, stats_df, on=group_cols + ['LTScheduledDatetime'], how='left')
    del rolling_gen, stats_df
    gc.collect()
    return df

def add_interaction_features(df: pd.DataFrame, base_col: str, feature_pattern: str, suffix: str = "x") -> pd.DataFrame:
    target_cols = [c for c in df.columns if feature_pattern in str(c) and c != base_col]
    if not target_cols: return df

    new_interactions = {}
    base_vals = cast(npt.NDArray[np.float32], df[base_col].values.astype(np.float32)).flatten()

    for col in target_cols:
        feat_vals = cast(npt.NDArray[np.float32], df[col].values.astype(np.float32))
        feat_vals = feat_vals[:, 0] if feat_vals.ndim > 1 else feat_vals.flatten()

        if base_vals.shape[0] == feat_vals.shape[0]:
            new_interactions[f"INT_{base_col}_{suffix}_{col}"] = base_vals * feat_vals

    if new_interactions:
        new_df = pd.DataFrame(new_interactions, index=df.index)
        df = pd.concat([df, new_df], axis=1)
    return df

def add_momentum_features(df: pd.DataFrame, short_term_pattern: str, long_term_pattern: str, suffix: str = "div") -> pd.DataFrame:
    new_momentum = {}
    for stat in STATISTICS_LIST:
        col_short = [c for c in df.columns if short_term_pattern in c and c.endswith(f'_{stat}')]
        for s_col in col_short:
            l_col = s_col.replace(short_term_pattern, long_term_pattern)
            if l_col in df.columns:
                s_vals = cast(npt.NDArray[np.float32], df[s_col].values.astype(np.float32))
                l_vals = cast(npt.NDArray[np.float32], df[l_col].values.astype(np.float32))
                
                new_momentum[f"MOM_{s_col}_{suffix}_{long_term_pattern}"] = np.divide(
                    s_vals, l_vals, out=np.ones_like(s_vals), where=l_vals > 0
                ).astype('float32')

    if new_momentum:
        df = pd.concat([df, pd.DataFrame(new_momentum, index=df.index)], axis=1)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df = date_columns_creation(df=df)

    for col in COLUMN_LIST_BASE:
        df = add_lag_features(df, group_cols=[col], lags=CUSTOM_LAGS)
        df = add_rolling_features(df, group_cols=[col], windows=ROLLING_CONFIG)
        
        for config_name, config in ROLLING_LAGS_CONFIG.items():
            df = add_lagged_rolling_features(df, [col], config["lag"], config["window"], f"{col}_{config_name}")

        for short_win, long_win in TREND_CONFIG:
            df = add_trend_features(df, [col], short_win, long_win)

    # Momentum après les calculs de base
    df = add_momentum_features(df, 'rolling_week', 'rolling_month', "accel")
    df = add_momentum_features(df, 'rolling_month', 'rolling_quarter', "month_ratio")
    
    # Interactions à la fin
    if "NbOfSeats" in df.columns:
        df = add_interaction_features(df, base_col="NbOfSeats", feature_pattern="_mean")

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/main.csv")
    df = add_features(df)