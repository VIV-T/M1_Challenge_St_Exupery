"""
features.py — Dynamic Bidirectional Feature Engineering for Project Saint-Exupéry.

This module extracts temporal, systemic, and external signals for
passenger flow prediction. All external data (Weather, School Calendars,
Global Holidays) are loaded from dynamic local caches (no hardcoding).
"""
import numpy as np
import pandas as pd
import os
import holidays
try:
    import airportsdata
    AIRPORTS = airportsdata.load('IATA')
except Exception:
    AIRPORTS = {}

# --- External Data Signal Mapping ---
WEATHER_FILE     = 'externals/weather_hubs.csv'
SCHOOL_CAL_FILE  = 'externals/school_holidays.csv'
WEATHER_DF       = pd.read_csv(WEATHER_FILE) if os.path.exists(WEATHER_FILE) else None
SCHOOL_CAL       = pd.read_csv(SCHOOL_CAL_FILE) if os.path.exists(SCHOOL_CAL_FILE) else None

# Standardize date objects for joining
if WEATHER_DF is not None: WEATHER_DF['date'] = pd.to_datetime(WEATHER_DF['date']).dt.date
if SCHOOL_CAL is not None:
    SCHOOL_CAL['start'] = pd.to_datetime(SCHOOL_CAL['start']).dt.date
    SCHOOL_CAL['end'] = pd.to_datetime(SCHOOL_CAL['end']).dt.date

def get_country(iata):
    """Maps IATA codes to ISO country codes via airportsdata."""
    if not isinstance(iata, str): return None
    apt = AIRPORTS.get(iata)
    return apt['country'] if apt else None

# --- Feature Configuration ---
CATEGORICAL = [
    'airlineOACICode', 'OperatorOACICodeNormalized', 'SysStopover',
    'AirportOrigin', 'IdAircraftType', 'Terminal', 'ServiceCode',
    'FuelProvider',
]

ALL_FEATURES = [
    'NbOfSeats', 'NbConveyor', 'NbAirbridge',
    'IdBusContactType', 'IdTerminalType', 'IdBagStatusDelivery',
    'is_arrival', 'is_charter',
    'temp_max_origin', 'precip_origin',
    'temp_max_dest', 'precip_dest',
    'is_school_holiday_zone_a', 'is_school_holiday_zone_b', 'is_school_holiday_zone_c',
    'is_origin_holiday', 'is_destination_holiday',
    *CATEGORICAL,
    'year', 'month', 'week', 'dayofweek', 'hour', 'dayofyear',
    'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
]

def add_features(df):
    """Orchestrates the transformation of raw flight logs into high-dimension features."""
    df = df.copy()
    dt = df['LTScheduledDatetime']
    
    # ⏱️ Temporal Cycles
    df['hour']      = dt.dt.hour
    df['dayofweek'] = dt.dt.dayofweek
    df['month']     = dt.dt.month
    df['week']      = dt.dt.isocalendar().week.astype(int)
    df['year']      = dt.dt.year
    df['dayofyear'] = dt.dt.dayofyear
    
    for col, period in [('hour', 24), ('month', 12), ('dayofweek', 7)]:
        df[f'sin_{col}'] = np.sin(2 * np.pi * df[col] / period)
        df[f'cos_{col}'] = np.cos(2 * np.pi * df[col] / period)

    # 🗓️ Binary Calendar Indicators
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # 🛫 Flight systemic attributes
    df['is_arrival'] = df['Direction'].str.lower().str.startswith('arr').astype(int)
    df['is_charter'] = (df['ScheduleType'].fillna('') == 'Non Régulier').astype(int)
    for col, fill in [('NbOfSeats', 0), ('NbConveyor', 1), ('NbAirbridge', 0)]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill)
    for col in ['IdBusContactType', 'IdTerminalType', 'IdBagStatusDelivery']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 🌩️ Bidirectional External Signals
    df['date'] = df['LTScheduledDatetime'].dt.date
    df['origin_iata'] = np.where(df['is_arrival'] == 1, df['AirportOrigin'], 'LYS')
    df['dest_iata']   = np.where(df['is_arrival'] == 0, df['SysStopover'], 'LYS')
    
    # 1. Weather Logic (Origin vs Destination)
    for prefix, iata_col in [('origin', 'origin_iata'), ('dest', 'dest_iata')]:
        if WEATHER_DF is not None:
            w_sub = WEATHER_DF[['date', 'iata', 'temp_max', 'precip']].copy()
            w_sub.columns = ['date', iata_col, f'temp_max_{prefix}', f'precip_{prefix}']
            df = df.merge(w_sub, on=['date', iata_col], how='left')
            
            # Proxy missing hub weather with Lyon baseline
            lys_w = WEATHER_DF[WEATHER_DF['iata'] == 'LYS'][['date', 'temp_max', 'precip']].copy()
            lys_w.columns = ['date', f'temp_max_{prefix}_lys', f'precip_{prefix}_lys']
            df = df.merge(lys_w, on='date', how='left')
            
            df[f'temp_max_{prefix}'] = df[f'temp_max_{prefix}'].fillna(df[f'temp_max_{prefix}_lys']).fillna(-1)
            df[f'precip_{prefix}']   = df[f'precip_{prefix}'].fillna(df[f'precip_{prefix}_lys']).fillna(-1)
            df = df.drop(columns=[f'temp_max_{prefix}_lys', f'precip_{prefix}_lys'])
        else:
            df[f'temp_max_{prefix}'], df[f'precip_{prefix}'] = -1, -1

    # 2. Dynamic multi-zone school calendars
    for zone in ['Zone A', 'Zone B', 'Zone C']:
        z_col = zone.lower().replace(' ', '_')
        df[f'is_school_holiday_{z_col}'] = 0
        if SCHOOL_CAL is not None:
            z_cal = SCHOOL_CAL[SCHOOL_CAL['zone'] == zone]
            for _, row in z_cal.iterrows():
                mask = (df['date'] >= row['start']) & (df['date'] <= row['end'])
                df.loc[mask, f'is_school_holiday_{z_col}'] = 1

    # 3. Dynamic Global Holiday Lookup (Origin vs Destination)
    df['origin_country'] = df['origin_iata'].map(get_country)
    df['dest_country']   = df['dest_iata'].map(get_country)
    unique_countries     = set(df['origin_country'].dropna().unique()) | set(df['dest_country'].dropna().unique())
    holiday_cache        = {c: set(holidays.country_holidays(c, years=[2023, 2024, 2025, 2026]).keys()) 
                            for c in unique_countries}
            
    df['is_origin_holiday']      = df.apply(lambda r: int(r['date'] in holiday_cache.get(r['origin_country'], set())), axis=1)
    df['is_destination_holiday'] = df.apply(lambda r: int(r['date'] in holiday_cache.get(r['dest_country'], set())), axis=1)
    
    return df.drop(columns=['date', 'origin_iata', 'dest_iata', 'origin_country', 'dest_country'])

def prepare_X(df, columns=None):
    """Final cast to categorical and numeric types before LightGBM ingestion."""
    columns = columns or [c for c in ALL_FEATURES if c in df.columns]
    X = df[columns].copy()
    for col in CATEGORICAL:
        if col in X: X[col] = X[col].astype('category')
    numeric_cols = [c for c in columns if c not in CATEGORICAL]
    X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(-1)
    return X, columns
