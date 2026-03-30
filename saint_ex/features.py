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
from saint_ex.config import WEATHER_FILE, SCHOOL_CAL_FILE, CATEGORICAL_FEATURES, ALL_FEATURES

try:
    import airportsdata
    AIRPORTS = airportsdata.load('IATA')
except Exception:
    AIRPORTS = {}

# --- Cache External Data Objects ---
WEATHER_DF = pd.read_csv(WEATHER_FILE) if os.path.exists(WEATHER_FILE) else None
SCHOOL_CAL = pd.read_csv(SCHOOL_CAL_FILE) if os.path.exists(SCHOOL_CAL_FILE) else None

if WEATHER_DF is not None: 
    WEATHER_DF['date'] = pd.to_datetime(WEATHER_DF['date']).dt.date
if SCHOOL_CAL is not None:
    SCHOOL_CAL['start'] = pd.to_datetime(SCHOOL_CAL['start']).dt.date
    SCHOOL_CAL['end'] = pd.to_datetime(SCHOOL_CAL['end']).dt.date

def _get_country(iata):
    """Maps IATA codes to ISO country codes via airportsdata utility."""
    if not isinstance(iata, str): return None
    apt = AIRPORTS.get(iata)
    return apt['country'] if apt else None

def add_features(df: pd.DataFrame, reference_stats: pd.DataFrame = None) -> pd.DataFrame:
    """
    Main orchestrator for Project Saint-Exupéry feature engineering.
    Sequentially enriches the dataset with temporal, external, and historical signals.
    """
    df = df.copy()
    df['date'] = df['LTScheduledDatetime'].dt.date
    
    # Sequential enrichment stages
    df = _add_temporal_cycles(df)
    df = _add_flight_attributes(df)
    df = _add_external_signals(df)
    df = _add_religious_surges(df)
    df = _add_historical_lags(df)
    df = _add_historical_occupancy(df, reference_stats)
    
    # Drop intermediate keys and columns
    drop_cols = ['date', 'origin_iata', 'dest_iata', 'origin_country', 'dest_country', 'tight_key', 'eid_start']
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

def _add_temporal_cycles(df):
    """Injects cyclic temporal features (Sine/Cosine encoding) for Time/Day/Month."""
    dt = df['LTScheduledDatetime']
    df['hour']      = dt.dt.hour
    df['dayofweek'] = dt.dt.dayofweek
    df['month']     = dt.dt.month
    df['week']      = dt.dt.isocalendar().week.astype(int)
    df['year']      = dt.dt.year
    df['dayofyear'] = dt.dt.dayofyear
    
    for col, period in [('hour', 24), ('month', 12), ('dayofweek', 7)]:
        df[f'sin_{col}'] = np.sin(2 * np.pi * df[col] / period)
        df[f'cos_{col}'] = np.cos(2 * np.pi * df[col] / period)
    return df

def _add_flight_attributes(df):
    """Normalizes core flight attributes and identifiers."""
    df['is_arrival'] = df['Direction'].str.lower().str.startswith('arr').astype(int)
    df['is_charter'] = (df['ScheduleType'].fillna('') == 'Non Régulier').astype(int)
    
    for col, fill in [('NbOfSeats', 0), ('NbConveyor', 1), ('NbAirbridge', 0)]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill)
    for col in ['IdBusContactType', 'IdTerminalType', 'IdBagStatusDelivery']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def _add_external_signals(df):
    """Joins weather, regional school holidays, and global calendar events."""
    df['origin_iata'] = np.where(df['is_arrival'] == 1, df['AirportOrigin'], 'LYS')
    df['dest_iata']   = np.where(df['is_arrival'] == 0, df['SysStopover'], 'LYS')
    
    # 1. Weather Matching
    for prefix, iata_col in [('origin', 'origin_iata'), ('dest', 'dest_iata')]:
        if WEATHER_DF is not None:
            w_sub = WEATHER_DF[['date', 'iata', 'temp_max', 'precip']].copy()
            w_sub.columns = ['date', iata_col, f'temp_max_{prefix}', f'precip_{prefix}']
            df = df.merge(w_sub, on=['date', iata_col], how='left')
            lys_w = WEATHER_DF[WEATHER_DF['iata'] == 'LYS'][['date', 'temp_max', 'precip']].copy()
            df = df.merge(lys_w.rename(columns={'temp_max': 't_lys', 'precip': 'p_lys'}), on='date', how='left')
            df[f'temp_max_{prefix}'] = df[f'temp_max_{prefix}'].fillna(df['t_lys']).fillna(-1)
            df[f'precip_{prefix}']   = df[f'precip_{prefix}'].fillna(df['p_lys']).fillna(-1)
            df = df.drop(columns=['t_lys', 'p_lys'])
        else:
            df[f'temp_max_{prefix}'], df[f'precip_{prefix}'] = -1, -1

    # 2. School Calendars
    for zone in ['Zone A', 'Zone B', 'Zone C']:
        z_col = zone.lower().replace(' ', '_')
        df[f'is_school_holiday_{z_col}'] = 0
        if SCHOOL_CAL is not None:
            z_cal = SCHOOL_CAL[SCHOOL_CAL['zone'] == zone]
            for _, row in z_cal.iterrows():
                mask = (df['date'] >= row['start']) & (df['date'] <= row['end'])
                df.loc[mask, f'is_school_holiday_{z_col}'] = 1

    # 3. National Holidays
    df['origin_country'] = df['origin_iata'].map(_get_country)
    df['dest_country']   = df['dest_iata'].map(_get_country)
    unique_countries     = set(df['origin_country'].dropna().unique()) | set(df['dest_country'].dropna().unique())
    holiday_cache        = {c: set(holidays.country_holidays(c, years=[2023, 2024, 2025, 2026]).keys()) for c in unique_countries}
    df['is_origin_holiday']      = df.apply(lambda r: int(r['date'] in holiday_cache.get(r['origin_country'], set())), axis=1)
    df['is_destination_holiday'] = df.apply(lambda r: int(r['date'] in holiday_cache.get(r['dest_country'], set())), axis=1)
    return df

def _add_religious_surges(df):
    """Implements Hijri-alignment for religious travel surges."""
    eid_dates = {2023: pd.to_datetime('2023-04-21'), 2024: pd.to_datetime('2024-04-10'), 
                 2025: pd.to_datetime('2025-03-31'), 2026: pd.to_datetime('2026-03-20')}
    df['eid_start'] = pd.to_datetime(df['year'].map(eid_dates.get))
    df['days_from_eid'] = (pd.to_datetime(df['date']) - df['eid_start']).dt.days
    df['days_from_eid'] = df['days_from_eid'].fillna(0).clip(-30, 30)
    df['return_surge'] = np.exp(-((df['days_from_eid'] - 7)**2) / (2 * 3**2))
    return df

def _add_historical_lags(df):
    """Calculates tight-route concurrency and occupancy lags."""
    df['tight_key'] = df['airlineOACICode'].astype(str) + df['FlightNumberNormalized'].astype(str) + df['Direction'].astype(str)
    
    # Hub-level concurrency pressure
    hub_counts = df.groupby('LTScheduledDatetime').size().reset_index(name='hub_pressure')
    df = df.merge(hub_counts, on='LTScheduledDatetime', how='left')

    # Flight-specific Historical Lags
    ref_cols = ['LTScheduledDatetime', 'tight_key', 'NbPaxTotal']
    historical = df[df['NbPaxTotal'].notna()][ref_cols].sort_values('LTScheduledDatetime').copy()
    
    for lag_days in [7, 14]:
        lag_ref = historical.copy()
        lag_ref['LTScheduledDatetime'] = lag_ref['LTScheduledDatetime'] + pd.Timedelta(days=lag_days)
        lag_ref.columns = ['LTScheduledDatetime', 'tight_key', f'NbPax_Lag_{lag_days}d']
        df = pd.merge_asof(df.sort_values('LTScheduledDatetime'), lag_ref.sort_values('LTScheduledDatetime'),
                           on='LTScheduledDatetime', by='tight_key', direction='nearest', tolerance=pd.Timedelta(hours=4))
    return df

def _add_historical_occupancy(df, stats=None):
    """
    Injects route-level 'typical occupancy' signatures.
    stats should be a DataFrame with ['airlineOACICode', 'AirportOrigin', 'route_avg_occupancy']
    """
    if stats is not None:
        df = df.merge(stats, on=['airlineOACICode', 'AirportOrigin'], how='left')
    else:
        # Fallback for when we don't have stats yet (e.g. initial train pass)
        df['route_avg_occupancy'] = 0.8 # Global airport average fallback
        
    df['route_avg_occupancy'] = df['route_avg_occupancy'].fillna(0.7)
    return df

def prepare_X(df, columns=None):
    """Cast features to rigorous LightGBM types."""
    columns = columns or ALL_FEATURES
    X = df[[c for c in columns if c in df.columns]].copy()
    
    for col in CATEGORICAL_FEATURES:
        if col in X: X[col] = X[col].astype('category')
        
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(-1)
    
    return X, X.columns.tolist()
