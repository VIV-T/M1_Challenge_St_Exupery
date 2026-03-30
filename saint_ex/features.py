"""
features.py — Dynamic Bidirectional Feature Engineering for Project Saint-Exupéry.

This module extracts temporal, systemic, and external signals for
passenger flow prediction. All external data (Weather, School Calendars,
Global Holidays) are loaded from dynamic local caches (no hardcoding).
"""
import numpy as np
import pandas as pd
import os
# --- External data logic moved to preprocessing.py ---

def add_features(df: pd.DataFrame, reference_stats: pd.DataFrame = None) -> pd.DataFrame:
    """
    Main orchestrator for Project Saint-Exupéry feature engineering.
    Sequentially enriches the dataset with temporal, external, and historical signals.
    """
    df = df.copy()
    df['date'] = df['LTScheduledDatetime'].dt.date
    
    # Sequential enrichment stages
    df = _add_temporal_cycles(df)
    df = _add_micro_temporal(df)
    df = _add_flight_attributes(df)
    df = _add_religious_surges(df)
    df = _add_historical_lags(df)
    df = _add_historical_occupancy(df, reference_stats)
    df = _add_hub_momentum(df)
    df = _add_route_momentum(df)
    df = _add_weather_interactions(df)
    
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

# Weather and Holiday joins are now handled in preprocessing.py

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

def _add_micro_temporal(df):
    """Encodes minute-of-day for high-resolution slot identification (e.g. the 8:04 rush)."""
    dt = df['LTScheduledDatetime']
    df['minute_of_day'] = dt.dt.hour * 60 + dt.dt.minute
    df['sin_min'] = np.sin(2 * np.pi * df['minute_of_day'] / 1440)
    df['cos_min'] = np.cos(2 * np.pi * df['minute_of_day'] / 1440)
    return df

def _add_hub_momentum(df):
    """Injects the 'Airport Pulse'—the average terminal-wide occupancy of the last week."""
    if 'NbPaxTotal' not in df.columns:
        df['hub_momentum_7d'] = 0.8
        return df

    # Calculate global daily yield on available labels ONLY
    df_labels = df[df['NbPaxTotal'].notna()].copy()
    if df_labels.empty:
        df['hub_momentum_7d'] = 0.8
        return df

    df_labels['temp_occ'] = (df_labels['NbPaxTotal'] / df_labels['NbOfSeats'].clip(lower=1)).clip(0, 1.2)
    daily_hub = df_labels.groupby('date')['temp_occ'].mean()
    
    # Ensure time-series continuity for rolling window
    full_range = pd.date_range(df['date'].min(), df['date'].max())
    daily_hub = daily_hub.reindex(full_range)
    
    # 7-day rolling average, shifted by 1 to prevent leakage
    hub_mom = daily_hub.rolling(window=7, min_periods=1).mean().shift(1).fillna(0.8)
    
    mom_map = hub_mom.to_dict()
    df['hub_momentum_7d'] = df['date'].map(mom_map).fillna(0.8)
    return df

def _add_weather_interactions(df):
    """Signals how weather impacts different segments (e.g. rain affects boarding vs landing)."""
    # Interaction: Rain at Destination * Direction (Does rain at LYS affect arrivals?)
    # We use 'precip_dest' which is LYS for arrivals
    df['rain_arrival_impact'] = df['is_arrival'] * df['precip_dest'].fillna(0).clip(lower=0)
    
    # Heat stress interaction (Temperatures > 32C impact passenger comfort/boarding speed)
    df['heat_stress'] = (df['temp_max_origin'] > 32).astype(int)
    return df

def _add_route_momentum(df):
    """Injects high-resolution carrier/origin yield signatures from the last week."""
    if 'NbPaxTotal' not in df.columns:
        df['route_momentum_7d'] = 0.8
        return df

    df_labels = df[df['NbPaxTotal'].notna()].copy()
    if df_labels.empty:
        df['route_momentum_7d'] = 0.8
        return df

    df_labels['temp_occ'] = (df_labels['NbPaxTotal'] / df_labels['NbOfSeats'].clip(lower=1)).clip(0, 1.2)
    
    # Calculate per-route daily means
    route_daily = df_labels.groupby(['date', 'airlineOACICode', 'AirportOrigin'])['temp_occ'].mean().reset_index()
    
    # For each route, we need a 7-day rolling average
    route_daily = route_daily.sort_values('date')
    route_daily['route_momentum_7d'] = route_daily.groupby(['airlineOACICode', 'AirportOrigin'])['temp_occ'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean().shift(1)
    )
    
    # Map back to the main dataframe
    df = df.merge(
        route_daily[['date', 'airlineOACICode', 'AirportOrigin', 'route_momentum_7d']], 
        on=['date', 'airlineOACICode', 'AirportOrigin'], 
        how='left'
    )
    df['route_momentum_7d'] = df['route_momentum_7d'].fillna(df.get('hub_momentum_7d', 0.8))
    return df

def prepare_X(df, columns=None):
    """Cast features to rigorous LightGBM types."""
    from saint_ex.config import ALL_FEATURES, CATEGORICAL_FEATURES
    columns = columns or ALL_FEATURES
    X = df[[c for c in columns if c in df.columns]].copy()
    
    for col in CATEGORICAL_FEATURES:
        if col in X: X[col] = X[col].astype('category')
        
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(-1)
    
    return X, X.columns.tolist()

def get_route_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates static historical occupancy for each route."""
    df = df.copy()
    temp_occ = (df['NbPaxTotal'] / df['NbOfSeats'].clip(lower=1)).clip(0, 1.2)
    stats = pd.concat([df[['airlineOACICode', 'AirportOrigin']], temp_occ.rename('occ')], axis=1)
    stats = stats.groupby(['airlineOACICode', 'AirportOrigin'])['occ'].mean().reset_index()
    stats.columns = ['airlineOACICode', 'AirportOrigin', 'route_avg_occupancy']
    return stats
