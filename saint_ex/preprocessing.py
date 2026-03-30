import pandas as pd
import numpy as np
import os
from google.cloud import bigquery
from saint_ex.config import (
    INFERENCE_START_DATE, LOAD_COLUMNS, DATA_FILE,
    USE_BIGQUERY, BQ_PROJECT, BQ_DATASET, BQ_TABLE, BQ_CREDS
)

def load_dataset(file_path: str = DATA_FILE) -> pd.DataFrame:
    """
    Main ingestion orchestrator. Toggles between local CSV snapshot 
    and live BigQuery production data based on configuration.
    """
    if USE_BIGQUERY:
        df = _load_from_bigquery()
    else:
        # Fallback to Local CSV
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing input dataset: {file_path}")
            
        print(f"Loading Local Snapshot: {file_path}...")
        df = pd.read_csv(file_path, low_memory=False, usecols=LOAD_COLUMNS)
        df = _normalize_schema(df)
    
    # ── External Data Joins ──────────────────────────────────────────────────
    # Identify destination IATA once for all joins
    df['AirportDestination'] = np.where(df['Direction'].str.startswith('Arr'), 'LYS', df['SysStopover'])
    
    df = _add_weather_signals(df)
    df = _add_school_holidays(df)
    df = _add_national_holidays(df)
    
    # Cleanup intermediate helpers
    df = df.drop(columns=['AirportDestination'], errors='ignore')
    
    return df

def _load_from_bigquery() -> pd.DataFrame:
    """
    Live ingestion from BigQuery using standard REST transport.
    Fetches the full ground-truth history including recent 2026 labels.
    """
    print(f"Syncing Live BigQuery Dataset: {BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}...")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = BQ_CREDS
    
    # Initialize Standard Client
    client = bigquery.Client(project=BQ_PROJECT)
    
    # 📝 Authoritative Query to ensure consistent schema with CSV
    cols_str = ", ".join(LOAD_COLUMNS)
    query = f"SELECT {cols_str} FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`"
    
    # Fallback to REST (Storage API requires specific permissions not in SA)
    df = client.query(query).to_dataframe()
    return _normalize_schema(df)

def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes types and filters technical operations to isolate commercial flow."""
    # Standardize Chronological Index (Force ns precision for merge compatibility)
    df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime']).dt.tz_localize(None).astype('datetime64[ns]')
    
    # ♿ Standardize PRM Target
    if 'FarmsNbPaxPHMR' in df.columns:
        df['NbPRMTotal'] = pd.to_numeric(df['FarmsNbPaxPHMR'], errors='coerce').fillna(0)
    else:
        df['NbPRMTotal'] = 0
    
    # Filter Commercial Flows (Exclude Ferry/Technical/Non-Pax)
    initial_count = len(df)
    
    # Strict Commercial Filter — Chapter IX Compliance
    # Included: J (Scheduled), C (Charter), G (General), S (Scheduled Revenue), O (Contract Charter), L (Charter mixed)
    # Excluded: T (Technical), E (Government), P (Positioning), X (Test), F (Freight), W (Military), R (Regional), etc.
    mask = (
        (df['ServiceCode'].isin(['J', 'S', 'C', 'G', 'O', 'L'])) & 
        (df['IdBusinessUnitType'] == 1) &
        (df['NbOfSeats'] > 0) &
        (df['flight_with_pax'].fillna('Oui') == 'Oui')
    )
    df = df[mask].copy()
    
    # For Backtests, we only care about realized rows
    if 'NbPaxTotal' in df.columns:
        df = df[df['NbPaxTotal'] > 0].copy()
        
    diff = initial_count - len(df)
    print(f"  Processed {len(df):,} commercial flights ({diff:,} ferry/technical filtered).")
    
    # Initialize high-quality identifiers
    df['OperatorOACICodeNormalized'] = df['airlineOACICode'].fillna('UNKNOWN')
    
    return df

def _add_weather_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Joins weather signals for both origin and destination (LYS Hub focus)."""
    from saint_ex.config import WEATHER_FILE
    if not os.path.exists(WEATHER_FILE):
        print(f"  ⚠️ Weather file missing at {WEATHER_FILE}. Skipping join.")
        return df

    weather = pd.read_csv(WEATHER_FILE)
    weather['date'] = pd.to_datetime(weather['date']).dt.date
    df['date_only'] = df['LTScheduledDatetime'].dt.date
    
    # 1. Join for Origin (Temp/Rain at departure point)
    df = df.merge(
        weather[['date', 'iata', 'temp_max', 'precip']],
        left_on=['date_only', 'AirportOrigin'],
        right_on=['date', 'iata'],
        how='left'
    ).rename(columns={'temp_max': 'temp_max_origin', 'precip': 'precip_origin'})
    
    # 2. Join for Destination (Rain at arrival point, mainly LYS)
    df = df.merge(
        weather[['date', 'iata', 'precip']],
        left_on=['date_only', 'AirportDestination'],
        right_on=['date', 'iata'],
        how='left'
    ).rename(columns={'precip': 'precip_dest'})
    
    # Cleanup join artifacts
    drop_cols = ['date_x', 'iata_x', 'date_y', 'iata_y', 'date_only']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Fill NAs for weather (Baseline 0 rain, 20C temp)
    df['precip_origin'] = df['precip_origin'].fillna(0)
    df['precip_dest'] = df['precip_dest'].fillna(0)
    df['temp_max_origin'] = df['temp_max_origin'].fillna(20)
    
    return df

def _add_school_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """Joins French school holiday zones (A, B, C)."""
    from saint_ex.config import SCHOOL_CAL_FILE
    if not os.path.exists(SCHOOL_CAL_FILE):
        return df
    
    school_cal = pd.read_csv(SCHOOL_CAL_FILE)
    school_cal['start'] = pd.to_datetime(school_cal['start']).dt.date
    school_cal['end'] = pd.to_datetime(school_cal['end']).dt.date
    
    df['date'] = df['LTScheduledDatetime'].dt.date
    for zone in ['Zone A', 'Zone B', 'Zone C']:
        z_col = f"is_school_holiday_{zone.lower().replace(' ', '_')}"
        df[z_col] = 0
        z_cal = school_cal[school_cal['zone'] == zone]
        for _, row in z_cal.iterrows():
            mask = (df['date'] >= row['start']) & (df['date'] <= row['end'])
            df.loc[mask, z_col] = 1
    
    df = df.drop(columns=['date'])
    return df

def _add_national_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """Joins bank holidays for origin and destination countries."""
    import holidays
    try:
        import airportsdata
        AIRPORTS = airportsdata.load('IATA')
    except Exception:
        AIRPORTS = {}

    def get_country(iata):
        if not isinstance(iata, str): return None
        apt = AIRPORTS.get(iata)
        return apt['country'] if apt else None

    df['date'] = df['LTScheduledDatetime'].dt.date
    df['origin_country'] = df['AirportOrigin'].map(get_country)
    df['dest_country']   = df['AirportDestination'].map(get_country)
    
    unique_countries = set(df['origin_country'].dropna().unique()) | set(df['dest_country'].dropna().unique())
    holiday_cache = {c: set(holidays.country_holidays(c, years=[2023, 2024, 2025, 2026]).keys()) for c in unique_countries}
    
    df['is_origin_holiday'] = df.apply(lambda r: int(r['date'] in holiday_cache.get(r['origin_country'], set())), axis=1)
    df['is_destination_holiday'] = df.apply(lambda r: int(r['date'] in holiday_cache.get(r['dest_country'], set())), axis=1)
    
    df = df.drop(columns=['date', 'origin_country', 'dest_country'])
    return df

def split_historical_inference(df: pd.DataFrame, val_ratio: float = 0.15):
    """Chronologically splits the dataset based on the configured INFERENCE_START_DATE."""
    snapshot_cutoff = pd.to_datetime(INFERENCE_START_DATE)
    
    historical = df[df['LTScheduledDatetime'] < snapshot_cutoff].copy()
    inference  = df[df['LTScheduledDatetime'] >= snapshot_cutoff].copy()
    
    historical = historical.sort_values(by='LTScheduledDatetime')
    cutoff_idx = int(len(historical) * (1 - val_ratio))
    
    train = historical.iloc[:cutoff_idx].copy()
    val   = historical.iloc[cutoff_idx:].copy()
    
    print("\nSplitting dynamic data stream...")
    print(f"  Training Range: {historical['LTScheduledDatetime'].min()} -> {historical['LTScheduledDatetime'].max()}")
    print(f"    - train     : {len(train):,}")
    print(f"    - validation: {len(val):,}")
    print(f"  Inference Pool: {len(inference):,}")
    
    return train, val, inference
