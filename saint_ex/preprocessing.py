import pandas as pd
import os
from google.cloud import bigquery
from saint_ex.config import (
    INFERENCE_START_DATE, LOAD_COLUMNS, 
    USE_BIGQUERY, BQ_PROJECT, BQ_DATASET, BQ_TABLE, BQ_CREDS
)

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Main ingestion orchestrator. Toggles between local CSV snapshot 
    and live BigQuery production data based on configuration.
    """
    if USE_BIGQUERY:
        return _load_from_bigquery()
    
    # Fallback to Local CSV
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing input dataset: {file_path}")
         
    print(f"Loading Local Snapshot: {file_path}...")
    df = pd.read_csv(file_path, low_memory=False, usecols=LOAD_COLUMNS)
    return _normalize_schema(df)

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
