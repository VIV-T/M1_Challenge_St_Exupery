"""
preprocessing.py — Core Data Cleaning & Normalization for Project Saint-Exupéry.

This module handles the loading of the raw IATA dataset and performs
the necessary technical filtering (technical flights, ferry travel, etc.)
to isolate commercial passenger flows.
"""
import pandas as pd
import os

def load_dataset(file_path):
    """Loads and performs schema-level normalization on the airport dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing input dataset: {file_path}")
        
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Standardize time dimension
    df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime'])
    
    # ♿ PRM Column Mapping (FarmsNbPaxPHMR is the target for Travelers with Reduced Mobility)
    if 'FarmsNbPaxPHMR' in df.columns:
        df['NbPRMTotal'] = pd.to_numeric(df['FarmsNbPaxPHMR'], errors='coerce').fillna(0)
    else:
        df['NbPRMTotal'] = 0
    
    # Technical Flight Filter (Ferry, Crew, Technical stops)
    initial_count = len(df)
    df = df[~df['ServiceCode'].isin(['T', 'E', 'C', 'P', 'X'])].copy()
    diff = initial_count - len(df)
    print(f"  Processed {len(df):,} commercial flights ({diff:,} ferry/technical filtered).")
    
    # Normalize airline identifiers
    df['OperatorOACICodeNormalized'] = df['airlineOACICode'].fillna('UNKNOWN')
    
    return df

from saint_ex.config import INFERENCE_START_DATE

def split_historical_inference(df, val_ratio=0.15):
    """
    Splits the full dataset into labeled (historical) and unlabeled (inference) pools.
    Performs a chronological 85/15 split on labeled data for internal validation.
    The cutoff is driven by INFERENCE_START_DATE in config.py.
    """
    # 🏁 Precise Snapshot Boundary from Configuration
    snapshot_cutoff = INFERENCE_START_DATE
    
    historical = df[df['LTScheduledDatetime'] < snapshot_cutoff].copy()
    inference  = df[df['LTScheduledDatetime'] >= snapshot_cutoff].copy()
    
    # Chronological Validation (Last 15% of historical data)
    historical = historical.sort_values(by='LTScheduledDatetime')
    cutoff_idx = int(len(historical) * (1 - val_ratio))
    
    train = historical.iloc[:cutoff_idx].copy()
    val   = historical.iloc[cutoff_idx:].copy()
    
    print("\nSplitting dynamic snapshot data...")
    print(f"  Historical Training Range: {historical['LTScheduledDatetime'].min()} -> {historical['LTScheduledDatetime'].max()}")
    print(f"    - train     : {len(train):,}")
    print(f"    - validation: {len(val):,}")
    print(f"  Inference Pool (Recent Blind Test): {len(inference):,}")
    
    return train, val, inference
