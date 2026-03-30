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

def split_historical_inference(df, inference_start_date='2026-03-01'):
    """Splits the full dataset into training/validation and the prediction pool."""
    historical = df[df['LTScheduledDatetime'] < inference_start_date].copy()
    inference  = df[df['LTScheduledDatetime'] >= inference_start_date].copy()
    
    # Internal validation split for final calibration (Last 5 months of history)
    val_cutoff = '2025-10-01'
    train = historical[historical['LTScheduledDatetime'] < val_cutoff].copy()
    val   = historical[historical['LTScheduledDatetime'] >= val_cutoff].copy()
    
    print("\nSplitting historical data...")
    print(f"  train: {len(train):,}")
    print(f"    val: {len(val):,}")
    print(f"  Inference Pool: {len(inference):,}")
    
    return train, val, inference
