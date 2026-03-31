"""
Data preprocessing module for Project Saint-Exupéry Airport Passenger Flow Prediction.

This module handles data ingestion, cleaning, filtering, and temporal splitting.
It supports both live BigQuery data and local CSV snapshots with automatic fallback.

Key Functions:
- load_dataset(): Main data ingestion orchestrator
- split_historical_inference(): Temporal data splitting for validation
- External data joins: Weather, school holidays, national holidays
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional
from google.cloud import bigquery
from saint_ex.config import (
    INFERENCE_START_DATE, LOAD_COLUMNS, DATA_FILE,
    USE_BIGQUERY, BQ_PROJECT, BQ_DATASET, BQ_TABLE, BQ_CREDS,
    VALID_SERVICE_CODES, COMMERCIAL_BUSINESS_UNIT, 
    MIN_SEATS_FOR_COMMERCIAL, FLIGHT_WITH_PAX_INDICATOR
)

def load_dataset(file_path: str = DATA_FILE) -> pd.DataFrame:
    """
    Main data ingestion orchestrator.
    
    Automatically toggles between live BigQuery data and local CSV snapshot
    based on the USE_BIGQUERY configuration flag. Applies consistent schema
    normalization and external data joins regardless of data source.
    
    Args:
        file_path: Path to local CSV file (used only when USE_BIGQUERY=False)
        
    Returns:
        DataFrame with cleaned and enriched flight data
        
    Raises:
        FileNotFoundError: If local CSV file is missing when USE_BIGQUERY=False
    """
    if USE_BIGQUERY:
        df = _load_from_bigquery()
    else:
        # Fallback to Local CSV
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing input dataset: {file_path}")
            
        df = pd.read_csv(file_path, low_memory=False, usecols=LOAD_COLUMNS)
        df = _normalize_schema(df)
    
    # Apply external data enrichment
    df = _enrich_with_external_data(df)
    
    return df

def _load_from_bigquery() -> pd.DataFrame:
    """
    Load live data from BigQuery using standard REST transport.
    
    Fetches the full ground-truth history including recent 2026 labels.
    Uses the authoritative query to ensure consistent schema with CSV.
    
    Returns:
        DataFrame with live BigQuery data
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = BQ_CREDS
    
    # Initialize Standard Client
    client = bigquery.Client(project=BQ_PROJECT)
    
    # Authoritative Query to ensure consistent schema with CSV
    cols_str = ", ".join(LOAD_COLUMNS)
    query = f"SELECT {cols_str} FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`"
    
    # Execute query and return results
    df = client.query(query).to_dataframe()
    return _normalize_schema(df)

def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes data types and identifies commercial vs non-commercial flights.
    - Standardizes datetime formats
    - Identifies commercial vs freight/ferry flights using `is_commercial` flag
    - Ensures target variable consistency
    """
    # Standardize Chronological Index (Force ns precision for merge compatibility)
    df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime']).dt.tz_localize(None).astype('datetime64[ns]')
    
    # Standardize PRM Target (Passengers with Reduced Mobility)
    if 'FarmsNbPaxPHMR' in df.columns:
        df['NbPRMTotal'] = pd.to_numeric(df['FarmsNbPaxPHMR'], errors='coerce').fillna(0)
    else:
        df['NbPRMTotal'] = 0
    
    # Identify Commercial Flights (Exclude Ferry/Technical/Non-Pax)
    # Apply commercial flight filters based on business rules
    df['is_commercial'] = (
        df['ServiceCode'].isin(VALID_SERVICE_CODES) &  # Valid service types
        (df['IdBusinessUnitType'] == COMMERCIAL_BUSINESS_UNIT) &  # Commercial business unit
        (df['NbOfSeats'] > MIN_SEATS_FOR_COMMERCIAL) &  # Has seating capacity
        # For future flights, flight_with_pax is NULL. Accept NULL or 'Oui '.
        (df['flight_with_pax'].isnull() | (df['flight_with_pax'] == FLIGHT_WITH_PAX_INDICATOR))
    )
    
    # Initialize high-quality identifiers
    df['OperatorOACICodeNormalized'] = df['airlineOACICode'].fillna('UNKNOWN')
    
    # Deduplicate by IdADL to ensure stable matching for evaluation
    # First drop cases with missing IdADL as they cannot be accurately backtested
    df = df.dropna(subset=['IdADL'])
    df = df.drop_duplicates(subset=['IdADL'])
    
    return df.reset_index(drop=True)



def _enrich_with_external_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches flight data with external signals (weather, holidays, etc.).
    
    This function orchestrates all external data joins:
    - Weather data for origin and destination airports
    - French school holidays by zone
    - National holidays for origin/destination countries
    
    Args:
        df: DataFrame with basic flight data
        
    Returns:
        DataFrame enriched with external features
    """
    # Identify destination IATA once for all joins
    df['AirportDestination'] = np.where(df['Direction'].str.startswith('Arr'), 'LYS', df['SysStopover'])
    
    # Apply external data enrichments
    df = _add_weather_signals(df)
    df = _add_school_holidays(df)
    df = _add_national_holidays(df)
    
    # Cleanup intermediate helpers
    df = df.drop(columns=['AirportDestination'], errors='ignore')
    
    return df

def _add_weather_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins weather data for both origin and destination airports.
    
    Enriches flights with temperature and precipitation data:
    - Origin airport: temperature and precipitation at departure
    - Destination airport: precipitation at arrival (mainly LYS)
    
    Args:
        df: DataFrame with flight data
        
    Returns:
        DataFrame with weather features added
    """
    from saint_ex.config import WEATHER_FILE
    if not os.path.exists(WEATHER_FILE):
        return df

    weather = pd.read_csv(WEATHER_FILE)
    weather['date'] = pd.to_datetime(weather['date']).dt.date
    df['date_only'] = df['LTScheduledDatetime'].dt.date
    
    # Join for Origin (Temp/Rain at departure point)
    df = df.merge(
        weather[['date', 'iata', 'temp_max', 'precip']],
        left_on=['date_only', 'AirportOrigin'],
        right_on=['date', 'iata'],
        how='left'
    ).rename(columns={'temp_max': 'temp_max_origin', 'precip': 'precip_origin'})
    
    # Join for Destination (Rain at arrival point, mainly LYS)
    df = df.merge(
        weather[['date', 'iata', 'precip']],
        left_on=['date_only', 'AirportDestination'],
        right_on=['date', 'iata'],
        how='left'
    ).rename(columns={'precip': 'precip_dest'})
    
    # Cleanup join artifacts
    drop_cols = ['date_x', 'iata_x', 'date_y', 'iata_y', 'date_only']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Fill missing weather values with sensible defaults
    df['precip_origin'] = df['precip_origin'].fillna(0)  # No rain = 0
    df['precip_dest'] = df['precip_dest'].fillna(0)
    df['temp_max_origin'] = df['temp_max_origin'].fillna(20)  # Mild temp
    
    return df
def _add_school_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds French school holiday indicators by zone.
    
    France has three school holiday zones (A, B, C) with different calendars.
    This function creates binary indicators for each zone.
    
    Args:
        df: DataFrame with flight data
        
    Returns:
        DataFrame with school holiday indicators added
    """
    from saint_ex.config import SCHOOL_CAL_FILE
    if not os.path.exists(SCHOOL_CAL_FILE):
        return df
    
    school_cal = pd.read_csv(SCHOOL_CAL_FILE)
    school_cal['start'] = pd.to_datetime(school_cal['start']).dt.date
    school_cal['end'] = pd.to_datetime(school_cal['end']).dt.date
    
    df['date'] = df['LTScheduledDatetime'].dt.date
    
    # Create holiday indicators for each zone
    for zone in ['Zone A', 'Zone B', 'Zone C']:
        z_col = f"is_school_holiday_{zone.lower().replace(' ', '_')}"
        df[z_col] = 0
        z_cal = school_cal[school_cal['zone'] == zone]
        
        # Mark dates within holiday periods
        for _, row in z_cal.iterrows():
            mask = (df['date'] >= row['start']) & (df['date'] <= row['end'])
            df.loc[mask, z_col] = 1
    
    df = df.drop(columns=['date'])
    return df

def _add_national_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds national holiday indicators for origin and destination countries.
    
    Creates binary indicators for whether a flight date falls on a national
    holiday in either the origin or destination country.
    
    Args:
        df: DataFrame with flight data
        
    Returns:
        DataFrame with national holiday indicators added
    """
    import holidays
    try:
        import airportsdata
        AIRPORTS = airportsdata.load('IATA')
    except Exception:
        AIRPORTS = {}

    def get_country(iata_code: str) -> Optional[str]:
        """Get country code from IATA airport code."""
        if not isinstance(iata_code, str):
            return None
        apt = AIRPORTS.get(iata_code)
        return apt['country'] if apt else None

    df['date'] = df['LTScheduledDatetime'].dt.date
    df['origin_country'] = df['AirportOrigin'].map(get_country)
    df['dest_country'] = df['AirportDestination'].map(get_country)
    
    # Get unique countries and cache their holidays
    unique_countries = set(df['origin_country'].dropna().unique()) | set(df['dest_country'].dropna().unique())
    holiday_cache = {
        country: set(holidays.country_holidays(country, years=[2023, 2024, 2025, 2026]).keys())
        for country in unique_countries
    }
    
    # Create holiday indicators
    df['is_origin_holiday'] = df.apply(
        lambda r: int(r['date'] in holiday_cache.get(r['origin_country'], set())), axis=1
    )
    df['is_destination_holiday'] = df.apply(
        lambda r: int(r['date'] in holiday_cache.get(r['dest_country'], set())), axis=1
    )
    
    # Cleanup intermediate columns
    df = df.drop(columns=['date', 'origin_country', 'dest_country'])
    return df

def split_historical_inference(df: pd.DataFrame, val_ratio: float = 0.15, snapshot_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs chronological temporal split of the dataset.
    
    Args:
        df: Complete dataset with temporal ordering
        val_ratio: Fraction of historical data to use for validation
        snapshot_date: Optional override for the inference start date (used for backtesting)
        
    Returns:
        Tuple of (train_df, val_df, inference_df) with chronological splits
    """
    if snapshot_date:
        snapshot_cutoff = pd.to_datetime(snapshot_date)
    else:
        snapshot_cutoff = pd.to_datetime(INFERENCE_START_DATE)
    
    # Separate historical from future data
    historical = df[df['LTScheduledDatetime'] < snapshot_cutoff].copy()
    inference = df[df['LTScheduledDatetime'] >= snapshot_cutoff].copy()
    
    # IMPORTANT: Historical training/validation data MUST have realized labels and be commercial
    if 'NbPaxTotal' in historical.columns:
        historical = historical[
            historical['is_commercial'] & 
            historical['NbPaxTotal'].notna() & 
            (historical['NbPaxTotal'] > 0)
        ].copy()

    
    # Split historical data into train/validation chronologically
    historical = historical.sort_values(by='LTScheduledDatetime')
    cutoff_idx = int(len(historical) * (1 - val_ratio))
    
    train = historical.iloc[:cutoff_idx].copy()
    val = historical.iloc[cutoff_idx:].copy()
    
    return train, val, inference
