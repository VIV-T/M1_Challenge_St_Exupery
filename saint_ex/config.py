"""
Configuration module for Project Saint-Exupéry Airport Passenger Flow Prediction.

This module contains all configuration parameters, paths, and model hyperparameters
for the passenger flow prediction pipeline. It serves as the central configuration
hub that can be easily modified without changing the core pipeline logic.

Key Configuration Areas:
- BigQuery connection settings
- File paths and data locations  
- Temporal data split parameters
- Model hyperparameters for LightGBM
- Feature engineering specifications
- Data loading and filtering options
"""

from pathlib import Path
from typing import Dict, List, Any

# =============================================================================
# BIGQUERY CONFIGURATION
# =============================================================================
# Toggle between live BigQuery data and local CSV snapshot
USE_BIGQUERY = True

# BigQuery connection parameters
BQ_PROJECT = "va-sdh-adl-staging"
BQ_DATASET = "aero_insa" 
BQ_TABLE = "mouvements_aero_insa"
BQ_CREDS = "insa/va-sdh-adl-staging.json"

# =============================================================================
# FILE PATHS AND DIRECTORIES
# =============================================================================
# Data files
DATA_FILE = Path('mouvements_aero_insa.csv')  # Local CSV snapshot (fallback)
WEATHER_FILE = Path('externals/weather_hubs.csv')  # Weather data for airports
SCHOOL_CAL_FILE = Path('externals/school_holidays.csv')  # French school holidays

# Output directories
OUTPUT_DIR = Path('outputs_new')  # Main results directory

# Random seed for reproducibility
SEED = 42

# =============================================================================
# TEMPORAL DATA SPLIT CONFIGURATION
# =============================================================================
# Temporal boundary for blind testing - all data from this date forward is used
# for inference only (no training data from this period)
INFERENCE_START_DATE = '2025-01-01'

# Occupancy ratio clipping to prevent unrealistic predictions
# Values above this ratio are clipped (e.g., 1.2 = max 120% occupancy)
OCCUPANCY_CLIP = 1.2

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# LightGBM parameters for total passenger prediction (PaxModel)
# Uses L1 loss (MAE) for robust passenger count prediction
LGB_PAX_PARAMS: Dict[str, Any] = {
    'objective': 'regression_l1',           # MAE loss function
    'n_estimators': 3000,                   # Maximum number of trees
    'learning_rate': 0.01,                  # Conservative learning rate
    'num_leaves': 127,                      # Model complexity
    'feature_fraction': 0.8,               # Feature subsampling
    'bagging_fraction': 0.8,                # Data subsampling  
    'bagging_freq': 5,                      # Bagging frequency
    'cat_smooth': 10,                       # Categorical smoothing
    'random_state': SEED,
    'verbosity': -1                         # Suppress training output
}

# LightGBM parameters for PRM (Passengers with Reduced Mobility) prediction
# Uses Tweedie loss suitable for count data with many zeros
LGB_PRM_PARAMS: Dict[str, Any] = {
    'objective': 'tweedie',                 # Tweedie loss for count data
    'n_estimators': 800,
    'learning_rate': 0.02,
    'num_leaves': 31,                       # Simpler model for PRM
    'feature_fraction': 0.9,
    'random_state': SEED,
    'verbosity': -1
}

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Categorical features that require special handling
CATEGORICAL_FEATURES: List[str] = [
    'airlineOACICode',           # Airline identifier
    'OperatorOACICodeNormalized', # Normalized operator code
    'SysStopover',              # Destination/stopover airport
    'AirportOrigin',             # Origin airport
    'IdAircraftType',           # Aircraft type
    'Terminal',                 # Airport terminal
    'ServiceCode',              # Flight service type
    'FuelProvider'              # Fuel service provider
]

# Complete feature list for model training
# Order matters for feature importance interpretation
ALL_FEATURES: List[str] = [
    # Aircraft and infrastructure features
    'NbOfSeats', 'NbConveyor', 'NbAirbridge',
    
    # Flight type indicators
    'is_arrival', 'is_charter',
    
    # Weather features
    'temp_max_origin', 'precip_origin', 'precip_dest',
    
    # Holiday indicators
    'is_origin_holiday', 'is_destination_holiday',
    
    # Religious event features (Hijri calendar)
    'days_from_eid', 'return_surge', 'hub_pressure',
    
    # Historical lag features
    'NbPax_Lag_7d', 'NbPax_Lag_14d', 'route_avg_occupancy',
    
    # Momentum features
    'hub_momentum_7d', 'route_momentum_7d',
    
    # Temporal features
    'minute_of_day', 'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
    
    # Weather interaction features
    'rain_arrival_impact', 'heat_stress',
    
    # School holiday zones (France A, B, C)
    'is_school_holiday_zone_a', 'is_school_holiday_zone_b', 'is_school_holiday_zone_c',
    
    # Categorical features (listed above)
    *CATEGORICAL_FEATURES,
]

# =============================================================================
# DATA LOADING CONFIGURATION
# =============================================================================

# Columns to load from BigQuery/CSV to optimize memory usage
# Only essential columns are loaded to reduce memory footprint
LOAD_COLUMNS: List[str] = [
    # Primary identifiers
    'IdMovement', 'FlightNumberNormalized', 'IdTraficType', 
    'IdBusinessUnitType', 'LTScheduledDatetime', 'Direction', 'Terminal',
    
    # Airline and route information
    'airlineOACICode', 'OperatorOACICodeNormalized',
    'SysStopover', 'AirportOrigin', 'IdAircraftType',
    
    # Aircraft and service details
    'NbOfSeats', 'ServiceCode', 'ScheduleType',
    'NbAirbridge', 'NbConveyor',
    
    # Operational identifiers
    'IdBusContactType', 'IdTerminalType', 'IdBagStatusDelivery',
    'FuelProvider',
    
    # Target variables
    'NbPaxTotal', 'flight_with_pax', 'FarmsNbPaxPHMR'
]

# =============================================================================
# BUSINESS RULES AND VALIDATION
# =============================================================================

# Commercial flight service codes (included in training data)
# J: Scheduled, S: Scheduled Revenue, C: Charter, G: General Aviation
# O: Contract Charter, L: Charter mixed
VALID_SERVICE_CODES = ['J', 'S', 'C', 'G', 'O', 'L']

# Business unit type for commercial flights
COMMERCIAL_BUSINESS_UNIT = 1

# Minimum seats requirement for commercial flights
MIN_SEATS_FOR_COMMERCIAL = 0

# Flight with passenger indicator
FLIGHT_WITH_PAX_INDICATOR = 'Oui'

# =============================================================================
# PERFORMANCE AND MONITORING
# =============================================================================

# Early stopping parameters
EARLY_STOPPING_ROUNDS = 100
