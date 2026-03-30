"""
config.py — Paths, dates, and constants.

This is the ONLY file you need to edit to change the pipeline behavior.
"""
from pathlib import Path

# ── BigQuery Configuration ──────────────────────────────────────────────────
USE_BIGQUERY = True
BQ_PROJECT   = "va-sdh-adl-staging"
BQ_DATASET   = "aero_insa"
BQ_TABLE     = "mouvements_aero_insa"
BQ_CREDS     = "insa/va-sdh-adl-staging.json"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_FILE       = Path('mouvements_aero_insa.csv')
WEATHER_FILE    = Path('externals/weather_hubs.csv')
SCHOOL_CAL_FILE = Path('externals/school_holidays.csv')
OUTPUT_DIR      = Path('outputs_new')
SEED            = 42

# ── Data Split Configuration ──────────────────────────────────────────────────
# Temporal Boundary for Blind Testing
INFERENCE_START_DATE = '2025-01-01'
OCCUPANCY_CLIP = 1.2

# ── Model Hyperparameters ────────────────────────────────────────────────────
LGB_PAX_PARAMS = {
    'objective': 'regression_l1',
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'num_leaves': 127,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'cat_smooth': 10,
    'random_state': SEED,
    'verbosity': -1
}

XGB_PAX_PARAMS = {
    'objective': 'reg:absoluteerror',
    'n_estimators': 800,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': SEED,
    'n_jobs': -1,
    'tree_method': 'hist',
    'device': 'cpu',
    'enable_categorical': True
}

LGB_PRM_PARAMS = {
    'objective': 'tweedie',
    'n_estimators': 800,
    'learning_rate': 0.02,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'random_state': SEED,
    'verbosity': -1
}

# ── Feature Manifest ──────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    'airlineOACICode', 'OperatorOACICodeNormalized', 'SysStopover',
    'AirportOrigin', 'IdAircraftType', 'Terminal', 'ServiceCode',
    'FuelProvider'
]

ALL_FEATURES = [
    'NbOfSeats', 'NbConveyor', 'NbAirbridge',
    'is_arrival', 'is_charter',
    'temp_max_origin', 'precip_origin', 'precip_dest',
    'is_origin_holiday', 'is_destination_holiday',
    'days_from_eid', 'return_surge', 'hub_pressure',
    'NbPax_Lag_7d', 'NbPax_Lag_14d', 'route_avg_occupancy',
    'hub_momentum_7d', 'route_momentum_7d', 'minute_of_day', 
    'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
    'rain_arrival_impact', 'heat_stress',
    'is_school_holiday_zone_a', 'is_school_holiday_zone_b', 'is_school_holiday_zone_c',
    *CATEGORICAL_FEATURES,
]

# ── CSV Loading Filter ───────────────────────────────────────────────────────
LOAD_COLUMNS = [
    'IdMovement', 'FlightNumberNormalized', 'IdTraficType', 
    'IdBusinessUnitType', 'LTScheduledDatetime', 'Direction', 'Terminal',
    'airlineOACICode', 'OperatorOACICodeNormalized',
    'SysStopover', 'AirportOrigin', 'IdAircraftType',
    'NbOfSeats', 'ServiceCode', 'ScheduleType',
    'NbAirbridge', 'NbConveyor',
    'IdBusContactType', 'IdTerminalType', 'IdBagStatusDelivery',
    'FuelProvider', 'NbPaxTotal', 'flight_with_pax',
    'FarmsNbPaxPHMR'
]
