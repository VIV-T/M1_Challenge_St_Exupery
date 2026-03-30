"""
config.py — Paths, dates, and constants.

This is the ONLY file you need to edit to change the pipeline behavior.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_FILE  = Path('mouvements_aero_insa.csv')
OUTPUT_DIR = Path('outputs_new')
SEED       = 42

# ── Data Split Configuration ──────────────────────────────────────────────────
# This date defines the boundary between Training and Inference.
# Everything BEFORE this date is for Training/Validation.
# Everything AFTER this date is for Inference/Benchmarking.
# Example: Use '2026-03-01' for Full March, '2026-03-24' for Edge of Reality.
INFERENCE_START_DATE = '2026-03-24'

# ── Columns to load from CSV ─────────────────────────────────────────────────
# We only load a subset of the ~200 columns to save memory.
COLUMNS = [
    # Core IDs
    'IdMovement', 'FlightNumberNormalized', 'IdTraficType', 
    'IdBusinessUnitType',

    # Temporal & Static
    'LTScheduledDatetime', 'Direction', 'Terminal',
    'airlineOACICode', 'OperatorOACICodeNormalized',
    'SysStopover', 'AirportOrigin', 'IdAircraftType',

    # Flight Attributes (all confirmed SAFE by BigQuery audit)
    'NbOfSeats', 'ServiceCode', 'ScheduleType',
    'NbAirbridge', 'NbConveyor',
    'IdBusContactType', 'IdTerminalType', 'IdBagStatusDelivery',
    'FuelProvider',

    # Target Outcomes & PRM
    'NbPaxTotal', 'flight_with_pax',
    'OzionNbReservations'
]

# End of config
