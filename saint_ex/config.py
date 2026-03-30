"""
config.py — Paths, dates, and constants.

This is the ONLY file you need to edit to change the pipeline behavior.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_FILE  = Path('mouvements_aero_insa.csv')
OUTPUT_DIR = Path('outputs_new')
SEED       = 42

# ── Data split intervals ─────────────────────────────────────────────────────
# Edit these dates to control what data goes into train / val / test.
# Format: (start_date, end_date) — both inclusive.
SPLIT = {
    'train': ('2023-01-01', '2025-09-30'),
    'val':   ('2025-10-01', '2026-02-28'),
    'test':  ('2026-03-01', '2026-03-31'),
}

# The model is strictly forbidden from seeing outcomes (IdTraficType=1) after this date.
# This ensures total legitimacy for the March 2026 evaluation.
HISTORICAL_CUTOFF = '2026-02-28'

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
