# 🛫 Project Saint-Exupéry: Airport Passenger Flow Prediction

High-performance predictive pipeline for anticipates hourly passenger peaks and PRM (Persons with Reduced Mobility) flows at Lyon-Saint Exupéry Airport.

## 🚀 Quick Start

### 1. Environment Setup
The project requires Python 3.10+ and a virtual environment.
```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. BigQuery Credentials
Ensure your Google Cloud service account JSON is located at:
`saint_ex/resources/bigquery_creds.json`

### 3. Generate Predictions (Production)
To generate the final 48-hour forecast for April 1st and 2nd, 2026:
```bash
.venv/bin/python run_pipeline.py
```
*Output: `outputs/predictions_flight.csv`*

---

## 🧪 Fair Backtesting & Evaluation

To test the model's accuracy on historical data **without any look-ahead bias**, use the dynamic backtest suite. It automatically handles temporal splitting and ensures the model only "sees" data available at that specific time.

### Test a specific window (e.g. February 2026)
```bash
.venv/bin/python dynamic_backtest.py --start 2026-02-01 --end 2026-03-01 --horizon 48
```

### Parameters:
- `--start`: The "Today" date for the model. Everything before this is for Training.
- `--end`: (Optional) Truncates the evaluation window.
- `--horizon`: (Default 48) Sets the forecast window in hours.

---

## 🏗️ Technical Architecture

### 1. Data Pipeline (`saint_ex/preprocessing.py`)
- **Dual Mode**: Supports both live BigQuery fetching and local CSV snapshots.
- **Normalization**: Standardizes aircraft capacity, airline codes, and flight types.
- **Deduplication**: Ensures one record per `IdADL` for stable evaluation.

### 2. Feature Engineering (`saint_ex/features.py`)
- **Hierarchical Lags**: 7-day and 14-day passenger counts at Flight -> Route -> Airline levels.
- **Airport Pulse (Momentum)**: 7-day rolling terminal occupancy with a **3-day safety shift** to account for real-world BQ update delays (9:00 AM next day).
- **Target Encoding**: High-resolution "Route-Month" yield signatures for seasonality.
- **External Signals**: Join logic for weather (origin/destination), French school holidays (Zones A/B/C), and national holidays.

### 3. Predictive Models (`saint_ex/models.py`)
- **PaxModel**: LightGBM regressor using **Occupancy-Weighted** transformation. This stabilizes predictions across different aircraft sizes.
- **PRMModel**: specialized LightGBM regressor using **Tweedie Loss**, ideal for count data with many zeros.

---

## 📅 Maintenance & Operations

### Recommended Frequency: **Daily**
For maximum operational accuracy, run the pipeline **every morning before 9:00 AM**.

**Why?**
- **Schedule Updates**: Captures last-minute airline equipment swaps or cancellations.
- **Fresh Momentum**: Uses the "freshest" 7-day window for the `Airport Pulse`.
- **Refinement**: Today's "Tomorrow" prediction becomes today's refined "Today" prediction.

---

## 📁 Repository Structure
```text
.
├── run_pipeline.py         # Main execution entry point
├── dynamic_backtest.py      # Fairness & Sensitivity evaluation suite
├── saint_ex/                # Core Package
│   ├── config.py            # Global hyperparams & paths
│   ├── preprocessing.py     # Data ingestion & cleaning
│   ├── features.py          # Signal extraction & Lags
│   ├── models.py            # LightGBM Pax & PRM implementations
│   ├── evaluation.py        # BQ Actuals link & Metics
│   └── resources/           # External data (Weather, Holidays, Keys)
├── outputs/                 # Prediction artifacts
└── README.md                # This document
```
