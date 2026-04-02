# AI Challenge: Lyon-Saint Exupéry Airport flows
**Anticipating passenger peaks for a smoother and more sustainable airport experience.**

---

## Team Members
- **Baptiste TRON**
- **Thibault VIVIER**
- **Ahmed MANSOUR**
- **Louis-Marie LOE**

---

## Project Overview
This project was developed for the **AI Challenge 2026** in collaboration with **Lyon-Saint Exupéry Airport**. The central goal is to leverage Artificial Intelligence to accurately forecast passenger flows per flight. 

By anticipating traffic peaks, the airport can:
- **Optimize Resource Allocation**: Security agents, border police, and staff.
- **Improve Passenger Experience**: Reduce waiting times and congestion.
- **Control Costs & Sustainability**: Avoid unnecessary infrastructure openings (lighting, energy).
- **Specialized PRM Support**: Enhance management for Passengers with Reduced Mobility (PRM).

## Key Features
- **End-to-End Automated Pipeline**: From BigQuery ingestion to hourly aggregated predictions.
- **Dynamic Prediction Windows**: Control look-ahead and validation periods via command-line arguments.
- **Advanced Feature Engineering**: 
  - Automated calculation of lags, rolling averages, and trends.
  - Integration of temporal features (day of week, month, holidays).
- **Dual-Model Architecture**: Specialized LightGBM regressors for:
  1. `NbPaxTotal`: Total passenger volume per flight.
  2. `FarmsNbPaxPHMR`: Passengers with Reduced Mobility (PRM).
- **Robust Merging Logic**: Ensures absolute alignment between flight predictions for consistent reporting.
- **In-Depth Notebooks**:
  - `training.ipynb`: Step-by-step model training, exploration, and evaluation.
  - `analysis.ipynb` :  Analysis of results and proposal for optimizing staff management.

---

## Repository Structure
```text
M1_Challenge_St_Exupery/
├── config/                  # GCP Credentials and project configuration
├── data/                    # Storage for raw, preprocessed
├── models/                  # Serialized model weights (.pkl)
├── output/                  # Hourly predictions per day, planning management of staff
└── scripts/
    ├── analysis/
    │   ├── analysis.ipynb   # Notebook with analysis and staff optimization
    ├── data_preparation/    # ETL pipeline & Feature Engineering
    │   ├── get_main.py      # BigQuery data ingestion
    │   ├── preprocessed.py  # Data cleaning and splitting
    │   └── utils/           # Specialized feature functions (lags, holidays)
    └── training/            
        ├── pipeline.py      # Main automated end-to-end script
        └── training.ipynb   # Interactive training and evaluation notebook
```


## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/VIV-T/M1_Challenge_St_Exupery.git
cd M1_Challenge_St_Exupery
```

### 2. Environment Setup
We recommend using a virtual environment (Python 3.9+).
```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# or .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Google Cloud Credentials
To query the dataset from BigQuery, create a JSON key for your service account and save it at:
`config/va-sdh-adl-staging.json`

---

## Usage
The main entry point is the **`pipeline.py`** script. It automatically calculates the best training, validation, and test splits based on the current date.

### Basic Run
Predicts for the next **2 days** using a **7-day** validation window.
```bash
python scripts/training/pipeline.py
```

### Advanced Usage
You can customize the ranges using arguments:
```bash
# Predict the next 5 days, with 14 days of validation, and force re-training
python scripts/training/pipeline.py --predict-days 5 --val-days 14 --force-train
```

**Available Flags:**
- `--predict-days`: Number of days to predict into the future (default: 2).
- `--val-days`: Number of days to use for internal validation before the 1st prediction (default: 7).
- `--force-train`: Ignore existing `.pkl` models and re-train from scratch.

---

## Methodology & Pipeline
The project follows a rigorous temporal split strategy to avoid data leakage:
1. **Data Ingestion**: Queries the `mouvements_aero_insa` table via BigQuery.
2. **Preprocessing**: Cleans data, handles duplicates, and splits targets.
3. **Feature Engineering**: Generates 500+ features including lags, rolling statistics, and temporal interactions.
4. **Training**: LightGBM Gradien Boosting with early stopping (WAPE/MAE optimization).
5. **Inference**: Generates predictions for future flights.
6. **Aggregation**: Resamples flight-level data to hourly intervals for operational use.

---

---

## CSR Dimension (Social & Environmental Impact)
Our model contributes to several Corporate Social Responsibility (CSR) goals:
- **Environmental**: Reducing the carbon footprint by minimizing unnecessary energy consumption in terminals.
- **Social**: Improving the quality of life for airport staff by preventing workload intensification during unexpected peaks.
- **Inclusivity**: Special focus on PRM passengers to ensure equitable service quality.

See the full [CSR Analysis Report](CSR_Analysis_Report.pdf) for more details on the environmental and social impacts of this project.

---

