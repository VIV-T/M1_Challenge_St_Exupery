# Project Saint-Exupéry - Clean Codebase Documentation

## Overview

This is a cleaned and well-documented version of the Lyon-Saint Exupéry Airport passenger flow prediction system. The codebase has been refactored for maximum readability, maintainability, and human understanding.

## 🏗️ Architecture

### Core Modules

```
saint_ex/
├── config.py          # Central configuration hub
├── preprocessing.py    # Data ingestion and cleaning
├── features.py        # Feature engineering pipeline
├── models.py          # Machine learning models
├── evaluation.py      # Validation and metrics
└── viz.py            # Visualization utilities
```

### Pipeline Entry Points

```
├── run_pipeline.py    # Main training and prediction pipeline
└── check_data_leakage.py  # Data leakage analysis utility
```

## 📋 Module Documentation

### `config.py` - Configuration Hub

**Purpose**: Central configuration management with clear documentation and type hints.

**Key Sections**:
- **BigQuery Configuration**: Database connection settings
- **File Paths**: Data files and output directories
- **Temporal Splits**: Training/validation/inference boundaries
- **Model Hyperparameters**: LightGBM configurations for Pax/PRM models
- **Feature Engineering**: Feature lists and categorical variables
- **Business Rules**: Commercial flight validation criteria
- **Performance**: Early stopping and monitoring parameters

**Example Usage**:
```python
from saint_ex.config import INFERENCE_START_DATE, LGB_PAX_PARAMS
print(f"Training data up to: {INFERENCE_START_DATE}")
print(f"Model params: {LGB_PAX_PARAMS['objective']}")
```

---

### `preprocessing.py` - Data Pipeline

**Purpose**: Handles data ingestion, cleaning, filtering, and temporal splitting.

**Key Functions**:

#### `load_dataset()`
- **Input**: Optional file path (defaults to config)
- **Output**: Cleaned DataFrame with commercial flights
- **Features**: Auto BigQuery/CSV toggle, schema normalization, external data joins

#### `split_historical_inference()`
- **Purpose**: Chronological temporal split preventing data leakage
- **Returns**: (train_df, val_df, inference_df)
- **Logic**: Respects `INFERENCE_START_DATE` boundary

#### External Data Enrichment
- **Weather**: Temperature and precipitation for origin/destination
- **School Holidays**: French zones A, B, C indicators
- **National Holidays**: Country-specific holiday indicators

**Data Flow**:
```
Raw Data → Schema Normalization → Commercial Filtering → External Joins → Temporal Split
```

---

### `models.py` - Machine Learning Core

**Purpose**: Implements specialized LightGBM models for different target distributions.

#### `PaxModel` - Total Passenger Prediction
- **Approach**: Occupancy-weighted regression (passengers/seat ratio)
- **Benefits**: Stable training across different aircraft sizes
- **Loss Function**: L1 (MAE) for robust passenger count prediction
- **Target Transformation**: `occupancy = passengers / seats` (clipped to 1.2)

#### `PRMModel` - Reduced Mobility Prediction
- **Approach**: Direct count prediction with Tweedie loss
- **Specialization**: Handles low-volume, zero-heavy count data
- **Loss Function**: Tweedie distribution (ideal for count data)
- **Use Case**: Critical for operational planning

**Training Pipeline**:
```
Features → Target Transform → LightGBM Training → Early Stopping → Validation Metrics
```

---

### `features.py` - Feature Engineering

**Purpose**: Creates temporal, historical, and external features for model training.

**Feature Categories**:

#### Temporal Features
- **Cyclic Encoding**: `sin_hour`, `cos_hour`, `sin_month`, `cos_month`
- **Time Components**: `hour`, `dayofweek`, `month`, `dayofyear`

#### Historical Features
- **Route Signatures**: Historical occupancy averages by airline/origin
- **Lag Features**: `NbPax_Lag_7d`, `NbPax_Lag_14d` (same flight, previous weeks)
- **Momentum**: `hub_momentum_7d`, `route_momentum_7d` (recent trends)

#### External Features
- **Weather**: Temperature, precipitation interactions
- **Religious Events**: Hijri calendar alignment (`days_from_eid`, `return_surge`)
- **Holidays**: School and national holiday indicators

#### Flight Attributes
- **Infrastructure**: `NbOfSeats`, `NbConveyor`, `NbAirbridge`
- **Service Type**: `is_arrival`, `is_charter`

---

### `evaluation.py` - Validation Framework

**Purpose**: Comprehensive model validation with multiple resolution levels.

**Validation Modes**:

#### `evaluate_predictions()`
- **Live Validation**: Syncs with BigQuery ground truth
- **Metrics**: Flight, hourly, daily accuracy calculations
- **Visualizations**: Automatic professional plot generation

#### `run_historical_backtest()`
- **Purpose**: Multi-year stability audit
- **Windows**: Hijri surges, summer peaks, seasonal patterns
- **Output**: Cross-window performance comparison

**Metrics Calculated**:
- **Flight Level**: MAE per flight, accuracy percentage
- **Hourly Level**: Aggregated passenger flow by hour
- **Daily Level**: Total daily passenger counts
- **Global**: Sum ratio (long-term bias detection)

---

### `run_pipeline.py` - Main Orchestrator

**Purpose**: Complete training and prediction pipeline with multiple execution modes.

**Execution Modes**:

#### Standard Mode (`python run_pipeline.py`)
- Data loading and preprocessing
- Feature engineering
- Model training
- Prediction generation
- **No validation** (fast iteration)

#### Validation Mode (`python run_pipeline.py --validate`)
- Everything from standard mode +
- Live BigQuery validation
- Performance metrics
- Professional visualizations
- Export to `exports/plots/`

#### Backtest Mode (`python run_pipeline.py --backtest`)
- Historical stability audit
- Multiple time window testing
- **No current predictions** (robustness testing)

**Pipeline Stages**:
```
1. Data Loading → 2. Feature Engineering → 3. Model Training → 4. Prediction → 5. Validation
```

---

## 🎯 Key Design Principles

### 1. **Temporal Integrity**
- Strict chronological splits prevent data leakage
- Future information never contaminates training
- Rolling features respect temporal boundaries

### 2. **Occupancy-Weighted Approach**
- Passenger/seat ratio stabilizes training
- Aircraft size independence
- Better generalization across route types

### 3. **Specialized Models**
- Pax model: Occupancy regression with MAE loss
- PRM model: Count prediction with Tweedie loss
- Appropriate loss functions for target distributions

### 4. **External Data Integration**
- Weather, holidays, religious events
- Automatic fallback handling
- Sensible default imputation

### 5. **Comprehensive Validation**
- Multiple resolution levels (flight/hourly/daily)
- Live BigQuery ground truth comparison
- Historical stability backtesting

## 🔧 Configuration Management

All parameters are centralized in `config.py` with clear documentation:

```python
# Temporal boundaries
INFERENCE_START_DATE = '2025-01-01'

# Model hyperparameters
LGB_PAX_PARAMS = {
    'objective': 'regression_l1',
    'n_estimators': 3000,
    # ... fully documented parameters
}

# Business rules
VALID_SERVICE_CODES = ['J', 'S', 'C', 'G', 'O', 'L']
```

## 📊 Performance Monitoring

### Built-in Metrics
- **Flight Accuracy**: Individual flight prediction quality
- **Hourly Accuracy**: Operational resource planning
- **Daily Accuracy**: Terminal volume management
- **Global Reliability**: Long-term bias detection

### Visual Reports
- Daily momentum trends
- Hourly distribution patterns
- Weekly signatures
- Feature importance
- Residual analysis

## 🚀 Usage Examples

### Quick Training
```bash
source .venv/bin/activate
python run_pipeline.py
```

### Full Validation
```bash
python run_pipeline.py --validate
```

### Robustness Testing
```bash
python run_pipeline.py --backtest
```

### Data Leakage Check
```bash
python check_data_leakage.py
```

## 🧪 Testing and Validation

### Data Leakage Prevention
- Zero ID overlap between splits
- Chronological feature engineering
- Route overlap analysis documented

### Model Robustness
- Multi-year backtesting
- Seasonal pattern validation
- Religious event handling

### Performance Benchmarks
- Flight Accuracy: ~88-90%
- Hourly Accuracy: ~93-95%
- Daily Accuracy: ~96-97%
- PRM Accuracy: ~49% (challenging low-volume target)

## 📁 File Organization

```
Project Root/
├── saint_ex/           # Core package
│   ├── config.py       # Configuration (EDIT THIS)
│   ├── preprocessing.py # Data pipeline
│   ├── features.py     # Feature engineering
│   ├── models.py       # ML models
│   ├── evaluation.py   # Validation
│   └── viz.py         # Visualizations
├── run_pipeline.py     # Main entry point
├── check_data_leakage.py # Utility script
├── outputs_new/        # Results directory
├── exports/            # Visualizations
├── externals/          # External data caches
└── .venv/             # Virtual environment
```

## 🔍 Code Quality Improvements

### Before Cleanup
- Minimal documentation
- Inconsistent naming
- Scattered configuration
- No type hints
- Unclear function purposes

### After Cleanup
- ✅ Comprehensive docstrings
- ✅ Type hints throughout
- ✅ Centralized configuration
- ✅ Clear function signatures
- ✅ Logical module organization
- ✅ Business rule documentation
- ✅ Performance monitoring
- ✅ Error handling and fallbacks

## 🎓 Learning Resources

This codebase demonstrates several machine learning best practices:

1. **Temporal Validation**: Proper time series splitting
2. **Feature Engineering**: Domain-specific feature creation
3. **Model Selection**: Appropriate algorithms for target distributions
4. **External Data**: Real-world data integration challenges
5. **Production Pipeline**: End-to-end ML system design
6. **Validation Framework**: Comprehensive performance assessment

## 📞 Support and Maintenance

The cleaned codebase is designed for:
- **Easy modification**: Centralized configuration
- **Clear debugging**: Well-documented functions
- **Simple extension**: Modular design
- **Reliable operation**: Robust error handling
- **Performance monitoring**: Built-in metrics and visualization

---

**This documentation reflects a production-ready, well-maintained machine learning system that follows industry best practices for code quality, documentation, and operational reliability.**
