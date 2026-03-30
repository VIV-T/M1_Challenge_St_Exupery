# 🛫 Project Saint-Exupéry: High-Performance Passenger Forecasting

Project Saint-Exupéry is an advanced AI predictive pipeline designed for **Lyon-Saint Exupéry Airport (LYS)**. The mission is to provide high-precision, flight-level passenger and PRM (Passenger with Reduced Mobility) forecasts to optimize operational resources, security staffing, and terminal capacity.

---

## 🏛️ Scientific Methodology: "Fly with Data"

Unlike conventional models that predict raw passenger counts, our architecture uses **Occupancy-Weighted Yield Regression** to capture the fundamental "fullness" of flight routes.

### 1. Occupancy-Factor Transformation (Phase 1)
We transform the target variable into a relative yield:
$$ \text{Target} = \frac{\text{Actual Passengers}}{\text{Total Seats}} $$
This allows the model to learn route "popularity" and "seasonality" independently of the aircraft size. A full A320 and a full A380 now represent the same "100% Full" signal, which significantly stabilizes the Gradient Boosting trees.

### 2. Historical Route Signatures (Phase 2)
The engine calculates a **Historical Route Profile** for every `(Airline, Origin)` pair from 2023–2025 data. This acts as a "long-term memory," giving the model a baseline yield (reputation) for established travel corridors before adjusting for current weather or holiday surges.

### 3. High-Precision Tuning & Tweedie Loss (Phase 3)
*   **Pax Model**: Optimized with an **L1 (MAE) Objective** to focus specifically on minimizing absolute passenger error.
*   **PRM Model**: Leverages **Tweedie Regression**, a specialized objective designed for low-volume, zero-heavy count data (Passengers with Reduced Mobility).

---

## 🚀 Latest Validation Results (March 2026)

Our model has been validated against the **Live 2026 BigQuery dataset**, achieving exceptional operational precision.

### 🎯 Core Performance Indicators (2026)
| Metric | Result | Operational Utility |
| :--- | :--- | :--- |
| **Flight Accuracy** | **86.9 %** | Precise Gate & Resource Planning |
| **Hourly Accuracy** | **94.4 %** | Security & Border Staffing Optimization |
| **Daily Accuracy** | **96.7 %** | Global Terminal Volume Management |
| **Global Reliability**| **100.9 %** | Near-Zero Long-Term Bias |

### 🛠️ Strategic 15-Month Blind Audit
To prove robustness, we ran a "Time-Travel" audit (Trained strictly on 2023–2024) to predict all of 2025.
*   **Audit Accuracy**: **87.4 %**
*   **Volume Resilience**: The model navigated an entire year of unseen seasonality (Summer, Christmas, Holidays) with **97.0 % daily precision**.

---

## 📂 Visual Intelligence Suite
The pipeline automatically exports high-resolution assets to `exports/plots/` for stakeholder reporting:
*   **`daily_momentum.png`**: Multi-month volume trends.
*   **`hourly_distribution.png`**: Intra-day peak/trough analysis.
*   **`weekly_signature.png`**: Pattern recognition across Monday–Sunday.
*   **`feature_importance.png`**: Transparency into model decision-making.
*   **`residual_analysis.png`**: Quality audit of prediction errors.

---

## 🛠️ Usage Instructions

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Update External Signals (Weather & Holidays)
Refresh the dynamic local caches for global weather hubs and school calendars:
```bash
python update_external_data.py
```

### 3. Run Production Forecast
Generates the final 2026 prediction file for submission:
```bash
python run_pipeline.py
```

### 4. Run Live Validation
Syncs with BigQuery for Real-Time Accuracy Audits:
```bash
python run_pipeline.py --validate
```

---
*Developed for the Lyon-Saint Exupéry 2026 AI Challenge.*
