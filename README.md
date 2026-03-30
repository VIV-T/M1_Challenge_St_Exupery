# 🛫 Project Saint-Exupéry: Dynamic Bidirectional Passenger Forecasting

This repository contains a high-performance machine learning pipeline designed to forecast passenger flows for Lyon-Saint Exupéry Airport. The architecture leverages **Dynamic Bidirectional Context**—automatically syncing global climate and calendar signals for both origin and destination airports to capture international demand surges.

## 🏛️ Core Architecture
Conventional airport models often focus solely on the local region. Project Saint-Exupéry implements a **Situationally Aware** engine:
*   **Bidirectional Climate:** Tracks Max Temperature and Precipitation for the top 50 international hubs (CDG, LHR, AMS, DXB, etc.).
*   **Tri-Zone Educational Awareness:** Dynamically monitors all three French school holiday zones (A, B, and C) via the National Education API.
*   **Global Holiday Engine:** Injects origin-country specific bank holidays for all 808 unique flight origins in the dataset.
*   **PRM-Adjusted Logic:** Dedicated sub-models for Passengers with Reduced Mobility (PRM) to ensure accessibility-compliant resource planning.

## 🚀 Performance Benchmarks (March 2026)
| Metric | Value |
| :--- | :--- |
| **Prediction Accuracy (Pax)** | **86.4 %** |
| **Flight MAE** | **15.87 passengers** |
| **PRM MAE** | **1.09 passengers** |

## 🛠️ Installation & Usage
1.  **Environment Setup:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Universal API Sync:**
    Refresh the local external signals cache (Weather/Schools):
    ```bash
    python update_external_data.py
    ```
3.  **Run Forecast Pipeline:**
    Train the models and generate March 2026 predictions:
    ```bash
    python run_pipeline.py
    ```

## 📂 Repository Structure
*   `run_pipeline.py`: Unified entry point (Train -> Predict -> Evaluate).
*   `update_external_data.py`: Dynamic API manager for global signals.
*   `saint_ex/`: Core engine package.
    *   `features.py`: Bidirectional feature engineering logic.
    *   `models.py`: LightGBM Gradient Boosting architecture.
    *   `preprocessing.py`: Commercial flight filtering and normalization.
*   `externals/`: Persistent cache for world weather and calendars.
*   `outputs_new/`: Final prediction artifacts for submission.

---
*Developed for the M1 Challenge - Project Saint-Exupéry 2026.*
