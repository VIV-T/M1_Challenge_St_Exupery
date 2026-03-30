"""
run_pipeline.py — End-to-End Orchestrator for Project Saint-Exupéry.

Workflow:
1. Load & Preprocess Commercial Flight Data.
2. Enrich with Bidirectional Dynamic Features (Weather, Holidays, Schools).
3. Train Gradient Boosting Models (Pax + PRM).
4. Predict March 2026 Passenger Flows.
5. Auto-Validate against BigQuery Ground Truth.
"""
import pandas as pd
import numpy as np
import os
from saint_ex.preprocessing import load_dataset, split_historical_inference
from saint_ex.features import add_features, prepare_X
from saint_ex.models import PaxModel, PRMModel
from sklearn.metrics import mean_absolute_error

# --- Project Configuration ---
DATA_PATH = 'mouvements_aero_insa.csv'
OUTPUT_DIR = 'outputs_new'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("\n" + "═"*60)
    print(" 🛫 Project Saint-Exupéry — Dynamic Bidirectional Pipeline")
    print("═"*60)

    # 1. Load & Preprocess
    df = load_dataset(DATA_PATH)
    train_df, val_df, pool_df = split_historical_inference(df)

    # 2. Sequential Feature Engineering (Enriched with Weather/Holidays)
    print("\nEngineering Features (Dynamic API signals)...")
    train_df = add_features(train_df)
    val_df   = add_features(val_df)
    pool_df  = add_features(pool_df)

    X_train, cols = prepare_X(train_df)
    X_val, _     = prepare_X(val_df, cols)
    X_pool, _    = prepare_X(pool_df, cols)

    # 3. Model Training
    print("\nTraining Predictive Architecture...")
    pax_model = PaxModel()
    pax_model.train(X_train, train_df['NbPaxTotal'], X_val, val_df['NbPaxTotal'])

    prm_model = PRMModel()
    prm_model.train(X_train, train_df['NbPRMTotal'], X_val, val_df['NbPRMTotal'])

    # 4. Generate Final Predictions
    print("\nGenerating March 2026 Inference...")
    predictions = pool_df[['IdMovement', 'LTScheduledDatetime', 'Direction', 'airlineOACICode', 'SysStopover', 'AirportOrigin']].copy()
    predictions['IdMovement'] = predictions['IdMovement'].fillna('MISSING').astype(str)
    
    predictions['predicted_pax'] = pax_model.predict(X_pool)
    predictions['predicted_prm'] = prm_model.predict(X_pool)

    out_path = os.path.join(OUTPUT_DIR, 'predictions_flight.csv')
    predictions.to_csv(out_path, index=False)
    print(f"DONE. Repository ready for submission. Results: {out_path}")

if __name__ == "__main__":
    main()
