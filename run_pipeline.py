import pandas as pd
import numpy as np
import os
import time
import argparse
from saint_ex.preprocessing import load_dataset, split_historical_inference
from saint_ex.features import add_features, prepare_X
from saint_ex.models import PaxModel, PRMModel
from saint_ex.config import DATA_FILE, OUTPUT_DIR
from saint_ex.evaluation import evaluate_predictions, run_historical_backtest

def main():
    parser = argparse.ArgumentParser(description="🛫 Project Saint-Exupéry Master Pipeline Orchestrator")
    parser.add_argument("--validate", action="store_true", help="Sync with BigQuery for ground truth validation + Visuals")
    parser.add_argument("--backtest", action="store_true", help="Execute 3-year historical stability audit")
    args = parser.parse_args()

    start_total = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "═" * 60)
    print(" 🛫 Project Saint-Exupéry — High-Performance Predictive Pipeline")
    print("═" * 60)

    # 1. Load & Preprocess
    df = load_dataset(DATA_FILE)
    
    if args.backtest:
        run_historical_backtest(df)
        return

    # Standard / Validate Mode
    train_df, val_df, pool_df = split_historical_inference(df)

    # 2. Sequential Feature Engineering
    print("\n[STAGE 1] Engineering Dynamic Features...")
    t0 = time.time()
    train_df = add_features(train_df)
    val_df   = add_features(val_df)
    pool_df  = add_features(pool_df)

    X_train, cols = prepare_X(train_df)
    X_val, _     = prepare_X(val_df, cols)
    X_pool, _    = prepare_X(pool_df, cols)
    print(f"  Engineering Complete ({time.time() - t0:.1f}s). Total Features: {len(cols)}")

    # 3. Model Training
    print("\n[STAGE 2] Training Predictive Architecture...")
    t0 = time.time()
    pax_model = PaxModel()
    pax_model.train(X_train, train_df['NbPaxTotal'], X_val, val_df['NbPaxTotal'])

    prm_model = PRMModel()
    prm_model.train(X_train, train_df['NbPRMTotal'], X_val, val_df['NbPRMTotal'])
    print(f"  Training Complete ({time.time() - t0:.1f}s).")

    # 4. Save Feature Importance Manifest for Reporting
    importance_df = pd.DataFrame({
        'Feature': pax_model.features,
        'Importance': pax_model.model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "importance.csv"), index=False)
    print(f"  ✅ Feature Importance manifest exported to {OUTPUT_DIR}/importance.csv")

    # 5. Generate Final Predictions
    print("\n[STAGE 3] Generating 2026 Inference Pool...")
    t0 = time.time()
    results = pool_df[['IdMovement', 'LTScheduledDatetime', 'Direction', 'airlineOACICode', 'SysStopover', 'AirportOrigin']].copy()
    results['IdMovement'] = results['IdMovement'].fillna('MISSING').astype(str)
    
    results['predicted_pax'] = pax_model.predict(X_pool)
    results['predicted_prm'] = prm_model.predict(X_pool)

    out_path = os.path.join(OUTPUT_DIR, 'predictions_flight.csv')
    results.to_csv(out_path, index=False)
    
    print(f"  Inference Complete ({time.time() - t0:.1f}s).")
    
    # 6. Optional Live Validation
    if args.validate:
        evaluate_predictions(results)

    print("\n" + "═" * 60)
    print(f" ✅ PIPELINE SUCCESS. Total Time: {time.time() - start_total:.1f}s")
    print(f" 📂 Prediction Artifact: {out_path}")
    print("═" * 60)

if __name__ == "__main__":
    main()
