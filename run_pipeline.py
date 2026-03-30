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
    
    # Store original pool date range for accurate reporting
    original_pool_min = pool_df['LTScheduledDatetime'].min()
    original_pool_max = pool_df['LTScheduledDatetime'].max()

    # 2. Sequential Feature Engineering
    print("\n[STAGE 1] Engineering Dynamic Features...")
    t0 = time.time()
    
    # 📈 Compute Route Signatures (Historical Yield) from Training Data only
    print("  Calculating Historical Route Signatures...")
    from saint_ex.features import get_route_stats
    route_stats = get_route_stats(train_df)
    
    # 💡 IMPORTANT: Combine ALL data to allow rolling Momentum windows to bridge the Jan 1st gap
    print("  Ensuring Time-Series Momentum Continuity...")
    combined_df = pd.concat([train_df, val_df, pool_df], axis=0).sort_values('LTScheduledDatetime')
    combined_df = add_features(combined_df, reference_stats=route_stats)
    
    # Re-split maintaining feature integrity
    train_df = combined_df[combined_df.index.isin(train_df.index)].copy()
    val_df   = combined_df[combined_df.index.isin(val_df.index)].copy()
    pool_df  = combined_df[combined_df.index.isin(pool_df.index)].copy()

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
    prm_model.train(X_train, train_df['FarmsNbPaxPHMR'], X_val, val_df['FarmsNbPaxPHMR'])
    print(f"  Training Complete ({time.time() - t0:.1f}s).")

    # 4. Save Feature Importance Manifest for Reporting
    importance_df = pd.DataFrame({
        'Feature': cols,
        'Importance': pax_model.model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "importance.csv"), index=False)
    print(f"  ✅ Feature Importance manifest exported to {OUTPUT_DIR}/importance.csv")

    # 5. Generate Final Predictions
    pool_start = original_pool_min.strftime('%Y')
    pool_end = original_pool_max.strftime('%Y')
    if pool_start == pool_end:
        period_desc = f"{pool_start}"
    else:
        period_desc = f"{pool_start}-{pool_end}"
    print(f"\n[STAGE 3] Generating Future Period Predictions ({period_desc})...")
    t0 = time.time()
    results = pool_df[['IdMovement', 'LTScheduledDatetime', 'Direction', 'airlineOACICode', 'SysStopover', 'AirportOrigin']].copy()
    
    results['Pred_NbPaxTotal'] = pax_model.predict(X_pool)
    results['Pred_FarmsNbPaxPHMR'] = prm_model.predict(X_pool)

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
