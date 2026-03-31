import pandas as pd
import numpy as np
import os
import time
from saint_ex.preprocessing import load_dataset, split_historical_inference
from saint_ex.features import add_features, prepare_X
from saint_ex.models import PaxModel, PRMModel
from saint_ex.config import DATA_FILE, OUTPUT_DIR, INFERENCE_START_DATE

def run_pax_pipeline(inference_start_date=None, val_ratio=0.15, forecast_horizon_hours=48, silent=False):

    """
    Main orchestrator for the Saint-Exupéry Passenger Flow Prediction Pipeline.
    Can be run as a standalone script or called programmatically for backtesting.
    """
    start_total = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not silent:
        print("\n" + "═" * 60)
        print(" 🛫 Project Saint-Exupéry — High-Performance Predictive Pipeline")
        print("═" * 60)

    # 1. Load & Preprocess
    df = load_dataset(DATA_FILE)
    
    # Optional override for backtesting
    if inference_start_date:
        train_df, val_df, pool_df = split_historical_inference(df, val_ratio=val_ratio, snapshot_date=inference_start_date)
    else:
        train_df, val_df, pool_df = split_historical_inference(df, val_ratio=val_ratio)
    
    original_pool_min = pool_df['LTScheduledDatetime'].min()
    original_pool_max = pool_df['LTScheduledDatetime'].max()

    # 2. Sequential Feature Engineering
    if not silent: print("\n[STAGE 1] Engineering Dynamic Features...")
    t0 = time.time()
    
    from saint_ex.features import get_route_stats
    route_stats = get_route_stats(train_df)
    
    train_df['__split__'] = 'train'
    val_df['__split__']   = 'val'
    pool_df['__split__']  = 'pool'
    
    combined_df = pd.concat([train_df, val_df, pool_df], axis=0).sort_values('LTScheduledDatetime')
    snapshot_date = inference_start_date or INFERENCE_START_DATE
    combined_df = add_features(combined_df, reference_stats=route_stats, split_date=snapshot_date)

    
    train_df = combined_df[combined_df['__split__'] == 'train'].copy()
    val_df   = combined_df[combined_df['__split__'] == 'val'].copy()
    pool_df  = combined_df[combined_df['__split__'] == 'pool'].copy()
    
    # Enforce forecast horizon if specified
    if forecast_horizon_hours is not None:
        max_forecast_date = original_pool_min + pd.Timedelta(hours=forecast_horizon_hours)
        pool_df = pool_df[pool_df['LTScheduledDatetime'] <= max_forecast_date].copy()

    
    # Prepare features for Training/Validation (Commercial only)
    X_train, cols = prepare_X(train_df[train_df['is_commercial'] == True])
    X_val, _     = prepare_X(val_df[val_df['is_commercial'] == True], cols)
    
    # Prepare features for Pool (Full set)
    X_pool, _    = prepare_X(pool_df, cols)

    
    if not silent:
        print(f"  Engineering Complete ({time.time() - t0:.1f}s). Total Features: {len(cols)}")
        print(f"  Inference Horizon: {original_pool_min} -> {pool_df['LTScheduledDatetime'].max()} ({len(pool_df)} flights)")

    # 3. Model Training
    if not silent: print("\n[STAGE 2] Training Predictive Architecture...")
    t0 = time.time()
    
    pax_model = PaxModel()
    pax_model.train(X_train, train_df.loc[train_df['is_commercial'] == True, 'NbPaxTotal'], 
                    X_val, val_df.loc[val_df['is_commercial'] == True, 'NbPaxTotal'], silent=silent)

    prm_model = PRMModel()
    prm_model.train(X_train, train_df.loc[train_df['is_commercial'] == True, 'FarmsNbPaxPHMR'], 
                    X_val, val_df.loc[val_df['is_commercial'] == True, 'FarmsNbPaxPHMR'], silent=silent)

    
    if not silent: print(f"  Training Complete ({time.time() - t0:.1f}s).")

    # 4. Save Feature Importance (only in main mode)
    if not silent:
        importance_df = pd.DataFrame({
            'Feature': cols,
            'Importance': pax_model.model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        importance_df.to_csv(os.path.join(OUTPUT_DIR, "importance.csv"), index=False)

    # 5. Generate Final Predictions
    if not silent: print(f"\n[STAGE 3] Generating Future Period Predictions (All Flights)...")
    t0 = time.time()
    
    results = pool_df[['IdADL', 'IdMovement', 'LTScheduledDatetime', 'Direction', 'airlineOACICode', 'SysStopover', 'AirportOrigin', 'is_commercial']].copy()
    
    # Generate predictions for all rows
    results['Pred_NbPaxTotal'] = pax_model.predict(X_pool)
    results['Pred_FarmsNbPaxPHMR'] = prm_model.predict(X_pool)
    
    # Force zero for non-commercial flights (Freight, Ferry, Technical)
    results.loc[results['is_commercial'] == False, 'Pred_NbPaxTotal'] = 0.0
    results.loc[results['is_commercial'] == False, 'Pred_FarmsNbPaxPHMR'] = 0.0
    
    # Cleanup output columns
    results = results.drop(columns=['is_commercial'])

    out_path = os.path.join(OUTPUT_DIR, 'predictions_flight.csv')
    results.to_csv(out_path, index=False)

    
    if not silent:
        print(f"  Inference Complete ({time.time() - t0:.1f}s).")
        print("\n" + "═" * 60)
        print(f" ✅ PIPELINE SUCCESS. Total Time: {time.time() - start_total:.1f}s")
        print(f" 📂 Prediction Artifact: {out_path}")
        print("═" * 60)
    
    return results

if __name__ == "__main__":
    run_pax_pipeline()
