import os
import pandas as pd
import numpy as np
import time
from google.cloud import bigquery
from sklearn.metrics import mean_absolute_error
from saint_ex import viz
from saint_ex.config import (
    BQ_PROJECT, BQ_DATASET, BQ_TABLE, BQ_CREDS, 
    INFERENCE_START_DATE, OUTPUT_DIR
)

def evaluate_predictions(preds_df):
    """
    Validates predictions against live BigQuery ground truth.
    Generates metrics and professional visualizations.
    """
    print("\n" + "─"*60)
    print(" [VALIDATION] Syncing with BigQuery Ground Truth...")
    print("─"*60)
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = BQ_CREDS
    client = bigquery.Client(project=BQ_PROJECT)
    
    start_date = INFERENCE_START_DATE
    table_ref = f"`{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`"
    
    query = f"""
        SELECT 
            IdMovement, LTScheduledDatetime, airlineOACICode, 
            SysStopover, Direction, NbPaxTotal, NbOfSeats, IdBusinessUnitType, flight_with_pax
        FROM {table_ref}
        WHERE LTScheduledDatetime >= '{start_date}'
          AND NbPaxTotal IS NOT NULL
    """
    
    try:
        actuals = client.query(query).to_dataframe()
        actuals['LTScheduledDatetime'] = pd.to_datetime(actuals['LTScheduledDatetime']).dt.tz_localize(None).astype('datetime64[ns]')
    except Exception as e:
        print(f"  ❌ BigQuery Connection Error: {e}")
        return
    
    # 1. Match Predictions vs Reality
    anchor_cols = ['LTScheduledDatetime', 'airlineOACICode', 'SysStopover', 'Direction']
    merged = preds_df.merge(actuals, on=anchor_cols, how='inner', suffixes=('', '_actual'))
    
    # 2. Strict Commercial Filtering (Chapter IX Compliance)
    commercial_mask = (
        (merged['IdBusinessUnitType'] == 1) & 
        (merged['NbOfSeats'] > 0) & 
        (merged['flight_with_pax'].fillna('Non') == 'Oui') &
        (merged['NbPaxTotal'] > 0)
    )
    merged = merged[commercial_mask].copy()
    
    if merged.empty:
        print("  ⚠️ No matching realized commercial flights found in BQ yet.")
        return

    # 3. Calculate Metrics
    flight_mae = (merged['predicted_pax'] - merged['NbPaxTotal']).abs().mean()
    flight_acc = 100 * max(0, 1 - (flight_mae / merged['NbPaxTotal'].mean()))
    global_reliability = 100 * (merged['predicted_pax'].sum() / merged['NbPaxTotal'].sum())

    # Hourly Aggregation (Operational Resolution)
    merged['hour_key'] = merged['LTScheduledDatetime'].dt.strftime('%Y-%m-%d %H:00')
    hourly_slots = merged.groupby('hour_key')[['predicted_pax', 'NbPaxTotal']].sum().reset_index()
    hourly_mae = (hourly_slots['predicted_pax'] - hourly_slots['NbPaxTotal']).abs().mean()
    hourly_acc = 100 * max(0, 1 - (hourly_mae / hourly_slots['NbPaxTotal'].mean()))

    # Daily Aggregation
    merged['day_key'] = merged['LTScheduledDatetime'].dt.date
    daily = merged.groupby('day_key')[['predicted_pax', 'NbPaxTotal']].sum().reset_index()
    daily_mae = (daily['predicted_pax'] - daily['NbPaxTotal']).abs().mean()
    daily_acc = 100 * max(0, 1 - (daily_mae / daily['NbPaxTotal'].mean()))

    print("\n" + "="*50)
    print(f"  FINAL VALIDATION RESULTS ({start_date} → Now)")
    print("="*50)
    print(f"  Commercial Flights   : {len(merged):,}")
    print(f"  Flight MAE           : {flight_mae:.2f} passengers")
    print(f"  Flight Accuracy      : {flight_acc:.1f} %")
    print(f"  Hourly Total MAE     : {hourly_mae:.2f} passengers/hour")
    print(f"  Hourly Total Accuracy: {hourly_acc:.1f} %")
    print(f"  Daily Total MAE      : {daily_mae:.2f} passengers/day")
    print(f"  Daily Total Accuracy : {daily_acc:.1f} %")
    print(f"  Global Reliability   : {global_reliability:.1f} % (Sum Ratio)")
    print("="*50)

    # 4. Auto-Generate Visualizations
    print("\n[REPORTING] Exporting Submission Assets...")
    viz.plot_daily_momentum(daily)
    viz.plot_error_distribution(merged)
    
    # Intra-Day Distribution (Avg per Hour)
    merged['hour'] = merged['LTScheduledDatetime'].dt.hour
    hourly = merged.groupby('hour')[['predicted_pax', 'NbPaxTotal']].sum().reset_index()
    viz.plot_hourly_distribution(hourly)
    
    # Weekly Signature (Mean Daily Total per Day of Week)
    merged['dayofweek'] = merged['LTScheduledDatetime'].dt.dayofweek
    daily_totals = merged.groupby(['day_key', 'dayofweek'])[['predicted_pax', 'NbPaxTotal']].sum().reset_index()
    weekly_sig = daily_totals.groupby('dayofweek')[['predicted_pax', 'NbPaxTotal']].mean().reset_index()
    viz.plot_weekly_signature(weekly_sig)
    
    importance_path = os.path.join(OUTPUT_DIR, "importance.csv")
    if os.path.exists(importance_path):
        importance_df = pd.read_csv(importance_path)
        viz.plot_feature_importance(importance_df)
    
    print(f" ✅ Portfolio High-Res Assets exported to exports/plots/")

def run_historical_backtest(df):
    """
    Executes the multi-year stability audit across 2024, 2025, and 2026.
    """
    from saint_ex.features import add_features, prepare_X
    from saint_ex.models import PaxModel
    
    windows = [
        {"name": "Hijri Surge 2024", "start": "2024-04-10", "end": "2024-04-20"},
        {"name": "Hijri Surge 2025", "start": "2025-03-31", "end": "2025-04-10"},
        {"name": "Summer Peak 2025", "start": "2025-07-01", "end": "2025-07-15"},
    ]
    
    results = []
    print("\n" + "═"*80)
    print(" [AUDIT] Running Chronological Multi-Year Backtest")
    print("═"*80)
    
    for window in windows:
        print(f"Evaluating Window: {window['name']}...")
        start_dt = pd.to_datetime(window['start'])
        end_dt   = pd.to_datetime(window['end'])
        
        hist = df[df['LTScheduledDatetime'] < start_dt].copy()
        test = df[(df['LTScheduledDatetime'] >= start_dt) & (df['LTScheduledDatetime'] <= end_dt)].copy()
        
        # Simple evaluation loop
        h_enriched = add_features(hist)
        t_enriched = add_features(test)
        
        X_h, cols = prepare_X(h_enriched)
        X_t, _    = prepare_X(t_enriched, cols)
        
        model = PaxModel()
        model.train(X_h, h_enriched['NbPaxTotal'])
        
        test['predicted_pax'] = model.predict(X_t)
        
        # Metrics
        f_mae = (test['predicted_pax'] - test['NbPaxTotal']).abs().mean()
        
        test['day_key'] = test['LTScheduledDatetime'].dt.date
        daily = test.groupby('day_key')[['predicted_pax', 'NbPaxTotal']].sum()
        d_acc = 100 * max(0, 1 - ((daily['predicted_pax'] - daily['NbPaxTotal']).abs().mean() / daily['NbPaxTotal'].mean()))
        
        results.append({
            "Window": window['name'],
            "Flights": len(test),
            "MAE": f_mae,
            "Daily Accuracy": d_acc
        })
    
    print("\n" + "="*80)
    print(" CROSS-WINDOW STABILITY REPORT")
    print("="*80)
    report = pd.DataFrame(results)
    print(report.to_string(index=False))
    print("="*80)
