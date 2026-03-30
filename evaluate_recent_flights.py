#!/usr/bin/env python3
"""
evaluate_recent_flights.py — External Validation Against Ground Truth.

This script compares local model predictions (March 2026) against the 
latest realized outcomes in BigQuery. It uses strict filtering to 
ensure we only compare against legitimate commercial passenger flights.
"""
import os
import pandas as pd
from google.cloud import bigquery
from sklearn.metrics import mean_absolute_error

# Configuration
PROJECT_ID = "va-sdh-adl-staging"
DATASET_ID = "aero_insa"
TABLE_ID = "mouvements_aero_insa"
CREDENTIALS_PATH = "insa/va-sdh-adl-staging.json"
PREDICTIONS_PATH = "outputs_new/predictions_flight.csv"

def main():
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"❌ Error: Missing predictions at {PREDICTIONS_PATH}")
        return

    # 1. Load Local Predictions
    print("Loading March predictions...")
    preds = pd.read_csv(PREDICTIONS_PATH, parse_dates=['LTScheduledDatetime'])
    
    # Force full March evaluation window
    start_date = '2026-03-01'
    print(f"  Evaluating performance for: {start_date} → 2026-03-31")

    # 2. Query BigQuery for Realized Flights (Ground Truth)
    print("\nQuerying BigQuery for actual outcomes...")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH
    client = bigquery.Client(project=PROJECT_ID)
    
    table_ref = f"`{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"
    
    # We select exactly the same columns used in the training split.
    query = f"""
        SELECT 
            IdMovement as ActualId,
            LTScheduledDatetime,
            airlineOACICode,
            SysStopover,
            Direction,
            NbPaxTotal,
            NbOfSeats,
            IdBusinessUnitType
        FROM {table_ref}
        WHERE IdTraficType = 1 
          AND IdBusinessUnitType = 1
          AND flight_with_pax = 'Oui'
          AND NbPaxTotal IS NOT NULL
          AND NbOfSeats > 0
          AND LTScheduledDatetime >= '{start_date}'
    """
    
    try:
        actuals = client.query(query).to_dataframe()
        actuals['LTScheduledDatetime'] = pd.to_datetime(actuals['LTScheduledDatetime'])
    except Exception as e:
        print(f"❌ BigQuery Error: {e}")
        return

    print(f"  Found {len(actuals):,} realized flights in BQ since {start_date}.")
    
    # 3. Match Predictions vs Reality
    print("\nMatching predictions vs actuals...")
    match_cols = ['LTScheduledDatetime', 'airlineOACICode', 'SysStopover', 'Direction']
    merged = preds.merge(actuals, on=match_cols, how='inner')
    
    # Prune technical/ferry flights (Large planes with <10 pax)
    commercial_mask = ~((merged['NbOfSeats'] >= 50) & (merged['NbPaxTotal'] < 10))
    n_anomalies = (~commercial_mask).sum()
    merged = merged[commercial_mask].copy()
    
    print(f"  Matched {len(merged):,} commercial flights ({n_anomalies} technical anomalies removed).")

    if merged.empty:
        print("  ⚠️ No matching realized flights found in BQ yet.")
        return

    # 4. Evaluation Metrics
    mae = mean_absolute_error(merged['NbPaxTotal'], merged['predicted_pax'])
    avg_pax = merged['NbPaxTotal'].mean()
    accuracy = max(0, 100 - (mae / avg_pax * 100)) if avg_pax > 0 else 0

    print("\n" + "="*50)
    print("  EVALUATION RESULTS (MARCH 2026)")
    print("="*50)
    print(f"  Commercial Flights : {len(merged)}")
    print(f"  Flight MAE         : {mae:.2f} passengers")
    print(f"  Accuracy           : {accuracy:.1f} %")
    print("="*50)
    
    # Display the "challenging" predictions (Worst errors)
    print("\nWorst Sample (Absolute Error Descending):")
    sample = merged[['LTScheduledDatetime', 'airlineOACICode', 'SysStopover', 
                     'predicted_pax', 'NbPaxTotal']]
    sample['Error'] = (sample['predicted_pax'] - sample['NbPaxTotal']).round(1)
    
    # Show worst predictions first
    sample = sample.sort_values(by='Error', key=abs, ascending=False).head(10)
    print(sample.to_string(index=False))

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    main()
