
import argparse
import pandas as pd
from run_pipeline import run_pax_pipeline
from saint_ex.evaluation import get_actuals_from_bq, calculate_metrics, print_validation_report

def main():
    parser = argparse.ArgumentParser(description="Saint-Exupéry Dynamic Backtesting Suite")
    parser.add_argument("--start", type=str, required=True, help="Inference Start date (everything before is Train/Val)")
    parser.add_argument("--end", type=str, default=None, help="Inference End date for metrics calculation")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon in hours (leave empty for full pool)")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print(f" 🧪 DYNAMIC BACKTEST: {args.start}")
    print("═" * 60)

    # 1. Run Pipeline with high-accuracy temporal isolation
    # Training will focus only on data PRIOR to args.start.
    # Inference will predict the entire available future pool relative to args.start.
    preds = run_pax_pipeline(
        inference_start_date=args.start, 
        forecast_horizon_hours=args.horizon,
        silent=True
    )

    if preds.empty:
        print("❌ Error: No predictions generated for the given period.")
        return

    # 2. Slice evaluation window if requested (e.g. Test only on February)
    if args.end:
        print(f"✂️ Truncating test window to end at {args.end}...")
        preds['LTScheduledDatetime'] = pd.to_datetime(preds['LTScheduledDatetime']).dt.tz_localize(None)
        preds = preds[preds['LTScheduledDatetime'] < pd.to_datetime(args.end)].copy()
        
    if preds.empty:
        print("❌ Error: No predictions left after applying the --end cutoff.")
        return

    # 3. Automatically determine the validation window
    # We use the min/max from the generated predictions
    actual_start = preds['LTScheduledDatetime'].min().strftime('%Y-%m-%d')
    actual_end = preds['LTScheduledDatetime'].max().strftime('%Y-%m-%d')

    print(f"📡 Syncing Ground Truth Labels: {actual_start} to {actual_end}...")
    
    # 3. Fetch Actuals from BigQuery (Includes ALL flight rows)
    actuals = get_actuals_from_bq(actual_start, actual_end)
    
    # 4. Calculate Final Multi-Month Metrics
    metrics = calculate_metrics(preds, actuals)
    
    # 5. Output Audit Report
    print_validation_report(metrics, title=f"Dynamic Audit: {args.start} → {actual_end}")
    
    print("\n✅ Dynamic Backtest Completed.")

if __name__ == "__main__":
    main()
