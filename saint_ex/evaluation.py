import pandas as pd
import numpy as np
import os
from google.cloud import bigquery
from google.oauth2 import service_account
from saint_ex.config import BQ_PROJECT, BQ_CREDS, BQ_DATASET, BQ_TABLE

def get_actuals_from_bq(start_date: str, end_date: str) -> pd.DataFrame:
    """ Retrieves ground truth labels from BigQuery for a specific window. """
    try:
        creds = service_account.Credentials.from_service_account_file(BQ_CREDS)
        client = bigquery.Client(credentials=creds, project=BQ_PROJECT)
        
        sql = f"""
        SELECT 
            IdADL, 
            LTScheduledDatetime,
            NbPaxTotal as Actual_NbPaxTotal,
            FarmsNbPaxPHMR as Actual_PRM
        FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE DATE(LTScheduledDatetime) BETWEEN '{start_date}' AND '{end_date}'
        """
        df = client.query(sql).to_dataframe()
        
        if df.empty:
            return df

        # Aggressive Deduplicate: Take the best record for each UNIQUE Flight ID (IdADL)
        df = df.dropna(subset=['IdADL'])
        df = df.groupby('IdADL').agg({
            'LTScheduledDatetime': 'first',
            'Actual_NbPaxTotal': 'max',
            'Actual_PRM': 'max'
        }).reset_index()
        
        return df
    except Exception as e:
        print(f"📡 Evaluation Error (BQ): {e}")
        return pd.DataFrame()


def calculate_metrics(preds: pd.DataFrame, actuals: pd.DataFrame) -> dict:
    """ Calculates key performance indicators by merging predictions with ground truth. """
    if preds.empty or actuals.empty:
        return {}

    # Standardized merge on IdADL and LTScheduledDatetime to ensure alignment
    merged = pd.merge(preds, actuals, on=['IdADL', 'LTScheduledDatetime'], how='inner')
    
    # Filter for realized flights (Those in the past, including freight/test with 0 pax)
    # We use a 2-hour buffer to account for BigQuery processing delays
    now_buffer = pd.Timestamp.now() - pd.Timedelta(hours=2)
    merged['LTScheduledDatetime'] = pd.to_datetime(merged['LTScheduledDatetime']).dt.tz_localize(None)
    merged = merged[merged['LTScheduledDatetime'] <= now_buffer].copy()
    
    if merged.empty:
        return {"flights_matched": 0}

    # Total Pax Metrics
    merged['Pax_Error'] = (merged['Pred_NbPaxTotal'] - merged['Actual_NbPaxTotal']).abs()
    mae = merged['Pax_Error'].mean()
    med_ae = merged['Pax_Error'].median()
    
    total_actual = merged['Actual_NbPaxTotal'].sum()
    total_pred = merged['Pred_NbPaxTotal'].sum()
    global_acc = 1 - (abs(total_pred - total_actual) / total_actual) if total_actual > 0 else 0

    # PRM Metrics
    merged['PRM_Error'] = (merged['Pred_FarmsNbPaxPHMR'] - merged['Actual_PRM']).abs()
    prm_mae = merged['PRM_Error'].mean()

    return {
        "flights_matched": len(merged),
        "total_flights": len(preds),
        "mae": mae,
        "median_ae": med_ae,
        "global_accuracy": global_acc,
        "total_actual": total_actual,
        "total_pred": total_pred,
        "prm_mae": prm_mae
    }

def print_validation_report(metrics: dict, title: str = "Performance Audit"):
    """ Prints a formatted summary of the evaluation results. """
    if not metrics or metrics.get('flights_matched', 0) == 0:
        print(f"⚠️ {title}: No flights matched for evaluation.")
        return

    print(f"════════════════════════════════════════════════════════════")
    print(f" 📊 {title}")
    print(f"════════════════════════════════════════════════════════════")
    print(f"  Matched Flights : {metrics['flights_matched']} / {metrics['total_flights']}")
    print(f"  Mean Abs Error  : {metrics['mae']:.2f} passengers")
    print(f"  Median Abs Error: {metrics['median_ae']:.2f} passengers")
    print(f"  Grand Variance  : {metrics['total_pred'] - metrics['total_actual']:,.0f} passengers")
    print(f"  Global Accuracy : {metrics['global_accuracy']*100:.2f}%")
    print(f"  PRM MAE         : {metrics['prm_mae']:.2f}")
    print(f"════════════════════════════════════════════════════════════")
