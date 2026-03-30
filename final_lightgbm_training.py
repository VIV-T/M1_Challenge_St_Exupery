# ===============================================
# LightGBM Final Evaluation on Correct Datasets
# ===============================================

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================
# Config
# ================================
TARGET = "NbPaxTotal"

# ================================
# Metrics function
# ================================
def compute_operational_metrics(y_true, y_pred, X_subset=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Weighted MAE using NbOfSeats if present
    if X_subset is not None and 'NbOfSeats' in X_subset.columns:
        weights = X_subset['NbOfSeats'].fillna(1)
        weighted_mae = np.average(np.abs(y_true - y_pred), weights=weights)
    else:
        weighted_mae = mae

    # Low-volume accuracy (<10 passengers)
    low_mask = y_true < 10
    low_volume_accuracy = np.mean(np.abs(y_true[low_mask] - y_pred[low_mask]) <= 1) if low_mask.any() else np.nan

    # Peak capture ratio (top 5% of target)
    threshold = np.percentile(y_true, 95)
    actual_peaks = y_true >= threshold
    predicted_peaks = y_pred >= threshold
    peak_capture_ratio = np.mean(predicted_peaks[actual_peaks]) if actual_peaks.any() else np.nan

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Weighted MAE': weighted_mae,
        'Low-volume accuracy': low_volume_accuracy,
        'Peak capture ratio': peak_capture_ratio
    }

# ================================
# Datasets and respective models
# ================================
datasets_and_models = {
    "FULL dataset": {
        "file": "/content/drive/MyDrive/final_clean_dataset_for_LightGBM.csv",
        "model": "/content/drive/MyDrive/lightgbm_model_20260328_135035.pkl"
    },
    "MIN-NaN dataset": {
        "file": "/content/drive/MyDrive/final_clean_dataset_for_LightGBM_min_NaN.csv",
        "model": "/content/drive/MyDrive/lightgbm_model_20260328_152929.pkl"
    }
}

# ================================
# Evaluation loop
# ================================
for name, paths in datasets_and_models.items():
    print(f"\n=== Evaluating {name} ===")

    # Load dataset
    df = pd.read_csv(paths["file"], parse_dates=['LTScheduledDatetime'])
    df = df.sort_values('LTScheduledDatetime').reset_index(drop=True)
    df = df.dropna(subset=[TARGET])

    # Select numeric features only, excluding ID columns and the target
    FEATURES = [c for c in df.columns if c not in ['LTScheduledDatetime', TARGET] and not c.startswith('Id')]
    X = df[FEATURES].select_dtypes(include=[np.number]).fillna(0)
    y = df[TARGET]

    # Temporal split 80/10/10 (test = last 10%)
    n = len(df)
    val_end = int(n * 0.9)
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    # Load model
    bundle = joblib.load(paths["model"])
    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle
    best_iter = bundle.get("best_iteration", None) if isinstance(bundle, dict) else None

    # Predict on test
    if best_iter:
        y_pred_test = model.predict(X_test, num_iteration=best_iter)
    else:
        y_pred_test = model.predict(X_test)

    # Compute metrics
    metrics = compute_operational_metrics(y_test.values, y_pred_test, X_test)
    print("Test Metrics:", metrics)
