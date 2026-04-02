import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime, timedelta
import lightgbm as lgb
import joblib
import logging
import sys
from typing import cast, Tuple
import numpy.typing as npt

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from scripts.data_preparation.get_main import main_query_db
from scripts.data_preparation.preprocessed import main_preprocessed
from scripts.data_preparation.utils.progress_bar import TqdmCallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

import argparse

# ------------- Global Variables -------------
DATA_FOLDER_PATH = os.path.join(root_path, "data")
MODEL_FOLDER_PATH = os.path.join(root_path, "models")
OUTPUT_FOLDER_PATH = os.path.join(root_path, "output")

DATASET_PATH_PAX = os.path.join(DATA_FOLDER_PATH, "main_preprocessed.csv")
DATASET_PATH_PHMR = os.path.join(DATA_FOLDER_PATH, "main_preprocessed_PHMR.csv")
RAW_DATA_PATH = os.path.join(DATA_FOLDER_PATH, "main.csv")

MODEL_FILENAME = os.path.join(MODEL_FOLDER_PATH, "lgbm_regressor.pkl")
MODEL_FILENAME_PHMR = os.path.join(MODEL_FOLDER_PATH, "lgbm_regressor_PHMR.pkl")

TARGET = ["NbPaxTotal", "FarmsNbPaxPHMR"]

def load_or_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed data or run the ingestion+preprocessing pipeline."""
    if os.path.exists(DATASET_PATH_PAX) and os.path.exists(DATASET_PATH_PHMR):
        logger.info("Loading existing preprocessed files...")
        df = pd.read_csv(DATASET_PATH_PAX, encoding='utf-8', low_memory=False)
        df_PHMR = pd.read_csv(DATASET_PATH_PHMR, encoding='utf-8', low_memory=False)
    else:
        logger.info("Preprocessed files not found. Starting ingestion and preprocessing...")
        if not os.path.exists(RAW_DATA_PATH):
            logger.info("Raw data missing. Querying BigQuery...")
            main_query_db()
        
        logger.info("Preprocessing data...")
        df, df_PHMR = main_preprocessed(with_holidays=False)
    
    return df, df_PHMR

def prepare_sets(df: pd.DataFrame, target: str, limit_train: pd.Timestamp, limit_valid: pd.Timestamp, limit_test: pd.Timestamp):
    """Splits data into train, valid, and test sets and handles categorical encoding."""
    df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime'])
    
    # Identify sets
    train_df = df[df['LTScheduledDatetime'] < limit_train].copy()
    valid_df = df[(df['LTScheduledDatetime'] >= limit_train) & (df['LTScheduledDatetime'] < limit_valid)].copy()
    test_df = df[(df['LTScheduledDatetime'] >= limit_valid) & (df['LTScheduledDatetime'] < limit_test)].copy()
    
    if test_df.empty:
        logger.warning(f"No flights found for target {target} in the test range {limit_valid} to {limit_test}")

    # Log set information
    logger.info(f"Dataset for {target}:")
    logger.info(f"Train set: {len(train_df)} rows / From {train_df['LTScheduledDatetime'].min()} to {train_df['LTScheduledDatetime'].max()}")
    logger.info(f"Valid set: {len(valid_df)} rows / From {valid_df['LTScheduledDatetime'].min()} to {valid_df['LTScheduledDatetime'].max()}")
    logger.info(f"Test set: {len(test_df)} rows / From {test_df['LTScheduledDatetime'].min()} to {test_df['LTScheduledDatetime'].max()}")
    
    # Store flight info for test set
    test_info = test_df[['FlightNumberNormalized', 'LTScheduledDatetime']].copy()
    
    # Convert dates to string for model
    for slice_df in [train_df, valid_df, test_df]:
        slice_df['LTScheduledDatetime'] = slice_df['LTScheduledDatetime'].astype(str)
        
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_valid = valid_df.drop(columns=[target])
    y_valid = valid_df[target]
    X_test = test_df.drop(columns=[target])
    
    # Handle categorical columns
    cat_cols = X_train.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X_train[col] = X_train[col].astype('category')
        X_valid[col] = X_valid[col].astype('category')
        X_test[col] = X_test[col].astype('category')
            
    return X_train, y_train, X_valid, y_valid, X_test, test_info

def get_model(model_path: str, X_train, y_train, X_valid, y_valid, force_train: bool = False):
    """Load or train a LightGBM model."""
    if os.path.exists(model_path) and not force_train:
        logger.info(f"Loading existing model from {model_path}")
        return joblib.load(model_path)
    
    if force_train:
        logger.info(f"Force training enabled. Training new model for {model_path}")
    else:
        logger.info(f"Training new model for {model_path}")

    n_estimators = 10000
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.01,
        num_leaves=255,
        min_child_samples=5,
        feature_fraction=0.8,
        random_state=42,
        verbose=-1
    )
    
    tqdm_callback = TqdmCallback(total=n_estimators)
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(100),
                tqdm_callback
            ]
        )
    finally:
        tqdm_callback.pbar.close()
        
    joblib.dump(model, model_path)
    return model

def run_pipeline(predict_days: int = 2, val_days: int = 7, force_train: bool = False):
    """Main execution function."""
    os.makedirs(MODEL_FOLDER_PATH, exist_ok=True)
    
    # Calculate Dynamic Dates
    now = datetime.now()
    today = pd.Timestamp(now.date())
    limit_valid = today
    limit_test = today + timedelta(days=predict_days)
    limit_train = today - timedelta(days=val_days)

    logger.info(f"Pipeline Config: Predict {predict_days} days / Val {val_days} days / Force Train: {force_train}")
    
    # 1. Data Preparation
    df, df_PHMR = load_or_prepare_data()
    
    # 2. Features and Sets Preparation
    logger.info("Preparing features for PAX and PHMR...")
    X_train, y_train, X_valid, y_valid, X_test, test_info_pax = prepare_sets(df, TARGET[0], limit_train, limit_valid, limit_test)
    X_train_PHMR, y_train_PHMR, X_valid_PHMR, y_valid_PHMR, X_test_PHMR, test_info_phmr = prepare_sets(df_PHMR, TARGET[1], limit_train, limit_valid, limit_test)
    
    # 3. Model Training/Loading
    model_Pax = get_model(MODEL_FILENAME, X_train, y_train, X_valid, y_valid, force_train=force_train)
    model_PHMR = get_model(MODEL_FILENAME_PHMR, X_train_PHMR, y_train_PHMR, X_valid_PHMR, y_valid_PHMR, force_train=force_train)
    
    # 4. Predictions
    logger.info("Generating predictions for the test set...")
    predictions_nb_pax = np.maximum(0.0, np.round(model_Pax.predict(X_test)))
    predictions_phmr_pax = np.maximum(0.0, np.round(model_PHMR.predict(X_test_PHMR)))
    
    df_pax_preds = test_info_pax.copy()
    df_pax_preds['PredNbPaxTotal'] = predictions_nb_pax.astype(int)
    
    df_phmr_preds = test_info_phmr.copy()
    df_phmr_preds['PredFarmsNbPaxPHMR'] = predictions_phmr_pax.astype(int)
    
    df_final = pd.merge(df_pax_preds, df_phmr_preds, on=['FlightNumberNormalized', 'LTScheduledDatetime'], how='inner')
    logger.info(f"Generated {len(df_final)} merged predictions")

    # 5. Hourly Aggregation
    logger.info("Aggregating predictions by hour...")
    hourly_preds = df_final.copy()
    hourly_preds['LTScheduledDatetime'] = pd.to_datetime(hourly_preds['LTScheduledDatetime'])
    hourly_agg = hourly_preds.resample('h', on='LTScheduledDatetime')[['PredNbPaxTotal', 'PredFarmsNbPaxPHMR']].sum().reset_index()
    hourly_agg = hourly_agg.sort_values('LTScheduledDatetime')
    hourly_agg[['PredNbPaxTotal', 'PredFarmsNbPaxPHMR']] = hourly_agg[['PredNbPaxTotal', 'PredFarmsNbPaxPHMR']].astype(int)
    
    # 6. Output
    current_date_str = today.strftime('%Y-%m-%d')
    output_filename = os.path.join(DATA_FOLDER_PATH, f"hourly_predictions_{current_date_str}.csv")
    hourly_agg.to_csv(output_filename, index=False)
    logger.info(f"Hourly predictions saved to {output_filename}")
    
    flight_output_filename = os.path.join(OUTPUT_FOLDER_PATH, f"flight_predictions_{current_date_str}.csv")
    df_final = df_final.sort_values('LTScheduledDatetime')
    df_final.to_csv(flight_output_filename, index=False)
    logger.info(f"Flight-level predictions saved to {flight_output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Airport Passenger Prediction Pipeline")
    parser.add_argument("--predict-days", type=int, default=2, help="Number of days to predict from today")
    parser.add_argument("--val-days", type=int, default=7, help="Number of days for validation context")
    parser.add_argument("--force-train", action="store_true", help="Force model retraining")
    
    args = parser.parse_args()
    
    run_pipeline(
        predict_days=args.predict_days, 
        val_days=args.val_days, 
        force_train=args.force_train
    )
