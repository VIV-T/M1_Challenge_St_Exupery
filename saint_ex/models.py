"""
Machine learning models module for Project Saint-Exupéry Airport Passenger Flow Prediction.

This module implements the core LightGBM-based predictive models:
- PaxModel: Total passenger flow prediction using occupancy-weighted approach
- PRMModel: Passengers with Reduced Mobility prediction using Tweedie loss

Key Features:
- Occupancy factor transformation for stable training
- Early stopping with validation monitoring
- Feature importance tracking
- Appropriate loss functions for different target distributions
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.metrics import mean_absolute_error
from saint_ex.config import LGB_PAX_PARAMS, LGB_PRM_PARAMS, OCCUPANCY_CLIP, EARLY_STOPPING_ROUNDS

class PaxModel:
    """
    LightGBM regressor for total passenger flow prediction.
    
    This model implements an occupancy-weighted approach where the target
    is transformed to a passenger/seat ratio before training. This stabilizes
    training across different aircraft sizes and improves generalization.
    
    Attributes:
        model: Trained LightGBM regressor
        features: List of feature names used for training
    """
    
    def __init__(self) -> None:
        """Initialize the Pax model with default hyperparameters."""
        self.model = lgb.LGBMRegressor(**LGB_PAX_PARAMS)
        self.features: Optional[List[str]] = None

    def train(self, X: pd.DataFrame, y: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> None:
        """
        Train the model to predict occupancy factor (passengers/seat ratio).
        
        The target is transformed to occupancy ratio to normalize across
        different aircraft capacities. This makes the model learn route
        popularity independently of aircraft size.
        
        Args:
            X: Training features
            y: Training target (actual passenger counts)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        self.features = X.columns.tolist()
        
        # Transform target to occupancy ratio for stable training
        seats_train = X['NbOfSeats'].clip(lower=1)
        y_transformed = (y / seats_train).clip(0, 1.2)
        
        # Prepare validation set if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            seats_val = X_val['NbOfSeats'].clip(lower=1)
            y_val_transformed = (y_val / seats_val).clip(0, 1.2)
            eval_set = [(X_val, y_val_transformed)]
        
        # Train model with early stopping
        self.model.fit(
            X, y_transformed,
            eval_set=eval_set,
            eval_metric='mae',
            callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS)] if eval_set else []
        )
        
        # Calculate and report validation metrics
        if eval_set:
            occ_pred = self.model.predict(X_val)
            y_pred = occ_pred * X_val['NbOfSeats']
            mae = mean_absolute_error(y_val, y_pred)
            print(f"  Pax (Occupancy-Weighted) Validation MAE: {mae:.2f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate passenger count predictions.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Array of predicted passenger counts (rounded to integers)
        """
        # Predict occupancy ratio and apply clipping
        occ_pred = self.model.predict(X[self.features]).clip(0, OCCUPANCY_CLIP)
        # Convert back to passenger counts
        return np.maximum(0, occ_pred * X['NbOfSeats']).round().astype(int)

class PRMModel:
    """
    LightGBM regressor for Passengers with Reduced Mobility (PRM) prediction.
    
    PRM data has special characteristics:
    - Low volume (few PRMs per flight)
    - Many zeros (flights with no PRMs)
    - High operational importance
    
    This model uses Tweedie loss which is well-suited for count data
    with many zeros and positive values.
    
    Attributes:
        model: Trained LightGBM regressor with Tweedie loss
        features: List of feature names used for training
    """
    
    def __init__(self) -> None:
        """Initialize the PRM model with Tweedie loss hyperparameters."""
        self.model = lgb.LGBMRegressor(**LGB_PRM_PARAMS)
        self.features: Optional[List[str]] = None

    def train(self, X: pd.DataFrame, y: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> None:
        """
        Train the PRM prediction model.
        
        Uses Tweedie loss which is appropriate for count data with
        many zeros, making it ideal for PRM prediction.
        
        Args:
            X: Training features
            y: Training target (PRM counts)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        self.features = X.columns.tolist()
        
        # Prepare validation set for early stopping
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        # Train model with early stopping
        self.model.fit(
            X, y,
            eval_set=eval_set,
            eval_metric='mae',
            callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS)] if eval_set else []
        )
        
        # Report validation performance
        if X_val is not None:
            y_pred = self.model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            print(f"  PRM LightGBM Validation MAE: {mae:.2f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate PRM count predictions.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Array of predicted PRM counts (non-negative)
        """
        return np.maximum(0, self.model.predict(X[self.features]))
