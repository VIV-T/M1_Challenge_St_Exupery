"""
models.py — Production Gradient Boosting Architecture for Project Saint-Exupéry.

This module implements the core LightGBM-based passenger flow predictor,
incorporating bidirectional feature weighting and PRM capacity adjustments.
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

class PaxModel:
    """Specialized LightGBM regressor for total passenger flow prediction."""
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
            verbosity=-1
        )
        self.features = None

    def train(self, X, y, X_val=None, y_val=None):
        """Train the model with Early Stopping (if validation set provided)."""
        self.features = X.columns.tolist()
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        self.model.fit(
            X, y,
            eval_set=eval_set,
            eval_metric='mae',
            callbacks=[lgb.early_stopping(stopping_rounds=50)] if eval_set else []
        )
        
        if eval_set:
            y_pred = self.model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            print(f"  Pax LightGBM Validation MAE: {mae:.2f}")

    def predict(self, X):
        """Generates passenger volume predictions for the inference pool."""
        return np.maximum(0, self.model.predict(X[self.features]))

class PRMModel:
    """Regressor for Passenger with Reduced Mobility (PRM) flows."""
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=15,
            random_state=42,
            verbosity=-1
        )
        self.features = None

    def train(self, X, y, X_val=None, y_val=None):
        self.features = X.columns.tolist()
        self.model.fit(X, y)
        if X_val is not None:
            y_pred = self.model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            print(f"  PRM LightGBM Validation MAE: {mae:.2f}")

    def predict(self, X):
        return np.maximum(0, self.model.predict(X[self.features]))
