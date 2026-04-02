# =========================================
# LIGHTGBM TRAINING SCRIPT (v6 semantic columns)
# =========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================
# 1. LOAD DATA
# =========================================
PATH = "/content/drive/MyDrive/final_dataset_preprocessed_v6_semantic.csv"
df = pd.read_csv(PATH, parse_dates=["UTCExternalDate"])
df = df.sort_values("UTCExternalDate").reset_index(drop=True)
print(" Loaded:", df.shape)

# =========================================
# 2. FEATURES
# =========================================
TARGET = "NbPaxTotal"
DROP_COLS = [TARGET, "UTCExternalDate"]
FEATURES = [c for c in df.columns if c not in DROP_COLS]

X = df[FEATURES].copy()  # <-- copy to avoid SettingWithCopyWarning
y = df[TARGET]

# =========================================
# 3. CATEGORICAL HANDLING
# =========================================
cat_cols = ["AirportOrigin", "airlineOACICode", "Direction", "ScheduleType", "route"]
for col in cat_cols:
    X[col] = X[col].astype("category")

# =========================================
# 4. TIME SPLIT
# =========================================
n = len(df)
train_end = int(n * 0.8)

X_train, X_test = X.iloc[:train_end], X.iloc[train_end:]
y_train, y_test = y.iloc[:train_end], y.iloc[train_end:]

# =========================================
# 5. MODEL
# =========================================
params = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": 8,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1
}

model = lgb.LGBMRegressor(**params, n_estimators=1000)

# =========================================
# 6. TRAIN
# =========================================
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="mae",
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

# =========================================
# 7. EVALUATION
# =========================================
y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n RESULTS")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# =========================================
# 8. SAVE MODEL
# =========================================
joblib.dump(model, "/content/drive/MyDrive/lgbm_with_lags_v6_semantic.pkl")
print("💾 Model saved")
