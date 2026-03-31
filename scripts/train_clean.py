import pandas as pd
import numpy as np
from datetime import date
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chargement et préparation
df = pd.read_csv("data/main_preprocessed_new.csv")
df["LTScheduledDatetime"] = pd.to_datetime(df["LTScheduledDatetime"])

TARGET = "NbPaxTotal"  

print("1. Load and prepare data...")
limit_date_valid = pd.Timestamp("2026-02-01")
limit_date = pd.Timestamp("2026-03-01")
yesterday = pd.Timestamp(date.today() - pd.Timedelta(days=1))

#df = df[df['IdBusinessUnitType'] == 1]
df = df.drop(columns=['LTExternalDate', 'Terminal'])  # Suppression de la colonne Terminal et de LTExternalDateTime
df = df.dropna(subset=['FlightNumberNormalized'])  # Supprimer les lignes où la cible est manquante



# Target Encoding pour catégories non encore encodées
print("  - Target Encoding for categories...")
train_df_temp = df[df['LTScheduledDatetime'] < limit_date_valid].copy()

for col in ['airlineOACICode', 'ScheduleType', 'FuelProvider']:
    if col in df.columns:
        target_encoding = train_df_temp.groupby(col)[TARGET].mean()
        df[f'{col}_TargetEnc'] = df[col].map(target_encoding).fillna(df[TARGET].mean())

train_df = df[df['LTScheduledDatetime'] < limit_date_valid].copy()
valid_df = df[(df['LTScheduledDatetime'] >= limit_date_valid) & (df['LTScheduledDatetime'] < limit_date)].copy()

# Nettoyage des valeurs infinies et NaN
for X_set in [train_df, valid_df]:
    numeric_cols = X_set.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        X_set[col] = X_set[col].replace([np.inf, -np.inf], np.nan)
        X_set[col] = X_set[col].fillna(X_set[col].median())

n_features = len(train_df.columns) - 2  # Exclure TARGET et LTScheduledDatetime
print(f"  - Features créées: {n_features} features")

# 2. Entraînement avec LightGBM (Modèle unifié)
print("\n2. STRATÉGIE: Modèle unifié LightGBM")

# Préparer les données
X_train = train_df.drop(columns=[TARGET, "LTScheduledDatetime"])
y_train = train_df[TARGET]
X_valid = valid_df.drop(columns=[TARGET, "LTScheduledDatetime"])
y_valid = valid_df[TARGET]

# Nettoyage des inf/nan
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_valid = X_valid.replace([np.inf, -np.inf], np.nan)
for col in X_train.select_dtypes(include=[np.number]).columns:
    X_train[col] = X_train[col].fillna(X_train[col].median())
    X_valid[col] = X_valid[col].fillna(X_valid[col].median())

# Conversion des catégories
for col in X_train.select_dtypes(include=['object']).columns:
    X_train[col] = X_train[col].astype('category')
    X_valid[col] = X_valid[col].astype('category')

print(f"   Training data: {len(X_train)} rows, {len(X_train.columns)} features")
print(f"   Validation data: {len(X_valid)} rows\n")

# Modèle LightGBM optimisé
print("  Training LightGBM model...")
model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=5000,
    learning_rate=0.02,
    num_leaves=31,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    verbose=-1,
    max_depth=12,
    min_data_in_leaf=10,
    lambda_l1=2.0,
    lambda_l2=2.0,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="mae",
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

# 3. Prédiction et évaluation
print("3. Predict and evaluate...\n")

predictions = np.maximum(0, model.predict(X_valid))

mae = mean_absolute_error(y_valid, predictions)
r2 = r2_score(y_valid, predictions)
rmse = root_mean_squared_error(y_valid, predictions)

print(f"\n── RÉSULTATS FINAUX ──")
print(f"   MAE : {mae:.2f} passengers")    
print(f"   R²  : {r2:.4f}")
print(f"   RMSE : {rmse:.2f}")

# 4. Feature importance
print("\n4. Feature importance\n")

imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).nlargest(15, "importance")

print("   Top 15 features:")
print(imp.to_string(index=False))

# 5. Analysis 
# Plots Pred vs Real
n_sample = 300 
plt.figure(figsize=(15, 7))
plt.plot(y_valid.values[:n_sample], label='Valeurs Réelles (NbPax)', color='royalblue', linewidth=2, marker='o', markersize=4)
plt.plot(predictions[:n_sample], label='Prédictions LightGBM', color='darkorange', linestyle='--', linewidth=2, marker='x', markersize=5)
plt.title(f'Comparaison : NbPax Réel vs Prédit (Échantillon de {n_sample} vols)')
plt.ylabel('Nombre de Passagers (NbPaxTotal)')
plt.xlabel('Index du Vol (Validation)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Top 10 big errors
analysis_df = valid_df.copy()
analysis_df['Predicted_NbPax'] = predictions
analysis_df['Abs_Error'] = np.abs(analysis_df[TARGET] - analysis_df['Predicted_NbPax'])
top_errors = analysis_df.sort_values(by='Abs_Error', ascending=False).head(10)

columns_to_show = ["FlightNumberNormalized",'LTScheduledDatetime', TARGET, 'Predicted_NbPax', 'Abs_Error']

print(f"── TOP 10 ERRORS ──")
top_errors_display = top_errors[columns_to_show].copy()
top_errors_display['Predicted_NbPax'] = top_errors_display['Predicted_NbPax'].round(1)
top_errors_display['Abs_Error'] = top_errors_display['Abs_Error'].round(1)

print(top_errors_display.to_string(index=False))