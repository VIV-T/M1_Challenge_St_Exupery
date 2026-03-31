import pandas as pd
import numpy as np
from datetime import date
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

TARGET = "NbPaxTotal"

df = pd.read_csv("data/main_preprocessed_new.csv")

print("1. Load and prepare data...")
# limit_date = pd.Timestamp("2026-03-01")
# yesterday = pd.Timestamp(date.today() - pd.Timedelta(days=10))

train_df = df[df['LTScheduledDatetime'] < "2025-12-31"].copy()
valid_df = df[(df['LTScheduledDatetime'] >= "2025-12-31") & (df['LTScheduledDatetime'] < "2026-01-30")].copy()
print(f"   Train set: {len(train_df)} rows")
print(f"   Valid set: {len(valid_df)} rows")


# train_df = df[df['LTScheduledDatetime'] < "2026-02-28"].copy()
# valid_df = df[(df['LTScheduledDatetime'] >= "2026-03-01") & (df['LTScheduledDatetime'] < "2026-03-23")].copy()

X_train = train_df.drop(columns=[TARGET, "LTScheduledDatetime"])
y_train = train_df[TARGET]
X_valid = valid_df.drop(columns=[TARGET, "LTScheduledDatetime"])
y_valid = valid_df[TARGET]


# Conversion en catégorie pour LightGBM
for col in X_train.select_dtypes(include=['object']).columns:
    X_train[col] = X_train[col].astype('category')
    X_valid[col] = X_valid[col].astype('category')

print("2. Train LightGBM ...")

model = lgb.LGBMRegressor(
    objective="regression",      
    n_estimators=10000,      
    learning_rate=0.01,
    num_leaves=60,               
    min_child_samples=5,
    feature_fraction=0.8,
    random_state=42,
    verbose=-1
)

model.fit(
    X_train, y_train,         # Utilisation de y_train_log
    eval_set=[(X_valid, y_valid)], # Evaluation sur y_valid_log
    eval_metric="mae",
    callbacks=[lgb.early_stopping(100)]
)

# 3. Prédiction et évaluation
print("3. Predict and evaluate...\n")


predictions = model.predict(X_valid)
predictions = np.maximum(0, predictions)

mae = mean_absolute_error(y_valid, predictions)
r2 = r2_score(y_valid, predictions)
rmse = root_mean_squared_error(y_valid, predictions)

print(f"\n── RÉSULTATS FINAUX ──")
print(f"   MAE : {mae:.2f} passengers")    
print(f"   R²  : {r2:.4f}")
print(f"   RMSE : {rmse:.2f}")


# 4. Feature importance
print("\n4. Feature importance\n")
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)
importance.sort_values(by="importance", ascending=False).to_csv("feature_importance.csv", index=False)
plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=importance.head(20))
plt.title("Top Features (NbPaxTotal)")
plt.tight_layout()
plt.show()

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

columns_to_show = ["IdMovement","FlightNumberNormalized",'LTScheduledDatetime', TARGET, 'Predicted_NbPax', "NbOfSeats", 'Abs_Error']

print(f"── TOP 10 ERRORS ──")
top_errors_display = top_errors[columns_to_show].copy()
top_errors_display['Predicted_NbPax'] = top_errors_display['Predicted_NbPax'].round(1)
top_errors_display['Abs_Error'] = top_errors_display['Abs_Error'].round(1)

print(top_errors_display.to_string(index=False))