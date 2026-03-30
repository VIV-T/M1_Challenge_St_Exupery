import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. CHARGEMENT ET PRÉPARATION
# ==========================================
base_path = Path(__file__).parent.parent.parent if "__file__" in locals() else Path(".")
filename = base_path / "data" / "main_preprocessed.csv"
params_path = Path("best_xgb_params.json")

df = pd.read_csv(filename, encoding='utf-8')
df = df[df['NbOfSeats'] > 0].copy()

df['ds'] = pd.to_datetime(df[['LTScheduledYear', 'LTScheduledMonth', 'LTScheduledDay', 'LTScheduledHour', 'LTScheduledMinute']]
                         .rename(columns={'LTScheduledYear': 'year', 'LTScheduledMonth': 'month', 
                                          'LTScheduledDay': 'day', 'LTScheduledHour': 'hour', 'LTScheduledMinute': 'minute'}))
df['y'] = df['OccupancyRate']
df = df.sort_values('ds')

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
df['month'] = df['ds'].dt.month
df['day_of_week'] = df['ds'].dt.dayofweek
df['hour'] = df['ds'].dt.hour
df['month_sin'] = np.sin(2 * np.pi * df['month']/12.0)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12.0)
df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)


cutoff_date = pd.Timestamp('2025-07-01')
train_mask = df['ds'] < cutoff_date
route_stats = df[train_mask].groupby(['AirportOrigin', 'AirportPrevious'])['y'].mean().to_dict()
global_mean = df[train_mask]['y'].mean()
df['Route_Mean'] = df.apply(lambda x: route_stats.get((x['AirportOrigin'], x['AirportPrevious']), global_mean), axis=1)

# Interactions
# df['Route_x_Stopover'] = df['Route_Mean'] * (df['SysStopover'] + 1)
df['Holiday_x_Weekend'] = df['IsScholarHolidays'] * df['IsWeekend']
df['Route_x_DoW'] = df['Route_Mean'] * df['day_of_week']
# df['Route_x_Aircraft'] = df['Route_Mean'] * (df['IdAircraftType'] + 1)

features = [
    'NbOfSeats', 'Direction', 'IdBusinessUnitType', 'SysStopover', 
    'AirportOrigin', 'AirportPrevious', 'IdAircraftType',
    'IsWeekend', 'IsPublicHolidays', 'IsScholarHolidays', 'Route_Mean', 'day_of_week',
    'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
    'Route_x_Stopover', 'Holiday_x_Weekend', 'Route_x_DoW', 'Route_x_Aircraft'
]

features = [
    'NbOfSeats', 'Direction', 'IdBusinessUnitType', 'SysStopover', 
    'AirportOrigin', 'AirportPrevious', 'IdAircraftType',
    'IsWeekend', 'IsPublicHolidays', 'IsScholarHolidays', 'Route_Mean', 'day_of_week',
    'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
    'Holiday_x_Weekend', 'Route_x_DoW'
]

X_train, y_train = df[train_mask][features], df[train_mask]['y']
X_test, y_test = df[~train_mask][features], df[~train_mask]['y']
test_df = df[~train_mask].copy()

# ==========================================
# 3. OPTIMISATION OPTUNA (Optionnelle si JSON existe)
# ==========================================
if not params_path.exists():
    print("Fichier JSON non trouvé. Lancement de l'optimisation Optuna...")
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 1800),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'early_stopping_rounds': 50,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mae'
        }
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        return mean_absolute_error(y_test, model.predict(X_test).clip(0, 1))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    # Sauvegarde immédiate après optimisation
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
else:
    print(f"Chargement des meilleurs paramètres depuis {params_path}...")
    with open(params_path, 'r') as f:
        best_params = json.load(f)

# ==========================================
# 4. ENTRAÎNEMENT FINAL
# ==========================================
# On s'assure que les paramètres fixes sont présents
best_params.update({'random_state': 42, 'early_stopping_rounds': 50, 'n_jobs': -1})

final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
final_model.save_model("airport_model_2026.json")

# Prédictions
test_df['Pred_Occupancy'] = final_model.predict(X_test).clip(0, 1)
test_df['Pred_Pax'] = (test_df['Pred_Occupancy'] * test_df['NbOfSeats']).round()
test_df['Real_Pax'] = (test_df['y'] * test_df['NbOfSeats']).round()

# ==========================================
# 5. VALIDATION ET TESTS
# ==========================================
def run_all_tests(results):
    print("\n" + "="*40)
    print("RAPPORT DE VALIDATION (PRODUCTION)")
    print("="*40)
    mae_pax = mean_absolute_error(results['Real_Pax'], results['Pred_Pax'])
    r2 = r2_score(results['y'], results['Pred_Occupancy'])
    print(f"[METRICS] MAE : {mae_pax:.2f} passagers | R² : {r2:.4f}")
    
    # Test par AircraftType (Top 5 plus grosses erreurs)
    results['Error_Pax'] = abs(results['Real_Pax'] - results['Pred_Pax'])
    aircraft_err = results.groupby('IdAircraftType')['Error_Pax'].mean().sort_values(ascending=False)
    print("\n[DEBUG] Top 5 erreurs par Type d'avion (ID):")
    print(aircraft_err.head(5))

run_all_tests(test_df)

# ==========================================
# 6. VISUALISATION
# ==========================================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
pd.Series(final_model.feature_importances_, index=features).nlargest(12).plot(kind='barh')
plt.title("Importance des Variables")

plt.subplot(1, 2, 2)
plt.scatter(test_df['Real_Pax'], test_df['Pred_Pax'], alpha=0.2, s=5, color='orange')
plt.plot([0, test_df['Real_Pax'].max()], [0, test_df['Real_Pax'].max()], 'k--')
plt.title("Réel vs Prédit")
plt.show()