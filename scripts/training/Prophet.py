import os
import json
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import optuna
import matplotlib.pyplot as plt
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

df = pd.read_csv(filename, encoding='utf-8')
df = df[df['NbOfSeats'] > 0].copy()

# Prophet requiert impérativement les colonnes 'ds' et 'y'
df['ds'] = pd.to_datetime(df[['LTScheduledYear', 'LTScheduledMonth', 'LTScheduledDay', 'LTScheduledHour', 'LTScheduledMinute']]
                         .rename(columns={'LTScheduledYear': 'year', 'LTScheduledMonth': 'month', 
                                          'LTScheduledDay': 'day', 'LTScheduledHour': 'hour', 'LTScheduledMinute': 'minute'}))
df['y'] = df['OccupancyRate']
df = df.sort_values('ds')

# ==========================================
# 2. FEATURE ENGINEERING & VARIABLES CROISÉES
# ==========================================
# Statistiques historiques pour la Route_Mean (sur données pre-2025)
cutoff_date = pd.Timestamp('2025-07-01')
train_mask = df['ds'] < cutoff_date
route_stats = df[train_mask].groupby(['AirportOrigin', 'AirportPrevious'])['y'].mean().to_dict()
global_mean = df[train_mask]['y'].mean()
df['Route_Mean'] = df.apply(lambda x: route_stats.get((x['AirportOrigin'], x['AirportPrevious']), global_mean), axis=1)

# Variables Croisées pour Prophet
df['Route_x_Stopover'] = df['Route_Mean'] * (df['SysStopover'] + 1)
df['IsWeekend'] = df['IsWeekend'].astype(float)
df['SysStopover'] = df['SysStopover'].astype(float)

# Liste des régresseurs externes
regressors = ['NbOfSeats', 'Direction', 'SysStopover', 'IsWeekend', 'Route_Mean', 'Route_x_Stopover']


### DF Holidays for Prophet

# 1. Filtrer les dates où au moins un des deux indicateurs est vrai
holidays_df = df[['ds', 'IsPublicHolidays', 'IsScholarHolidays']].copy()
holidays_df = holidays_df[(holidays_df['IsPublicHolidays'] == 1) | (holidays_df['IsScholarHolidays'] == 1)]

# 2. Créer une colonne 'holiday' qui décrit le type de jour férié
holidays_df['holiday'] = holidays_df.apply(
    lambda x: 'PublicHoliday' if x['IsPublicHolidays'] == 1 else 'ScholarHoliday',
    axis=1
)

# 3. Garder uniquement les colonnes 'ds' et 'holiday'
holidays_df = holidays_df[['ds', 'holiday']].drop_duplicates()



# Split Train / Test
train_df = df[train_mask][['ds', 'y'] + regressors].copy()
test_df = df[~train_mask][['ds', 'y'] + regressors].copy()

# ==========================================
# 3. OPTIMISATION OPTUNA (Hyperparamètres Prophet)
# ==========================================
def objective(trial):
    # Paramètres spécifiques à Prophet
    params = {
        'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
        'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10.0, log=True),
        'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10.0, log=True),
        'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
    }
    
    model = Prophet(
        **params,
        holidays=holidays_df,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    for reg in regressors:
        model.add_regressor(reg)
    
    model.fit(train_df)
    
    # Prédiction sur la période de test
    forecast = model.predict(test_df.drop(columns='y'))
    mae = mean_absolute_error(test_df['y'], forecast['yhat'].clip(0, 1))
    return mae

print("Lancement de l'optimisation Optuna pour Prophet...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15) # Prophet est plus lent, on limite les essais

# ==========================================
# 4. ENTRAÎNEMENT FINAL ET SAUVEGARDE
# ==========================================
best_params = study.best_params
with open('best_prophet_params.json', 'w') as f:
    json.dump(best_params, f, indent=4)

final_model = Prophet(
    **best_params,
    holidays=holidays_df,
    yearly_seasonality=True,
    weekly_seasonality=True
)

for reg in regressors:
    final_model.add_regressor(reg)

final_model.fit(train_df)

# Prédictions finales
forecast_test = final_model.predict(test_df.drop(columns='y'))
test_df['Pred_Occupancy'] = forecast_test['yhat'].values.clip(0, 1)
test_df['Pred_Pax'] = (test_df['Pred_Occupancy'] * test_df['NbOfSeats']).round()
test_df['Real_Pax'] = (test_df['y'] * test_df['NbOfSeats']).round()

# ==========================================
# 5. TESTS DE VALIDATION & VISUALISATION
# ==========================================
def run_prophet_tests(results):
    print("\n" + "="*40)
    print("RAPPORT DE VALIDATION PROPHET")
    print("="*40)
    mae_pax = mean_absolute_error(results['Real_Pax'], results['Pred_Pax'])
    r2 = r2_score(results['y'], results['Pred_Occupancy'])
    print(f"MAE : {mae_pax:.2f} passagers | R² : {r2:.4f}")
    
    # Test de cohérence
    over_cap = len(results[results['Pred_Pax'] > results['NbOfSeats']])
    print(f"Vols en surcapacité : {over_cap}")

run_prophet_tests(test_df)

# Visualisation des composantes (Tendance, Semaine, Année)
fig = final_model.plot_components(forecast_test)
plt.show()

# Comparaison Temporelle
plt.figure(figsize=(12, 6))
plt.plot(test_df['ds'], test_df['Real_Pax'], label='Réel', alpha=0.5)
plt.plot(test_df['ds'], test_df['Pred_Pax'], label='Prophet', color='red', linestyle='--')
plt.title("Prédiction Prophet vs Réel (2025 H2)")
plt.legend()
plt.show()