import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib
import os
import holidays
from sklearn.metrics import mean_absolute_error, r2_score
import optuna
from pathlib import Path

# 1. CONFIGURATION
FEATURE_COLS = [
    #"IdMovement",
    #"IdADL",
    #"IdIrregularityCode",
    "IdAircraftType",
    "IdBusinessUnitType",
    "IdBusContactType",
    "IdTerminalType",
    "IdBagStatusDelivery",
    "NbFlight",
    "AirportCode",
    "airlineOACICode",
    "SysStopover",
    "AirportOrigin",
    "AirportPrevious",
    "ServiceCode",
    "flightNumber",
    "OperatorFlightNumber",
    "FlightNumberNormalized",
    "OperatorOACICodeNormalized",
    "Direction",
    "Terminal",
    "SysTerminal",
    "FuelProvider",
    "ScheduleType",
    "NbOfSeats",
    # "etl_origin", # Last column from the origininal dataset
    # "OccupancyRate", 
    # New features
    "IdRoute", 
    "AvgOcc_1W",
    "AvgOcc_1M",
    "AvgOcc_3M",
    "AvgOcc_6M",
    "AvgOcc_1Y",
    "FlightOcc_1W",
    "FlightOcc_1M",
    "FlightOcc_3M",
    "FlightOcc_6M",
    "FlightOcc_1Y",
    # Date/time features    
    "LTScheduledYear", 
    # "LTScheduledMonth", 
    "LTScheduledDay", 
    # "LTScheduledHour", 
    # "LTScheduledMinute", 
    # "LTScheduledDayOfWeek", 
    "cos_LTScheduledHour", "sin_LTScheduledHour",
    "cos_LTScheduledMonth", "sin_LTScheduledMonth",
    "cos_LTScheduledDayOfWeek", "sin_LTScheduledDayOfWeek",
    "cos_LTScheduledMinute", "sin_LTScheduledMinute",
    # holidays columns
    "IsPublicHolidays", "IsScholarHolidays"
]

# Colonnes à traiter comme catégories par LightGBM
CAT_COLS = [
    #"IdMovement",
    #"IdADL",
    "IdAircraftType",
    "IdBusinessUnitType",
    "IdBusContactType",
    "IdTerminalType",
    "IdBagStatusDelivery",
    "NbFlight",
    "AirportCode",
    "airlineOACICode",
    "SysStopover",
    "AirportOrigin",
    "AirportPrevious",
    "ServiceCode",
    "flightNumber",
    "OperatorFlightNumber",
    "FlightNumberNormalized",
    "OperatorOACICodeNormalized",
    "Direction",
    "Terminal",
    "SysTerminal",
    "FuelProvider",
    "ScheduleType",
    "IdRoute"
]
# On filtre CAT_COLS pour ne garder que celles présentes dans FEATURE_COLS
CAT_FEATURES = [c for c in CAT_COLS if c in FEATURE_COLS]

TARGET_COL = "OccupancyRate" # "NbPaxTotal" # On peut aussi essayer de prédire directement le taux d'occupation, mais cela peut être plus difficile à apprendre pour le modèle.

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'n_estimators': 4000,
    'learning_rate': 0.005,
    'num_leaves': 31,      # Équivalent profondeur mais plus flexible
    'max_depth': 8,
    'random_state': 42,
    'n_jobs': -1,
    'importance_type': 'gain', # Pour l'importance des variables
    'verbosity': -1
}


def prepare_aligned_data(df_subset, is_training=True):
    data = df_subset.copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if is_training:
        data = data.dropna(subset=[TARGET_COL])
    
    X = data[FEATURE_COLS].copy()
    
    # Conversion explicite des types catégoriels pour LightGBM
    for col in CAT_FEATURES:
        X[col] = X[col].astype('category')
        
    X = X.fillna(X.median(numeric_only=True))
    
    if is_training:
        y = data[TARGET_COL]
        return X, y
    else:
        return X

def plot_feature_importance(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 8))
    plt.title("Importance des Variables (LightGBM Gain)")
    plt.barh(range(len(indices)), importances[indices], color='darkorange', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Importance Relative')
    plt.tight_layout()
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/feature_importance_lgb.png")
    plt.show()



def optimize_lgbm(X_train, y_train, X_valid, y_valid, n_trials=50):
    """Optimisation des hyperparamètres de LightGBM avec Optuna"""
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )

        val_preds = model.predict(X_valid).clip(0, 1)
        mae = mean_absolute_error(y_valid, val_preds)
        return mae

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=3600)

    print("Meilleurs hyperparamètres trouvés :")
    print(study.best_params)
    print(f"Meilleur MAE : {study.best_value:.4f}")

    return study.best_params



def run_full_pipeline(csv_path):
    print("── 1. Chargement et Feature Engineering...")
    df = pd.read_csv(csv_path)
    df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime'])

    # 2. SÉPARATION TEMPORELLE
    train_df = df[df['LTScheduledDatetime'] < "2026-02-28"]
    valid_df = df[(df['LTScheduledDatetime'] >= "2026-03-01") & (df['LTScheduledDatetime'] < "2026-03-25")]
    future_df = df[df['LTScheduledDatetime'] >= "2026-01-01"]

    print("── 2. Préparation des datasets...")
    X_train, y_train = prepare_aligned_data(train_df, is_training=True)
    X_valid, y_valid = prepare_aligned_data(valid_df, is_training=True)

    # 3. OPTIMISATION AVEC OPTUNA
    print("── 3. Optimisation des hyperparamètres avec Optuna...")
    best_params = optimize_lgbm(X_train, y_train, X_valid, y_valid, n_trials=50)

    # 3. ENTRAÎNEMENT
    print(f"── 4. Entraînement LightGBM sur {len(X_train)} lignes...")
    model = lgb.LGBMRegressor(**best_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=1000),
            lgb.log_evaluation(period=100)
        ]
    )

    # 4. ÉVALUATION ET VISUALISATION
    val_preds = model.predict(X_valid).clip(0, 1)
    print(f"\n── Résultats Finale (LGBM) ──")
    print(f"   MAE: {mean_absolute_error(y_valid, val_preds):.4f}")
    print(f"   R² : {r2_score(y_valid, val_preds):.4f}")

    plot_feature_importance(model, FEATURE_COLS)

    # # 5. PRÉDICTION 2026
    # if not future_df.empty:
    #     X_2026 = prepare_aligned_data(future_df, is_training=False)
    #     preds_2026 = model.predict(X_2026).clip(0, 1)
    #     results_2026 = future_df.copy()
    #     results_2026['OccupancyRate_Prediction'] = preds_2026
    #     results_2026.to_csv("data/predictions_2026_lgb.csv", index=False)
    #     print("✅ Prédictions 2026 terminées.")

    # joblib.dump(model, "models/lgbm_final.pkl")


    # Sauvegarde et affichage
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/reel_vs_predit_lgb.png", dpi=300, bbox_inches='tight')
    plt.show()

    mae = mean_absolute_error(y_valid, val_preds)
    r2 = r2_score(y_valid, val_preds)
    plt.text(0.02, 0.95,
            f"MAE: {mae:.3f}\nR²: {r2:.3f}",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8))

    val_preds = model.predict(X_valid).clip(0, 1)
    print(f"\n── Résultats Finaux (LGBM) ──")
    print(f"   MAE: {mean_absolute_error(y_valid, val_preds):.4f}")
    print(f"   R² : {r2_score(y_valid, val_preds):.4f}")

    # ==== AJOUTE ICI LE CODE DE TRACÉ ====
    
    # Tracé des valeurs réelles vs prédites
    plt.figure(figsize=(12, 6))
    plt.scatter(y_valid, val_preds, alpha=0.3, color='blue', label='Prédictions')
    plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--', lw=2, label='Ligne idéale')
    plt.xlabel('Valeurs Réelles')
    plt.ylabel('Valeurs Prédites')
    plt.title('Comparaison Valeurs Réelles vs Prédites (Validation)')
    plt.legend()
    plt.grid(True, alpha=0.3)


if __name__ == "__main__":
    data_folder = os.path.join(Path(__file__).parent.parent.parent, "data")
    csv_path = os.path.join(data_folder, "main_preprocessed.csv")
    run_full_pipeline(csv_path)