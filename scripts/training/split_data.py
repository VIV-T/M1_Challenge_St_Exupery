import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np
import os 
from pathlib import Path
from datetime import datetime, timedelta


### Initialization
data_folder = os.path.join(Path(__file__).parent.parent.parent, 'data')
dataset_filename = os.path.join(data_folder, 'dataset_training.csv')
training_metadata_folder = os.path.join(data_folder, "training_metadata")
df = pd.read_csv(dataset_filename, encoding='utf-8')


# Sort the df following the chronological order
today = datetime.now().date()

# 3. Calculer hier en soustrayant un delta de 1 jour
yesterday = today - timedelta(days=1)

df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime'])
df = df[df['LTScheduledDatetime'].dt.date <= yesterday]
df = df.sort_values(by='LTScheduledDatetime').reset_index(drop=True)

target_col = 'NbPaxTotal'


avg_nb_pax_total = df[target_col].mean()


# First split: Train (90%) / Test (10%)
train_set, test_set = train_test_split(df, test_size=0.10, shuffle=False)

# Préparation du Test Set final
y_test = test_set[target_col]
X_test_all = test_set.drop(columns=[target_col])
X_test_all.select_dtypes(exclude=['number']).to_csv(os.path.join(training_metadata_folder, 'X_test_metadata.csv'), index=False)
X_test = X_test_all.select_dtypes(include=['number'])

# Préparation du Train Set global (numérique)
train_data = train_set.select_dtypes(include=['number'])

### TimeSeriesSplit avec XGBoost
tscv = TimeSeriesSplit(n_splits=5)
scores = []

print("Début de la validation croisée...")

for i, (train_index, val_index) in enumerate(tscv.split(train_data)):
    # Découpage
    train_fold = train_data.iloc[train_index]
    val_fold = train_data.iloc[val_index]

    # Isolation de la cible
    X_train = train_fold.drop(columns=[target_col])
    y_train = train_fold[target_col]
    
    X_val = val_fold.drop(columns=[target_col])
    y_val = val_fold[target_col]

    # Initialisation du modèle XGBoost Regression
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        objective='reg:squarederror',
        random_state=42
    )

    # Entraînement avec arrêt précoce (early stopping) pour éviter l'overfitting
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Prédiction et évaluation
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    scores.append(rmse)
    
    print(f"Fold {i+1} - RMSE: {rmse:.2f}")

mean_error_val = np.mean(scores)
print(f"\nRMSE moyen sur la validation: {mean_error_val:.2f}")

### Entraînement final (sur tout le train_set) et test
# On ré-entraîne sur 90% des données pour prédire les 10% de test
final_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=500)
final_model.fit(train_data.drop(columns=[target_col]), train_data[target_col])

test_preds = final_model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
print(f"RMSE FINAL sur le Test Set: {final_rmse:.2f}")



print(f"Average number of pax total:{avg_nb_pax_total}")
print(f"Mean error on validation sets:{mean_error_val/avg_nb_pax_total:.2}")
print(f"Error on test set:{final_rmse/avg_nb_pax_total:.2}")