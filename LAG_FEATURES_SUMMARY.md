# Vérification et Amélioration des Lag Features

## 📋 Résumé des Changements

### 1. **Vérification des Features Existantes**
✅ Les features historiques existantes sont correctement calculées :
- **AvgOcc_1W, 1M, 3M, 6M, 1Y** : Moyennes par route sur fenêtres temporelles (merge_asof + rolling window)
- **FlightOcc_1W, 1M, 3M, 6M, 1Y** : Moyennes historiques spécifiques au vol
- **AvgPax_AirportOrigin, AvgPax_AirportPrevious** : Moyennes par aéroport

### 2. **Nouvelles Lag/Trend Features Ajoutées**

#### A. **TREND FEATURES** (6 features)
Capturent les changements temporels sans data leakage :
```
- Trend_OccWeekToMonth = AvgOcc_1W - AvgOcc_1M
- Trend_OccMonthTo3M = AvgOcc_1M - AvgOcc_3M
- Trend_OccMonthToYear = AvgOcc_1M - AvgOcc_1Y
- Trend_Occ3MToYear = AvgOcc_3M - AvgOcc_1Y
- Trend_FlightWeekToMonth = FlightOcc_1W - FlightOcc_1M
- Trend_FlightMonthToYear = FlightOcc_1M - FlightOcc_1Y
```
**Utilité** : Détecte les tendances d'occupation (croissance/décroissance)

#### B. **VOLATILITY FEATURES** (2 features)
Mesurent la prévisibilité/variabilité des vols :
```
- Volatility_Route = |AvgOcc_1M - AvgOcc_6M| / AvgOcc_6M
- Volatility_Flight = |FlightOcc_1W - FlightOcc_1Y| / FlightOcc_1Y
```
**Utilité** : Certains vols sont stables, d'autres imprévisibles

#### C. **SEASONAL FEATURES** (2 features)
Capturent les variations saisonnières :
```
- Seasonal_Ratio_Route = AvgOcc_1M / AvgOcc_1Y
- Seasonal_Ratio_Flight = FlightOcc_1M / FlightOcc_1Y
```
**Utilité** : Permet de normaliser par rapport à la saisonnalité

#### D. **DAY-OF-WEEK FEATURES** (2 features)
Patterns spécifiques au jour de la semaine :
```
- AvgOcc_DayOfWeek_1W : Moyenne historique pour le MÊME jour (semaine)
- AvgOcc_DayOfWeek_1M : Moyenne historique pour le MÊME jour (mois)
```
**Utilité** : Les lundis ≠ vendredis en terme d'occupation

#### E. **HOUR-SPECIFIC FEATURES** (2 features)
Patterns spécifiques à l'heure du décollage :
```
- AvgOcc_ByHour_1W : Moyenne historique pour cette heure (semaine)
- AvgOcc_ByHour_1M : Moyenne historique pour cette heure (mois)
```
**Utilité** : Vols du matin vs soir ont des patterns très différents

#### F. **ROLLING STATISTICS** (3 features)
Volatilité mesurée sur une fenêtre glissante :
```
- Occ_Std_7D : Écart-type de l'occupation sur 7 jours passés
- Occ_Min_7D : Occupation minimale sur 7 jours
- Occ_Max_7D : Occupation maximale sur 7 jours
```
**Utilité** : Capture la variabilité récente (risque/prévisibilité)

---

## 🛡️ Vérification : PAS DE DATA LEAKAGE

Toutes les nouvelles features sont basées UNIQUEMENT sur des données passées :

| Feature Type | Data Source | Leakage Risk |
|---|---|---|
| Trend Features | AvgOcc historiques (passé) | ✅ SAFE |
| Volatility | Variances historiques | ✅ SAFE |
| Seasonal | Ratios année/mois précédents | ✅ SAFE |
| DayOfWeek | Moyennes historiques | ✅ SAFE |
| Hour-specific | Moyennes historiques par heure | ✅ SAFE |
| Rolling Stats | Fenêtre glissante passée | ✅ SAFE |

⚠️ **Important** : Ces features NE contiennent jamais d'information FUTURE

---

## 📊 Implémentation

### Fichiers Modifiés
1. **`scripts/data_preparation/utils/main/add_features.py`**
   - ✅ 4 nouvelles fonctions ajoutées :
     - `add_lag_trend_features()` 
     - `add_dayofweek_histories()`
     - `add_hour_histories()`
     - `add_rolling_statistics()`

2. **`scripts/data_preparation/preprocessed.py`**
   - ✅ Imports mises à jour
   - ✅ 4 appels aux nouvelles fonctions dans le pipeline

### Comment les Utiliser

```python
# Dans le pipeline de preprocessing :
data = add_optimized_historical_features(df=data)
data = add_optimized_flight_history(df=data)
# Les nouvelles features basées sur les historiques existants :
data = add_lag_trend_features(df=data)           # +6 features
data = add_dayofweek_histories(df=data)          # +2 features
data = add_hour_histories(df=data)               # +2 features
data = add_rolling_statistics(df=data)           # +3 features
```

**Total de nouvelles features** : ~13 nouvelles colonnes

---

## 🧪 Test & Validation

Run le script de test pour vérifier les features :
```bash
python scripts/test_lag_features.py
```

Ce script :
- ✅ Valide que toutes les features sont présentes
- ✅ Compte le nombre de valeurs valides vs NaN
- ✅ Affiche les statistiques (mean, std, min, max)
- ✅ Confirme absence de data leakage

---

## 📈 Exemple de Bénéfice Attendu

Ces features doivent aider le modèle à :
1. **Détecter les tendances** → Mieux prédire les changements
2. **Gérer la saisonnalité** → Normaliser les variations périodiques
3. **Identifier les patterns intra-jour** → Vols du matin ≠ nuit
4. **Évaluer la prévisibilité** → Certains vols sont naturellement variables

**Impact attendu sur MAE** : -1 à -2 points (18.51 → ~16.5-17.5)

---

## ⚠️ Notes Importantes

### Ordre d'Exécution
Les nouvelles fonctions DOIVENT être appelées APRÈS :
- ✅ `add_optimized_historical_features()` (crée AvgOcc_*)
- ✅ `add_optimized_flight_history()` (crée FlightOcc_*)

### IdRoute Dependency
- `add_dayofweek_histories()` et `add_hour_histories()` nécessitent la colonne `IdRoute`
- ✅ Créée par `add_route_index()` avant leur appel

### Cold Start Handling
Toutes les figures utilisent le `.fillna()` pour gérer les valeurs manquantes au démarrage :
- Moyenne progressive de la route
- Moyenne globale du dataset
