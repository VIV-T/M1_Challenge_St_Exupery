"""
Explainations:
This python script allow us to add new features to our main dataset, based on the existing ones. 
The new features are created by calling the different functions defined in the "add_features.py" file, and can be easily modified by changing the parameters of these functions.


There are 5 different types of features created in this script:
- Date related features: creation of new features based on the date (day, month, hour, day of the week, etc.) and cyclical encoding of these features.
- Lag features: creation of lag features based on different groupings (ex: par avion, par route, etc.) and differents lags (ex: 1 day, 1 week, 1 month, 6 months, 1 year).
- Rolling features: creation of rolling features based on different aggregations and temporal windows (ex: mean of the last 30 days, max of the last 90 days, etc.).
- Trend features: creation of trend features based on the ratio between a short window and a long window.
- Lagged rolling features: creation of features based on rolling statistics, but lagged in the time (ex: mean of the last 14 days from 1 year ago). => Allow to capture historical tendencies while avoiding data leakage.




- Global ones ? (/!\) Data leakage risk, to be used with caution and only if they are based on historical data (ex: median of the last 30 days for this airline, etc.)


"""


### Imports
import pandas as pd
import numpy as np 
from datetime import date

TARGET = "NbPaxTotal"
# trimester and semester missing for now, as they are not really useful with the current features, and they can be easily added later if needed (based on the month feature)

# Column configurations
COLUMN_LIST_BASE = [
    "FlightNumberNormalized", 
    "airlineOACICode",
    "IdAircraftType",
    "SysTerminal", 
    "Direction"
    ]


# Statistics to calculate for the lag features
STATISTICS_LIST = ['mean', 'min', 'max', 'std', 'median']

# Rolling configuration
ROLLING_CONFIG = {
    "week": "7D",
    "month": "30D",
    "quarter": "91D",
    "semester": "182D",
    "year": "365D"
}


# Lags configuration
CUSTOM_LAGS = {
    "1year": pd.DateOffset(years=1),
    "6months": pd.DateOffset(months=6),
    "3months": pd.DateOffset(months=3),
    "1month": pd.DateOffset(months=1),
    "1week": pd.DateOffset(weeks=1),
    "1day": pd.DateOffset(days=1)
}

# Trend features configuration
TREND_CONFIG = {
    ("7D", "14D"),  # Trend based on the ratio between the mean of the last 7 days and the mean of the last 14 days
    ("14D", "30D"), 
    ("30D", "91D"),  
    ("91D", "182D"), 
    ("182D", "365D") 
}

# Rolling lags windows configuration --- TO MODIFY
ROLLING_LAGS_CONFIG = {
    "lag365_win28": {"lag": "365D", "window": "28D"}, # D-365  +/- 14 days
    "lag182_win20": {"lag": "182D", "window": "20D"}, # D-182  +/- 10 days
    "lag91_win20": {"lag": "91D", "window": "20D"}, # D-91  +/- 10 days
    "lag30_win14": {"lag": "30D", "window": "14D"}, # D-30  +/- 7 days
    "lag7_win6": {"lag": "7D", "window": "6D"} # D-7  +/- 3 days
}



def date_columns_creation(df : pd.DataFrame) -> pd.DataFrame:
        df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime'])
        # Creation of date related features
        df["Year"] = pd.to_datetime(df['LTScheduledDatetime']).dt.year
        df['Semester'] = np.where(df['LTScheduledDatetime'].dt.month <= 6, 1, 2)
        df['Quarter'] = df['LTScheduledDatetime'].dt.quarter
        df["Month"] = pd.to_datetime(df['LTScheduledDatetime']).dt.month
        df["Day"] = pd.to_datetime(df['LTScheduledDatetime']).dt.day
        df["Hour"] = pd.to_datetime(df['LTScheduledDatetime']).dt.hour
        df["Minute"] = pd.to_datetime(df['LTScheduledDatetime']).dt.minute
        df["DayOfWeek"] = pd.to_datetime(df['LTScheduledDatetime']).dt.dayofweek
        df['Hour_Of_Week'] = df['LTScheduledDatetime'].dt.dayofweek * 24 + df['LTScheduledDatetime'].dt.hour
        # df['Avg_Pax_Hour_Week'] = df.groupby('Hour_Of_Week')[TARGET].transform('mean')

        # Cyclical encoding
        # the cyclical encoding for hours is here on a base of 60 minutes, but we can also do it on a base of 24h, or even 168h (hour of the week)
        for col, period in [('Minute', 60), ('Hour', 24), ('Month', 12), ('DayOfWeek', 7), ('Hour_Of_Week', 168)]:
            df[f'sin_{col}'] = np.sin(2 * np.pi * df[col] / period)
            df[f'cos_{col}'] = np.cos(2 * np.pi * df[col] / period)
            df = df.drop(columns=[col])

        return df 

def add_lag_features(df: pd.DataFrame, group_cols: list, lags: dict) -> pd.DataFrame:
    # 1. NETTOYAGE DES TYPES
    for col in group_cols:
        df[col] = df[col].astype(str)
    
    df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime'], utc=True).dt.tz_localize(None).dt.floor('min')


    for lag_name, offset in lags.items():
        # 2. CALCUL DU RÉFÉRENTIEL
        # Utiliser un dictionnaire {TARGET: stats} résout l'erreur Pylance
        temp_df = (
            df.groupby(group_cols + ['LTScheduledDatetime'])
            .agg({TARGET: STATISTICS_LIST})
            .reset_index()
        )

        # Après l'agrégation sur une seule colonne avec une liste, 
        # Pandas crée souvent un MultiIndex pour les colonnes. On l'aplatit.
        temp_df.columns = group_cols + ['LTScheduledDatetime'] + STATISTICS_LIST

        # 3. APPLICATION DU LAG
        temp_df['LTScheduledDatetime'] = temp_df['LTScheduledDatetime'] + offset
        
        # 4. RÉ-AGRÉGATION (Gestion des collisions après shift)
        # On regroupe par date et on moyenne les statistiques calculées
        temp_df = (
            temp_df.groupby(group_cols + ['LTScheduledDatetime'])
            .agg({s: 'mean' for s in STATISTICS_LIST})
            .reset_index()
        )

        # 5. RENOMMAGE FINAL
        rename_dict = {
            s: f"{'_'.join(group_cols)}_lag_{lag_name}_{s}" 
            for s in STATISTICS_LIST
        }
        temp_df = temp_df.rename(columns=rename_dict)

        # 6. MERGE
        df = pd.merge(
            df, 
            temp_df, 
            on=group_cols + ['LTScheduledDatetime'], 
            how='left'
        )
        
        import gc
        del temp_df
        gc.collect()

    return df

def add_rolling_features(df: pd.DataFrame, group_cols: list, windows: dict) -> pd.DataFrame:
    """
    Calcule des statistiques glissantes basées sur des durées réelles (fenêtres temporelles).
    
    Params:
    - df: DataFrame contenant 'LTScheduledDatetime' et la cible TARGET.
    - group_cols: Colonnes de regroupement (ex: ['airlineOACICode', 'FlightNumberNormalized']).
    - windows: Dictionnaire {nom_feature: offset_string} 
               Ex: {"month": "30D", "quarter": "91D", "semester": "182D", "year": "365D"}
    """
    # 1. Tri indispensable
    df = df.sort_values(by=group_cols + ['LTScheduledDatetime'])
    
    # 2. Indexation temporelle sur une copie pour le calcul par offset
    df_indexed = df.set_index('LTScheduledDatetime')
    
    # Liste des stats à calculer
    stats_list = ['mean', 'min', 'max', 'std', 'median']
    
    for name, window_size in windows.items():
        # 3. Groupement et création de la fenêtre glissante
        # closed='left' est crucial pour éviter le Data Leakage
        rolling_group = (
            df_indexed.groupby(group_cols)[TARGET]
            .rolling(window=window_size, closed='left', min_periods=1)
        )
        
        # 4. Calcul de toutes les statistiques d'un coup
        # On utilise .agg() pour calculer la liste complète
        temp_rolling_stats = rolling_group.agg(stats_list).reset_index()
        
        # 5. Réalignement et renommage des colonnes
        prefix = f"{'_'.join(group_cols)}_rolling_{name}"
        
        for stat in stats_list:
            col_name = f"{prefix}_{stat}"
            # On utilise .values pour s'assurer que l'ordre des lignes (triées en étape 1) 
            # correspond parfaitement au DataFrame original
            df[col_name] = temp_rolling_stats[stat].values
            
            # Optimisation RAM : passage en float32
            df[col_name] = df[col_name].astype('float32')

        # Nettoyage mémoire entre chaque fenêtre
        import gc
        del temp_rolling_stats
        gc.collect()

    return df

def add_trend_features(df: pd.DataFrame, group_cols: list, short_win: str = "7D", long_win: str = "30D") -> pd.DataFrame:
    """
    Calcule les ratios de tendance (Trend = Valeur_Courte / Valeur_Longue)
    pour chaque statistique de STATISTICS_LIST.
    """
    df = df.sort_values(by=group_cols + ['LTScheduledDatetime'])
    df_indexed = df.set_index('LTScheduledDatetime')
    
    stats_list = ['mean', 'min', 'max', 'std', 'median']
    
    # 1. Calcul groupé des agrégations
    short_rolling = (
        df_indexed.groupby(group_cols)[TARGET]
        .rolling(window=short_win, closed='left', min_periods=1)
        .agg(stats_list)
        .reset_index()
    )
    
    long_rolling = (
        df_indexed.groupby(group_cols)[TARGET]
        .rolling(window=long_win, closed='left', min_periods=1)
        .agg(stats_list)
        .reset_index()
    )
    
    # 2. Collecte des nouvelles colonnes dans un dictionnaire
    new_features = {}
    prefix = f"{'_'.join(group_cols)}_trend_{short_win}_vs_{long_win}"
    
    for stat in stats_list:
        s_values = short_rolling[stat].values
        l_values = long_rolling[stat].values
        
        col_name = f"{prefix}_{stat}"
        
        # Calcul du ratio
        new_features[col_name] = np.divide(
            s_values, 
            l_values, 
            out=np.ones_like(s_values, dtype='float32'), 
            where=l_values > 0
        ).astype('float32')

    # 3. Concaténation massive (Une seule opération d'écriture)
    # On crée un DataFrame à partir du dictionnaire et on le joint au DF original
    new_cols_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_cols_df], axis=1)

    # Nettoyage
    import gc
    del short_rolling, long_rolling, new_features, new_cols_df
    gc.collect()
    
    return df


def add_lagged_rolling_features(df: pd.DataFrame, group_cols: list, lag: str, window: str, new_col_name: str) -> pd.DataFrame:
    """
    Calcule des statistiques mobiles (STATISTICS_LIST), puis les décale dans le temps.
    Utilise une approche de fusion pour éviter la fragmentation.
    """
    # 1. Préparation et tri
    # On s'assure que les dates sont propres et sans timezone pour éviter les erreurs de merge
    df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime']).dt.floor('min')
    df = df.sort_values(by=group_cols + ['LTScheduledDatetime'])
    
    # 2. Calcul des stats mobiles sur le flux actuel
    df_temp = df.set_index('LTScheduledDatetime')
    
    # Liste des statistiques demandées
    stats_list = ['mean', 'min', 'max', 'std', 'median']
    
    # Calcul groupé de toutes les stats
    # center=True permet de regarder +/- (window/2) autour du point historique
    rolling_gen = (
        df_temp.groupby(group_cols)[TARGET]
        .rolling(window=window, min_periods=1, center=True)
        .agg(stats_list)
        .reset_index()
    )
    
    # 3. LE SAUT DANS LE TEMPS (Le Lag)
    # On décale la date pour que l'historique s'aligne sur le présent
    rolling_gen['LTScheduledDatetime'] = rolling_gen['LTScheduledDatetime'] + pd.to_timedelta(lag)
    
    # Sécurité : Ré-agrégation si deux dates fusionnent après le décalage (ex: heure d'été)
    stats_df = (
        rolling_gen.groupby(group_cols + ['LTScheduledDatetime'])[stats_list]
        .mean()
        .reset_index()
    )
    
    # 4. RENOMMAGE DES COLONNES
    # On crée un dictionnaire de renommage pour identifier clairement la feature
    rename_dict = {s: f"{new_col_name}_{s}" for s in stats_list}
    stats_df = stats_df.rename(columns=rename_dict)
    
    # Optimisation mémoire avant le merge
    for col in rename_dict.values():
        stats_df[col] = stats_df[col].astype('float32')
    
    # 5. FUSION UNIQUE
    # Utiliser merge est ici plus sûr que concat car on aligne des dates décalées
    df = pd.merge(df, stats_df, on=group_cols + ['LTScheduledDatetime'], how='left')
    
    # Nettoyage RAM
    import gc
    del rolling_gen, stats_df
    gc.collect()
    
    return df



from typing import cast
import numpy as np
import numpy.typing as npt
import pandas as pd

def add_interaction_features(df: pd.DataFrame, base_col: str, feature_pattern: str, suffix: str = "x") -> pd.DataFrame:
    # 1. Identification des colonnes cibles
    # On s'assure que 'c' est traité comme une string pour la recherche du pattern
    target_cols = [c for c in df.columns if feature_pattern in str(c) and c != base_col]
    
    if not target_cols:
        return df

    new_interactions = {}
    
    # 2. Extraction optimisée de la colonne de base (NbOfSeats)
    # .flatten() garantit un vecteur 1D de forme (N,) pour éviter les erreurs de dimension
    base_vals = cast(npt.NDArray[np.float32], df[base_col].values.astype(np.float32)).flatten()

    for col in target_cols:
        new_col_name = f"INT_{base_col}_{suffix}_{col}"
        
        # 3. Extraction de la feature cible avec conversion forcée en 1D
        # On utilise .flatten() pour s'assurer que même si 'col' renvoie plusieurs colonnes 
        # identiques (doublons de noms), le calcul ne plante pas (ou on ne prend que la 1ère)
        feat_vals = cast(npt.NDArray[np.float32], df[col].values.astype(np.float32))
        
        # Si feat_vals est une matrice (N, k) à cause de noms dupliqués, on prend la première colonne
        if feat_vals.ndim > 1:
            feat_vals = feat_vals[:, 0]
        else:
            feat_vals = feat_vals.flatten()

        # 4. Calcul mathématique (Vecteur * Vecteur)
        if base_vals.shape[0] == feat_vals.shape[0]:
            new_interactions[new_col_name] = base_vals * feat_vals
        else:
            print(f"Saut de la colonne {col} : dimensions incompatibles.")

    if not new_interactions:
        return df

    # 5. Création du DataFrame de features
    new_df = pd.DataFrame(new_interactions, index=df.index)
    
    # 6. Concaténation avec gestion de l'index
    try:
        # Si l'index est propre, concat est ultra rapide
        df = pd.concat([df, new_df], axis=1)
    except ValueError:
        # En cas de "duplicate labels", on synchronise les index via reset_index
        print(f"Index dupliqués détectés lors de l'interaction {feature_pattern}. Correction en cours...")
        df = df.reset_index(drop=True)
        new_df.index = df.index
        df = pd.concat([df, new_df], axis=1)
    
    return df



def add_momentum_features(df: pd.DataFrame, short_term_pattern: str, long_term_pattern: str, suffix: str = "div") -> pd.DataFrame:
    """
    Calcule des ratios de dynamique entre deux types de features existantes.
    Exemple : short_term_pattern='rolling_7D', long_term_pattern='rolling_30D'
    """
    # 1. Identifier les paires de colonnes correspondantes
    # On cherche les colonnes qui ont la même statistique (mean, max, etc.) 
    # mais des horizons temporels différents.
    stats_list = ['mean', 'min', 'max', 'std', 'median']
    new_momentum = {}
    
    for stat in stats_list:
        # Recherche de la colonne courte (ex: airline_rolling_7D_mean)
        col_short = [c for c in df.columns if short_term_pattern in c and c.endswith(f'_{stat}')]
        # Recherche de la colonne longue (ex: airline_rolling_30D_mean)
        col_long = [c for c in df.columns if long_term_pattern in c and c.endswith(f'_{stat}')]
        
        # On itère sur les correspondances trouvées (par groupe de colonnes)
        for s_col in col_short:
            # On cherche la version "longue" qui correspond au même groupe (airline, route, etc.)
            # On remplace simplement le pattern court par le long dans le nom
            l_col = s_col.replace(short_term_pattern, long_term_pattern)
            
            if l_col in df.columns:
                new_col_name = f"MOM_{s_col}_{suffix}_{long_term_pattern}"
                
                from typing import cast
                import numpy as np
                import numpy.typing as npt

                # 1. Extraction et Cast explicite vers un type NDArray numérique
                # On dit à Pylance : "Je te garantis que c'est un tableau NumPy de float32"
                l_values = cast(npt.NDArray[np.float32], df[l_col].values.astype(np.float32))
                s_values = cast(npt.NDArray[np.float32], df[s_col].values.astype(np.float32))

                # 2. Maintenant, l'opérateur ">" est parfaitement supporté par le type NDArray
                condition_mask = l_values > 0

                # 3. Calcul du ratio
                new_momentum[new_col_name] = np.divide(
                    s_values, 
                    l_values, 
                    out=np.ones_like(s_values), 
                    where=condition_mask
                )

    # 2. Concaténation pour éviter le PerformanceWarning de fragmentation
    if new_momentum:
        new_df = pd.DataFrame(new_momentum, index=df.index)
        df = pd.concat([df, new_df], axis=1)
    else:
        print(f"Aucune correspondance trouvée entre {short_term_pattern} et {long_term_pattern}")
        
    return df



def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to add new features. 
    Note: Order is critical (Momentum must come after Rolling/Lag).
    """
    # 0. SÉCURITÉ : Supprimer les index dupliqués potentiels avant de commencer
    df = df.reset_index(drop=True)

    ### 1. Date related features
    df = date_columns_creation(df=df)

    for col in COLUMN_LIST_BASE:
        ### 2. Lag features (Saisonnalité pure)
        df = add_lag_features(df, group_cols=[col], lags=CUSTOM_LAGS)

        ### 3. Rolling features (Moyennes mobiles récentes)
        df = add_rolling_features(df, group_cols=[col], windows=ROLLING_CONFIG)

        ### 4. Lagged rolling features (Moyennes mobiles historiques)
        for config_name, config in ROLLING_LAGS_CONFIG.items():
            df = add_lagged_rolling_features(
                df, 
                group_cols=[col], 
                lag=config["lag"], 
                window=config["window"], 
                new_col_name=f"{col}_{config_name}"
            )

        ### 5. Trend features (Ratios internes au rolling actuel)
        for short_win, long_win in TREND_CONFIG:
            df = add_trend_features(df, group_cols=[col], short_win=short_win, long_win=long_win)

        # --- NOUVEAU : MOMENTUM FEATURES ---
        # On compare les horizons temporels entre eux pour chaque colonne de base
        
        # A. Dynamique de court terme (Ex: accélération de la semaine vs le mois)
        df = add_momentum_features(df, 
                                   short_term_pattern='rolling_week', 
                                   long_term_pattern='rolling_month', 
                                   suffix="accel")

        df = add_momentum_features(df, 
                                   short_term_pattern='rolling_month', 
                                   long_term_pattern='rolling_quarter', 
                                   suffix="month_ratio")
        
        df = add_momentum_features(df, 
                                   short_term_pattern='rolling_quarter', 
                                   long_term_pattern='rolling_year', 
                                   suffix="season_ratio")

        df = add_momentum_features(df, 
                           short_term_pattern='lag91_win20', 
                           long_term_pattern='lag365_win28', 
                           suffix="year_over_year")
        
        df = add_momentum_features(df, 
                           short_term_pattern='lag30_win14', 
                           long_term_pattern='lag91_win20', 
                           suffix="short_vs_mid_hist")

    ### 6. Interaction features (Poids capacité)
    # On multiplie NbOfSeats par les moyennes (incluant les nouvelles colonnes de momentum)
    df = add_interaction_features(df, base_col="NbOfSeats", feature_pattern="_mean")

    return df





### Test 
if __name__ == "__main__":
    df = pd.read_csv("data/main.csv")
    df = add_features(df)
    print(df.columns)
    print(df.shape)
    df.to_csv("data/main_preprocessed_new.csv", index=False)
