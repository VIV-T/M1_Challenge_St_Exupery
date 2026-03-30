
import json
import numpy as np
import pandas as pd


def add_route_feature(data):   
        origin = data['AirportOrigin'].astype(str)
        # Construction de la Route selon la Direction
        # 1 = Arrivée (Provenance -> LYON)
        # 0 = Départ  (LYON -> Destination)
        data['Route'] = np.where(
            data['Direction'] == 1,
            origin + "_LYS", # Arrivée
            "LYS_" + origin  # Départ
        )
        return data
    



def add_route_index(data):
        unique_routes = sorted(data['Route'].unique())
        route_to_id = {route: i for i, route in enumerate(unique_routes)}
        route_to_id["UNKNOWN"] = len(unique_routes)
        data['IdRoute'] = data['Route'].map(route_to_id)

        with open("mappings_routes.json", 'w', encoding='utf-8') as f:
            json.dump(route_to_id, f, indent=4, ensure_ascii=False)
        
        return data



def add_optimized_historical_features(df):
    """
    Calcule les moyennes historiques (1W, 1M, 3M, 1Y) de manière vectorisée.
    """
    # 1. Préparation et tri (CRUCIAL pour merge_asof)
    df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime'])
    df = df.sort_values(['IdRoute', 'LTScheduledDatetime'])
    
    # Base de données historique (on ne garde que les vols passés avec un score connu)
    hist = df[['LTScheduledDatetime', 'IdRoute', 'OccupancyRate']].dropna().copy()
    
    def get_past_window_avg(main_df, days_offset, window_size_days):
        """
        Trouve la moyenne des vols autour de (T - days_offset) +/- window_size_days
        """
        lookup = main_df[['LTScheduledDatetime', 'IdRoute']].copy()
        lookup['target_date'] = lookup['LTScheduledDatetime'] - pd.Timedelta(days=days_offset)
        
        # On pré-calcule une moyenne glissante sur l'historique pour lisser la fenêtre
        # window = window_size_days * 2 (pour couvrir le +/-)
        hist_rolling = hist.set_index('LTScheduledDatetime').groupby('IdRoute')['OccupancyRate'].rolling(
            window=pd.Timedelta(days=window_size_days * 2), 
            center=True, 
            min_periods=1
        ).mean().reset_index()
        
        # Jointure temporelle ultra-rapide
        merged = pd.merge_asof(
            lookup.sort_values('target_date'), 
            hist_rolling.sort_values('LTScheduledDatetime'),
            left_on='target_date',
            right_on='LTScheduledDatetime',
            by='IdRoute',
            direction='nearest', # Cherche le vol le plus proche de la cible
            tolerance=pd.Timedelta(days=window_size_days) # Ne prend rien si trop loin
        )
        
        # Retourne la colonne alignée sur l'index d'origine
        return merged.set_index(lookup.index).sort_index()['OccupancyRate']

    # --- CONFIGURATION DES FENÊTRES ---
    # (Label, Offset en jours, Demi-fenêtre de lissage en jours)
    tasks = [
        ('AvgOcc_1W', 7, 3),    # J-7  +/- 3 jours
        ('AvgOcc_1M', 30, 7),   # J-30 +/- 7 jours
        ('AvgOcc_3M', 90, 10),  # J-90 +/- 10 jours
        ('AvgOcc_6M', 180, 10), # J-180 +/- 10 jours
        ('AvgOcc_1Y', 365, 15)  # J-365 +/- 15 jours (Remplace OccupancyPreviousYears)
    ]

    for col_name, offset, win in tasks:
        print(f"── Calcul de {col_name}...")
        df[col_name] = get_past_window_avg(df, offset, win)

    # --- GESTION DU COLD START & FALLBACK ---
    print("── Finalisation et remplissage des NaN...")
    
    # On crée une hiérarchie de secours pour éviter les trous (NaN)
    # 1. Si 1W est vide, on tente 1M
    # 2. Si 1M est vide, on tente 1Y
    # 3. Si tout est vide, on prend la moyenne cumulative de la route
    
    route_mean_progressive = df.groupby('IdRoute')['OccupancyRate'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    for col_name, _, _ in tasks:
        # Remplissage par la moyenne progressive de la route
        df[col_name] = df[col_name].fillna(route_mean_progressive)
        # Sécurité ultime par la moyenne globale du dataset
        df[col_name] = df[col_name].fillna(df['OccupancyRate'].mean())

    return df





def add_optimized_flight_history(df):
    """
    Calcule les moyennes historiques basées sur le OperatorFlightNumber
    plutôt que sur la route globale.
    """
    # 1. Préparation : s'assurer que les dates sont au bon format et trier
    df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime'])
    # On trie par vol ET par date pour que merge_asof fonctionne
    df = df.sort_values(['OperatorFlightNumber', 'LTScheduledDatetime'])
    
    # Base de données historique (uniquement vols passés connus)
    hist = df[['LTScheduledDatetime', 'OperatorFlightNumber', 'OccupancyRate']].dropna().copy()
    
    def get_flight_window_avg(main_df, days_offset, window_size_days):
        lookup = main_df[['LTScheduledDatetime', 'OperatorFlightNumber']].copy()
        lookup['target_date'] = lookup['LTScheduledDatetime'] - pd.Timedelta(days=days_offset)
        
        # Moyenne glissante sur l'historique du vol spécifique
        hist_rolling = hist.set_index('LTScheduledDatetime').groupby('OperatorFlightNumber')['OccupancyRate'].rolling(
            window=pd.Timedelta(days=window_size_days * 2), 
            center=True, 
            min_periods=1
        ).mean().reset_index()
        
        # Jointure temporelle par numéro de vol
        merged = pd.merge_asof(
            lookup.sort_values('target_date'), 
            hist_rolling.sort_values('LTScheduledDatetime'),
            left_on='target_date',
            right_on='LTScheduledDatetime',
            by='OperatorFlightNumber',
            direction='nearest',
            tolerance=pd.Timedelta(days=window_size_days)
        )
        
        return merged.set_index(lookup.index).sort_index()['OccupancyRate']

    # --- CALCUL DES FENÊTRES ---
    tasks = [
        ('FlightOcc_1W', 7, 3),    # Le même vol la semaine dernière
        ('FlightOcc_1M', 30, 7),   # Le même vol le mois dernier
        ('FlightOcc_3M', 90, 10),  # Le même vol il y a 3 mois
        ('FlightOcc_6M', 180, 15), # Le même vol il y a 6 mois
        ('FlightOcc_1Y', 365, 15)  # Le même vol l'an dernier (N-1)
    ]

    for col_name, offset, win in tasks:
        print(f"── Calcul de {col_name}...")
        df[col_name] = get_flight_window_avg(df, offset, win)

    # --- GESTION DU COLD START (Fallback) ---
    print("── Remplissage des trous (Fallback Route/Global)...")
    
    # Sécurité 1 : Si le numéro de vol est nouveau, on prend la moyenne de la ROUTE
    # (On suppose que tu as gardé ta colonne 'Route' créée précédemment)
    route_mean = df.groupby('Route')['OccupancyRate'].transform(lambda x: x.expanding().mean().shift(1))
    
    for col_name, _, _ in tasks:
        df[col_name] = df[col_name].fillna(route_mean)
        # Sécurité 2 : Moyenne globale
        df[col_name] = df[col_name].fillna(df['OccupancyRate'].mean())

    return df