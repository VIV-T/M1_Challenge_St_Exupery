import requests
import pandas as pd
import time

# --- CONFIGURATION ---
API_URL = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/records"
START_DATE = "2023-01-01"
END_DATE = "2027-12-31"

ZONES = [
    "Zone A", "Zone B", "Zone C", 
    "Corse", "Guadeloupe", "Réunion", "Martinique", "Guyane", "Mayotte"
]

def get_calendar_df():
    all_records = []
    
    for zone in ZONES:        
        # Requête pour récupérer toutes les vacances sur la période
        params = {
            "where": f'(zones = "{zone}" OR location = "{zone}") '
                     f'AND end_date >= "{START_DATE}" '
                     f'AND start_date <= "{END_DATE}" '
                     f'AND (description LIKE "Vacances" OR description LIKE "Été")',
            "order_by": "start_date ASC",
            "limit": 100 
        }
        
        try:
            response = requests.get(API_URL, params=params, timeout=15)
            data = response.json()
            
            if data.get('total_count', 0) > 0:
                for res in data['results']:
                    # On crée une ligne par période de vacances
                    record = {
                        "Zone_Ou_Region": zone,
                        "Description": res.get('description'),
                        "Date_Debut": res.get('start_date').split('T')[0],
                        "Date_Fin": res.get('end_date').split('T')[0],
                        "Annee_Scolaire": res.get('annee_scolaire'),
                        "Population": res.get('population') # Utile pour distinguer Enseignants/Élèves
                    }
                    all_records.append(record)
            
            # Petite pause pour respecter l'API
            time.sleep(0.1)
            
        except Exception as e:
            raise ValueError(f"Error for zone {zone}: {e}")

    # DataFrame creation
    df = pd.DataFrame(all_records)
    
    return df





def get_calendar_scholar_holidays():
    df_periodes = get_calendar_df()
    df_periodes['Date_Debut'] = pd.to_datetime(df_periodes['Date_Debut'])
    df_periodes['Date_Fin'] = pd.to_datetime(df_periodes['Date_Fin'])

    dates_range = pd.date_range(start="2023-01-01", end="2027-12-31", freq='D')
    df_daily = pd.DataFrame({'date': dates_range})

    for zone in ZONES:
        col_name = f"{zone.replace(' ', '_')}"
        df_daily[col_name] = 0

        # 4. Remplissage binaire (1 si la date est comprise entre Debut et Fin)
        # On filtre les périodes correspondant à la zone actuelle
        filtre_zone = df_periodes[df_periodes['Zone_Ou_Region'] == zone]
        
        for _, row in filtre_zone.iterrows():
            mask = (df_daily['date'] >= row['Date_Debut']) & (df_daily['date'] <= row['Date_Fin'])
            df_daily.loc[mask, col_name] = 1
    
    
    return df_daily
