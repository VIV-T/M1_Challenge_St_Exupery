import requests
import pandas as pd
import time
from scripts.data_preparation.utils.holidays.env_variables import START_DATE, END_DATE, FR_ZONES

# --- CONFIGURATION ---
API_URL = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/records"


def get_calendar_df():
    all_records = []
    
    for zone in FR_ZONES:        
        # Query to get all the holiday period on a date range.
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
                    # One record per holiday
                    record = {
                        "Zone_Ou_Region": zone,
                        "Description": res.get('description'),
                        "Date_Debut": res.get('start_date').split('T')[0],
                        "Date_Fin": res.get('end_date').split('T')[0],
                        "Annee_Scolaire": res.get('annee_scolaire'),
                        "Population": res.get('population')
                    }
                    all_records.append(record)
            
            # Time break - Smooth API call
            time.sleep(0.1)
            
        except Exception as e:
            raise ValueError(f"Error for zone {zone}: {e}")

    # DataFrame creation
    df = pd.DataFrame(all_records)
    
    return df





def get_calendar_scholar_holidays():
    # Initialization & preprocessing
    df_periodes = get_calendar_df()
    df_periodes['Date_Debut'] = pd.to_datetime(df_periodes['Date_Debut'])
    df_periodes['Date_Fin'] = pd.to_datetime(df_periodes['Date_Fin'])

    dates_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    df_daily = pd.DataFrame({'date': dates_range})

    for zone in FR_ZONES:
        col_name = f"{zone.replace(' ', '_')}"
        df_daily[col_name] = 0

        # Binary filling (0/1) if the date is in a holiday period.
        filtre_zone = df_periodes[df_periodes['Zone_Ou_Region'] == zone]
        
        for _, row in filtre_zone.iterrows():
            mask = (df_daily['date'] >= row['Date_Debut']) & (df_daily['date'] <= row['Date_Fin'])
            df_daily.loc[mask, col_name] = 1
    
    
    return df_daily
