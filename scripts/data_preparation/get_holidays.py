import requests
import pandas as pd

def get_calendar_dataframe() -> pd.DataFrame:
    # 1. Scholar calendar
    url_v = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/records"
    params_v = {
        "where": "(location='Lyon' OR location='Éducation nationale') AND end_date >= '2023-01-01'",
        "order_by": "end_date ASC",
        "limit": 100
    }

    # 2. Public holiday
    url_f = "https://calendrier.api.gouv.fr/jours-feries/metropole.json"

    try:
        # API call
        res_v = requests.get(url_v, params=params_v).json().get('results', [])
        res_f = requests.get(url_f).json()

        data_list = []

        # Scholar calendar Extraction
        for v in res_v:
            data_list.append({
                "start_date": v['start_date'][:10],
                "end_date": v['end_date'][:10],
                "name": v['description'],
                "type": "VACANCES"
            })

        # Public Holidays Extraction
        for d, nom in res_f.items():
            if d >= "2023-01-01":
                data_list.append({
                    "start_date": d,
                    "end_date": d, # Pour un jour férié, début = fin
                    "name": nom,
                    "type": "JOUR FÉRIÉ"
                })
        df = pd.DataFrame(data_list)

        # Convert to datetime
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

        # Sort by date
        df_filtre = df.sort_values(by='start_date')

        return df_filtre

    except Exception as e:
        print(f"Error : {e}")
        return pd.DataFrame()

# Exécution et affichage
df_resultat = get_calendar_dataframe()
df_resultat.to_csv("data/holidays.csv", encoding='utf-8')

if df_resultat is not None:
    print(f"Number of results: {len(df_resultat)}")
    print(df_resultat.head(50)) # Affiche les 20 premières lignes