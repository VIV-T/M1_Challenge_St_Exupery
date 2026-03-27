import requests
import json
import time
from utils.holidays.env_variables import FR_MAPPING_ACADEMIE

# --- CONFIGURATION ---
API_URL = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/records"
CACHE_FILE = 'cache_vacances.json'


# --- FONCTIONS DE GESTION DU CACHE ---
def load_cache():
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)

# --- NETTOYAGE ET APPEL API ---
def get_clean_name(raw_name):
    # 1. On isole la partie avant le '/'
    name = raw_name.split('/')[0].strip()
    
    # 2. Cas prioritaires (Hardcoding pour les noms complexes)
    if "Figari" in name: return "Figari"
    if "Pointe" in name: return "Pointe-a-Pitre"
    if "La Rochelle" in name: return "La Rochelle"
    
    # 3. Normalisation des 'St ' en 'Saint-'
    if name.startswith("St "):
        name = name.replace("St ", "Saint-")
        
    # 4. Pour les autres, on ne garde que le premier mot AVANT un espace ou un tiret
    # sauf si c'est un nom composé déjà mappé (ex: Saint-Louis)
    clean_city = name.split(' ')[0].split('-')[0].strip()
    
    # Si le nom nettoyé est juste "Saint", on reprend le nom complet avant le '/'
    # pour éviter le bug Saint-Pierre-et-Miquelon
    if clean_city == "Saint":
        clean_city = name.split(' ')[0].strip()
        
    return clean_city


def fetch_holiday_status(raw_city, cache):
    clean_name = get_clean_name(raw_city)
    
    # Vérification Redirection + Cache
    target = FR_MAPPING_ACADEMIE.get(clean_name, clean_name)
    if target in cache:
        return cache[target]

    # Requête API Strict (=) pour éviter les faux positifs (ex: Saint-Pierre-et-Miquelon)
    params = {
        "where": f'(location = "{target}" OR zones = "{target}")',
        "order_by": "start_date ASC",
        "limit": 1
    }

    try:
        response = requests.get(API_URL, params=params, timeout=5)
        data = response.json()
        
        if data.get('total_count', 0) > 0:
            res = data['results'][0]
            # Priorité à la Zone (A, B, C), sinon la Localisation (Corse, DOM)
            zone = res.get('zones') if res.get('zones') else res.get('location')
            status = {
                "zone": zone,
                "is_holiday": "OUI" if res['start_date'] <= res['end_date'] else "NON"
            }
        else:
            status = {"zone": "Non trouvée", "is_holiday": "NON"}
    except Exception as e:
        status = {"zone": "Erreur API", "is_holiday": "Inconnu"}

    # Stockage et respect du serveur
    cache[target] = status
    time.sleep(0.05) 
    return status

# --- EXÉCUTION PRINCIPALE ---
def get_zone_airports(data_list):
    cache = load_cache()
    results_zone = []
    
    for city in data_list:
        info = fetch_holiday_status(city, cache)
        results_zone.append(info['zone'])

    save_cache(cache)
    return results_zone


