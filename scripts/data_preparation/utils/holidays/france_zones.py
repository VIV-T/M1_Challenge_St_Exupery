import requests
import json
import time
from utils.holidays.env_variables import FR_MAPPING_ACADEMIE

# --- CONFIGURATION ---

API_URL = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/records"
CACHE_FILE = 'cache_vacances.json'



# --- Cache management functions ---
# To avoid to recall the API for each run - can be heavy for no reason.

def load_cache():
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)



# --- CLEANING and API CALL ---

def get_clean_name(raw_name):
    # We only keep the first name -> enought to get the holiday zone.
    name = raw_name.split('/')[0].strip()
    
    # 2. Hard code for exceptions
    # /!\ Code weakness
    if "Figari" in name: return "Figari"
    if "Pointe" in name: return "Pointe-a-Pitre"
    if "La Rochelle" in name: return "La Rochelle"
    
    # Normalisation of 'St ' to 'Saint-'
    if name.startswith("St "):
        name = name.replace("St ", "Saint-")
        
    # Main cleaning case
    clean_city = name.split(' ')[0].split('-')[0].strip()
    
    # 'Saint' exception: get the full name to avoid bug.
    if clean_city == "Saint":
        clean_city = name.split(' ')[0].strip()
        
    return clean_city


def fetch_holiday_status(raw_city, cache):
    clean_name = get_clean_name(raw_city)
    
    # Cache verification and redirection
    target = FR_MAPPING_ACADEMIE.get(clean_name, clean_name)
    if target in cache:
        return cache[target]

    # API call 
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
            # Zone managment or localization for island and non-metropolitan territory
            zone = res.get('zones') if res.get('zones') else res.get('location')
            status = {
                "zone": zone,
                "is_holiday": "OUI" if res['start_date'] <= res['end_date'] else "NON"
            }
        else:
            status = {"zone": "Non trouvée", "is_holiday": "NON"}
    except Exception as e:
        status = {"zone": "Erreur API", "is_holiday": "Inconnu"}

    # Save the cache
    cache[target] = status
    time.sleep(0.05) 
    return status



# --- Main Execution ---

def get_zone_airports(data_list):
    cache = load_cache()
    results_zone = []
    
    for city in data_list:
        info = fetch_holiday_status(city, cache)
        results_zone.append(info['zone'])

    save_cache(cache)
    return results_zone


