"""
update_external_data.py — Universal API Signal Sync for Project Saint-Exupéry.

This script manages the dynamic retrieval of world weather and national
educational calendars. It populates the local 'externals/' cache to
ensure the pipeline remains maintainable and accurate for future years.
"""
import pandas as pd
import subprocess
import json
import airportsdata
import os
import time
from datetime import datetime, timedelta

def run_curl(url):
    """Bypasses Python SSL bottlenecks using system-level curl."""
    try:
        result = subprocess.run(['curl', '-s', url], capture_output=True, text=True, timeout=15)
        return result.stdout if result.returncode == 0 else None
    except Exception as e:
        print(f"Error fetching {url[:50]}...: {e}")
        return None

def fetch_school_holidays():
    """Syncs the 3-Zone French National Education calendar via API."""
    print("Syncing French School Holidays (Zones A, B, C)...")
    all_results = []
    
    # We fetch each zone separately to ensure 100% coverage and avoid API pagination limits
    for zone in ["Zone A", "Zone B", "Zone C"]:
        print(f"  Fetching {zone}...")
        encoded_zone = zone.replace(" ", "%20")
        url = f"https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-calendrier-scolaire/records?limit=100&where=zones%3D%22{encoded_zone}%22&refine=annee_scolaire%3A2023-2024&refine=annee_scolaire%3A2024-2025&refine=annee_scolaire%3A2025-2026"
        raw = run_curl(url)
        if raw:
            try:
                data = json.loads(raw)
                results = [{"zone": r.get('zones'), "start": r['start_date'][:10], "end": r['end_date'][:10]}
                           for r in data.get('results', []) if r.get('start_date') and r.get('end_date')]
                all_results.extend(results)
            except:
                print(f"  Failed to parse {zone} JSON.")
                
    if all_results:
        pd.DataFrame(all_results).drop_duplicates().to_csv('externals/school_holidays.csv', index=False)
        print(f"  Saved {len(all_results)} school holiday blocks to externals/school_holidays.csv")

def fetch_hub_weather():
    """Syncs bidirectional weather for the top 50 international hubs."""
    print("Syncing Bidirectional Weather (Top 50 Hubs)...")
    output_file = 'externals/weather_hubs.csv'
    airports = airportsdata.load('IATA')
    targets = ['LYS', 'CDG', 'BOD', 'TLS', 'NTE', 'AMS', 'FRA', 'MAD', 'LIS', 'TUN', 'ALG', 'MUC', 'LGW', 'LHR', 'OPO', 'BCN', 'NCE', 'RAK', 'BRU', 'IST', 'BIQ', 'FCO', 'CMN', 'SAW', 'BES', 'MRS', 'RNS', 'CFR', 'PMI', 'DJE', 'SXB', 'QSF', 'LIG', 'ORN', 'PUF', 'DUB', 'MIR', 'BIA', 'AJA', 'VCE', 'MXP', 'LTN', 'ATH', 'YUL', 'DXB', 'VIE', 'HER', 'CZL', 'FAO', 'CGN', 'AGP']
    
    fetched_hubs = pd.read_csv(output_file)['iata'].unique().tolist() if os.path.exists(output_file) else []
    end_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')

    for iata in targets:
        if iata in fetched_hubs or iata not in airports: continue
        apt = airports[iata]
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={apt['lat']}&longitude={apt['lon']}&start_date=2023-01-01&end_date={end_date}&daily=temperature_2m_max,precipitation_sum&timezone=UTC"
        raw = run_curl(url)
        if raw:
            data = json.loads(raw)
            if 'daily' in data:
                df_w = pd.DataFrame({"date": data['daily']['time'], "temp_max": data['daily']['temperature_2m_max'],
                                    "precip": data['daily']['precipitation_sum'], "iata": iata})
                df_w.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
                time.sleep(0.1)

if __name__ == "__main__":
    os.makedirs('externals', exist_ok=True)
    fetch_school_holidays()
    fetch_hub_weather()
