"""
Find the holidays period of each airport countries (with differents areas A/B/C for france airports)

"""
### Imports
import airportsdata
import pandas as pd
from datetime import date
import os
from pathlib import Path
from pandasql import sqldf
from utils.holidays_france_zones import get_zone_airports


### Global variable
data_folder = os.path.join(Path(__file__).parent.parent.parent, "data")
filename = os.path.join(data_folder, "main.csv")

# data import and light preprocessing
data = pd.read_csv(filename, encoding="utf-8")
data = data[data['IdBusinessUnitType'] == 1] # Limit to commercial (passenger) flight for prediction
pysqldf = lambda q: sqldf(q, globals())


### 1. get the oaci codes from the 'AirportPrevious' feature
# Query the data
res =  pysqldf("""
    SELECT AirportPrevious
    FROM data;
""")
if type(res) is pd.DataFrame :
    oaci_codes = res["AirportPrevious"]
else : 
    raise ValueError("Invalid query.") 
list_oaci_code_main = list(set(oaci_codes))


# Match with external data source - use of aiportsdata.
# OACI and IATA codes are used in our dataframe => need of the 2 to cover a larger number of airport codes.
icao_airports = airportsdata.load('ICAO')
iata_airports = airportsdata.load('IATA')

oaci_list = []
undetermined_codes = []
mapping_code = dict() #keys : new_code, values : AirportPrevious code

for code in list_oaci_code_main:
    if len(code) == 3:  # Tentative de conversion IATA -> OACI
        data = iata_airports.get(code)
        if data and data.get('icao'):
            oaci_list.append(data['icao'])
            mapping_code[data['icao']] = code
        else:
            # only 2 airport without mathing code into the database are an issue 
            # why? The codes are obsolete 'KIV' = Chișinău Airport (Moldavia) ; 'SXF' = Berlin-Schönefeld Airport (Germany)
            # => find manually the needed information for those airports.
            # What interest us is only the country of those airports (city if in france).
            undetermined_codes.append(code)
    else: # Already in OACI format (4 letters)
        oaci_list.append(code)
        mapping_code[code] = code

# Keep unique codes
list_oaci_uniques = list(set(oaci_list))

## Print the results 
# print(f"Codes from source : {len(list_oaci_code_main)}")
# print(f"Unique codes identified : {len(list_oaci_uniques)}")
# print(f"Undetermined codes : {undetermined_codes}")
# print(list_oaci_uniques)
# print("\n\n")


#### Pipeline division ####
founded_coutries = set()

dict_international = {
    "code_AirportPrevious" : [],
    "new_code" : [],
    "country_code" : [], 
    "country_name" : []
}

list_code_france = ["FR", "LF", "GP", "MQ", "RE"] # metroplitan codes + islands and extra-marine territories
dict_france = {
    "code_AirportPrevious" : [],
    "new_code" : [],
    "country_code" : [], 
    "city" : [], 
    "holiday_zone": []
}


for code in list_oaci_uniques:
    info = icao_airports.get(code)
    if info:
        founded_coutries.add(info['country'])
        if info['country'] in list_code_france : 
            # france pipeline
            dict_france['code_AirportPrevious'].append(mapping_code[code])
            dict_france['new_code'].append(code)
            dict_france['country_code'].append(info['country'])
            dict_france["city"].append(info["city"])

        else : 
            # international pipeline
            # date needed
            dict_international['code_AirportPrevious'].append(mapping_code[code])
            dict_international['country_code'].append(info['country'])



    else:
        # Debug : voir quels codes ne sont pas trouvés
        print(f"Code non trouvé : {code}")


### France Pipeline - next steps.
# Use the additional utils function to find the holiday zone depending of the airport city name. 
dict_france["holiday_zone"] = get_zone_airports(dict_france["city"])
# Save res as df (France) => preparation for merging
df_france = pd.DataFrame(dict_france)
df_france.to_csv("df_2.csv")

print(df_france)

print('\n\n\n\n')




