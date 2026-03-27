### Imports
import airportsdata
import pandas as pd
from datetime import date
import os
import requests
from pathlib import Path
from pandasql import sqldf
import holidays
from utils.holidays.france_pipeline import pipeline_france
from utils.holidays.international_scholar_holidays import add_school_holiday_international


### Global variable
data_folder = os.path.join(Path(__file__).parent.parent.parent, "data")
filename = os.path.join(data_folder, "main.csv")
CODE_LIST_FR = ["FR", "LF", "GP", "MQ", "RE"] # metroplitan codes + islands and extra-marine territories

# data import and light preprocessing
data = pd.read_csv(filename, encoding="utf-8")
data = data[data['IdBusinessUnitType'] == 1] # Limit to commercial (passenger) flight for prediction
pysqldf = lambda q: sqldf(q, globals())



### 1. get the oaci codes from the 'AirportPrevious' feature
# Query the data
def get_main_simplify(data:pd.DataFrame):
    res =  pysqldf("""
        SELECT AirportPrevious, LTScheduledDatetime
        FROM data;
    """)
    if type(res) is pd.DataFrame :
        AirportPrevious = list(res["AirportPrevious"])
        LTScheduledDatetime = pd.to_datetime(list(res["LTScheduledDatetime"]))
    else : 
        raise ValueError("Invalid query.")

    main_simplify_df = pd.DataFrame({"LTScheduledDatetime" : LTScheduledDatetime, "LTScheduledDatetime-day" : LTScheduledDatetime.date, "AirportPrevious":AirportPrevious})

    return main_simplify_df



def get_code_infos(main_simplify_df : pd.DataFrame):
    code_list_unique = list(set(main_simplify_df["AirportPrevious"]))
    # Match with external data source - use of aiportsdata.
    # OACI and IATA codes are used in our dataframe => need of the 2 to cover a larger number of airport codes.
    icao_airports = airportsdata.load('ICAO')
    iata_airports = airportsdata.load('IATA')


    undetermined_codes = []
    mapping_code = {"AirportPrevious":[], "OACI_code":[], "country":[], 'city':[]}#keys : new_code, values : AirportPrevious code

    for code in code_list_unique:
        if len(code) == 3:  # Tentative de conversion IATA -> OACI
            data = iata_airports.get(code)
            if data and data.get('icao'):
                mapping_code["AirportPrevious"].append(code)
                mapping_code["OACI_code"].append(data['icao'])
                mapping_code["country"].append(data['country'])
                mapping_code["city"].append(data['city'])
            else:
                # only 2 airport without mathing code into the database are an issue 
                # why? The codes are obsolete 'KIV' = Chișinău Airport (Moldavia) ; 'SXF' = Berlin-Schönefeld Airport (Germany)
                # => find manually the needed information for those airports.
                # What interest us is only the country of those airports (city if in france).
                undetermined_codes.append(code)
        else: # Already in OACI format (4 letters)
            data = icao_airports.get(code)
            if data and data.get('icao'):
                mapping_code["AirportPrevious"].append(code)
                mapping_code["OACI_code"].append(code)
                mapping_code["country"].append(data['country'])
                mapping_code["city"].append(data['city'])
            else:
                undetermined_codes.append(code)
    
    return pd.DataFrame(mapping_code)




def add_public_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an 'IsPublicHoliday' column (0 or 1) to a DataFrame.

    Parameters:
        df (pd.DataFrame): Must contain:
            - 'LTScheduledDatetime-day'         : datetime or date column (daily granularity)
            - 'country' : ISO 3166-1 alpha-2 country code (e.g. 'US', 'FR')

    Returns:
        pd.DataFrame: Same DataFrame with an added 'IsPublicHoliday' column.
    """
    df = df.copy()
    df["LTScheduledDatetime-day"] = pd.to_datetime(df["LTScheduledDatetime-day"])

    # Cache holiday objects per (country, year) to avoid redundant lookups
    holiday_cache = {}

    def is_holiday(row):
        country = row["country"]
        date = row["LTScheduledDatetime-day"].date()
        year = date.year
        key = (country, year)

        if key not in holiday_cache:
            try:
                holiday_cache[key] = holidays.country_holidays(country, years=year)
            except (KeyError, NotImplementedError):
                # Country not supported by the library
                holiday_cache[key] = {}

        return 1 if date in holiday_cache[key] else 0

    df["IsPublicHoliday"] = df.apply(is_holiday, axis=1)
    return df






if __name__ == '__main__' :
    df_main_simplify = get_main_simplify(data=data)
    df_code_infos = get_code_infos(main_simplify_df=df_main_simplify)

    df_merge = pd.merge(df_main_simplify, df_code_infos, on="AirportPrevious")
    df_public_holidays = add_public_holidays(df=df_merge)


    # To keep only French airports
    code_list_fr = CODE_LIST_FR
    data_france = df_public_holidays[df_public_holidays['country'].isin(code_list_fr)]
    data_international = df_public_holidays[~df_public_holidays['country'].isin(code_list_fr)]

    df_scholar_holdidays_international = add_school_holiday_international(df=data_international)

    df_scholar_holdidays_france = pipeline_france(data=data_france)

    print(df_scholar_holdidays_france)
    print(df_scholar_holdidays_international)
    # print(df_public_holidays)
    df_scholar_holdidays_france.to_csv('df_france.csv', encoding='utf-8')
    df_scholar_holdidays_international.to_csv('df_international.csv', encoding='utf-8')


