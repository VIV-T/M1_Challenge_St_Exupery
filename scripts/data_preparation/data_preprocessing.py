import pandas as pd
import numpy as np
import os
from pathlib import Path

import utils.weather as wthr


# ------------- Global variables -------------
data_folder = os.path.join(Path(__file__).parent.parent.parent, "data")
main_old_filename = os.path.join(data_folder, "main.csv")
main_new_filename = os.path.join(data_folder, "main_preprocessed.csv")

holidays_old_filename = os.path.join(data_folder, "holidays.csv")
holidays_new_filename = os.path.join(data_folder, "holidays_preprocessed.csv")

weather_old_filename = os.path.join(data_folder, "weather.csv")
weather_new_filename = os.path.join(data_folder,"weather_preprocessed.csv")




# -------------  Utils -------------
### Holidays data ###
def pre_process_holidays(data_old_filename, data_new_filename)-> None:
    """
    Function used to pre-processed the holidays data get from the data.gouv.fr API.

    Params:
        - data_old_filename: the filename to load the raw dataset.
        - data_new_filename: the filename to save the pre-processed dataset.
    """
    data = pd.read_csv(data_old_filename, encoding='utf-8')

    data["start_date"] = pd.to_datetime(data["start_date"])
    data["end_date"] = pd.to_datetime(data["end_date"])

    all_dates = pd.date_range(start='2023-01-01', end='2026-03-01', freq='D')
    df_final = pd.DataFrame({'holidays_date': all_dates})


    def find_period(current_date):
        # Looking if the date is between start et end in our dataset
        match = data[(current_date >= data['start_date']) & 
                        (current_date <= data['end_date'])]
        
        if not match.empty:
            return match['holidays_name'].values[0]
        return None

    # Apply this function each line of the calendar
    df_final['holidays_name'] = df_final['holidays_date'].apply(find_period)
    df_final['holiday'] = df_final['holidays_name'].notnull().astype(int)
    rmv_col_list = ['holidays_name']
    df_final = df_final.drop(columns=rmv_col_list)
    
    # Save the new df
    df_final.to_csv(data_new_filename, encoding='utf-8', index=False)



### Weather data ###
def pre_process_weather(data_old_filename:str, data_new_filename:str):
    """
    Function used to pre-processed the main dataset get from the Bigquery table.

    Params:
        - data_old_filename: the filename to load the raw dataset.
        - data_new_filename: the filename to save the pre-processed dataset.
    """
    df = wthr.remove_missing_values(data_old_filename)
    df_1 = wthr.remove_bad_quality(df)
    df_2 = wthr.remove_unnecesserary_columns(df_1)
    df_2.to_csv(data_new_filename, encoding="utf-8", index=False)
    print(f"Shape of the final weather dataset : {df_2.shape}")
    print(f"Preprocessed weather dataset saved in {data_new_filename}")




### Main data ###
def pre_process_main(data_old_filename, data_new_filename)-> None:
    """
    Function used to pre-processed the main dataset get from the Bigquery table.

    Params:
        - data_old_filename: the filename to load the raw dataset.
        - data_new_filename: the filename to save the pre-processed dataset.
    """
    data = pd.read_csv(data_old_filename, encoding='utf-8')

    # to datetime conversion
    data['LTScheduledDatetime'] = pd.to_datetime(data['LTScheduledDatetime'])

    # Creation of the index columns used for external merge (holidays and weather data) 
    data["LTScheduledDatetime-day"] = data['LTScheduledDatetime'].dt.date
    data["LTScheduledDatetime-hour-code"] = data['LTScheduledDatetime'].dt.strftime('%Y%m%d%H')

    ### Additional transformations and pre-processing to add HERE...
    # flight_with_pax - Oui -> 1 , Non -> 0
    data['flight_with_pax'] = data['flight_with_pax'].map({'Oui': 1, 'Non': 0})

    # Direction - Arrivée -> 1 , Départ -> 0
    data['Direction'] = data['Direction'].map({'Arrivée': 1, 'Départ': 0})


    # SysStopover (destination) / AirportOrigin (escale) / AirportPrevious (origine)
    # 1. Retrieve all airport codes in the dataset
    cols_airports = ['SysStopover', 'AirportOrigin', 'AirportPrevious']
    for col in cols_airports:
        data[col] = data[col].fillna('UNKNOWN').astype(str)

    # 2. Create the mapping dictionary (Code -> Index)
    tous_les_codes = pd.unique(data[cols_airports].values.ravel())
    airport_to_idx = {code: i for i, code in enumerate(tous_les_codes)}

    # 3. Apply the same mapping to all three columns
    for col in cols_airports:
        data[col] = data[col].map(airport_to_idx)

    # IdAircraftType (e.g Airbus 320)
    # 1. Create a dictionary for aircraft type
    unique_aircraft = data['IdAircraftType'].unique()

    # 2. Create the mapping dictionary (Code -> Index)
    aircraft_to_idx = {avion: i for i, avion in enumerate(unique_aircraft)}

    # 3. Apply the mapping to the data
    data['IdAircraftType'] = data['IdAircraftType'].map(aircraft_to_idx)

    # Convert date for delay computation
    data['LTScheduledDatetime'] = pd.to_datetime(data['LTScheduledDatetime'])
    data['LTBlockDatetime'] = pd.to_datetime(data['LTBlockDatetime'])

    # Delay computation
    # WARNING : DATA LEAKAGE FOR THE FUTURE FLIGHT
    data['Delay_Minutes'] = (data['LTBlockDatetime'] - data['LTScheduledDatetime']).dt.total_seconds() / 60
    data['Is_Delayed'] = (data['Delay_Minutes'] > 15).astype(int) #WARNING : 0 means "no delay and missing value"

    # removed_columns = ["FlightNumberNormalized", "LTScheduledDatetime", "LTBlockDatetime", 
    #                 "LTScheduledDatetime-day", "LTScheduledDatetime-hour-code"]

    # data = data.drop(columns=removed_columns)

    data.to_csv(data_new_filename, encoding='utf-8', index=False)







# -------------  Main execution -------------

if __name__=='__main__':
    pre_process_main(data_old_filename=main_old_filename, data_new_filename=main_new_filename)
    pre_process_holidays(data_old_filename=holidays_old_filename, data_new_filename=holidays_new_filename)
    pre_process_weather(data_old_filename=weather_old_filename, data_new_filename=weather_new_filename)