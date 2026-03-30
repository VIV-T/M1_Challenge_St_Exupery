"""
Condition to run this script, run:
    - get_main.py
    - get_holidays.py

Purpose of the script:
    Prepare the final dataset, ready for training, with additional columns using or not external data souorces.
"""

### Imports
import pandas as pd
from pathlib import Path
import numpy as np
import json
import os 
from datetime import datetime, timedelta
from utils.holidays.env_variables import FEATURE_NAME_AIRPORT_CODE
from utils.main.add_features import add_route_feature, add_route_index, add_optimized_historical_features, add_optimized_flight_history

data_folder = os.path.join(Path(__file__).parent.parent.parent, "data")
config_folder = os.path.join(Path(__file__).parent.parent.parent, "config")
main_old_filename = os.path.join(data_folder, "main.csv")
main_new_filename = os.path.join(data_folder, "main_preprocessed.csv")
holidays_filename = os.path.join(data_folder, "holidays.csv")


### Holidays
def add_holidays_data(current_data : pd.DataFrame) -> pd.DataFrame:
    """
    Read the 'holidays.csv' file and add its content to our main data.
    Params:
        - current_data : pd.DataFrame containing our main data during the preprocess pipeline.
    """
    data_holidays = pd.read_csv(holidays_filename, encoding='utf-8')

    # Set the "LTScheduledDatetime" column as Datetime for each df, avoid type issues
    current_data['LTScheduledDatetime'] = pd.to_datetime(current_data['LTScheduledDatetime'])
    data_holidays['LTScheduledDatetime'] = pd.to_datetime(data_holidays['LTScheduledDatetime'])

    # Inner join of the current dataset and the holidays one, we keep only the shared flight.
    # Why? the preprocessed performed on the holdiadays dataset also applied to our main data.
    df_merge = pd.merge(current_data, data_holidays, on=["LTScheduledDatetime", FEATURE_NAME_AIRPORT_CODE], how="inner")

    return df_merge




### Main function ###
def pre_process_main(data_old_filename, data_new_filename)-> None:
    """
    Function used to pre-processed the main dataset get from the Bigquery table.

    Params:
        - data_old_filename: the filename to load the raw dataset.
        - data_new_filename: the filename to save the pre-processed dataset.
    """
    data = pd.read_csv(data_old_filename, encoding='utf-8')
    
    def date_columns_creation(data : pd.DataFrame):
        data["LTScheduledYear"] = pd.to_datetime(data['LTScheduledDatetime']).dt.year
        data["LTScheduledMonth"] = pd.to_datetime(data['LTScheduledDatetime']).dt.month
        data["LTScheduledDay"] = pd.to_datetime(data['LTScheduledDatetime']).dt.day
        data["LTScheduledHour"] = pd.to_datetime(data['LTScheduledDatetime']).dt.hour
        data["LTScheduledMinute"] = pd.to_datetime(data['LTScheduledDatetime']).dt.minute
        data["LTScheduledDayOfWeek"] = pd.to_datetime(data['LTScheduledDatetime']).dt.dayofweek


        for col, period in [('LTScheduledMinute', 60), ('LTScheduledHour', 24), ('LTScheduledMonth', 12), ('LTScheduledDayOfWeek', 7)]:
            data[f'sin_{col}'] = np.sin(2 * np.pi * data[col] / period)
            data[f'cos_{col}'] = np.cos(2 * np.pi * data[col] / period)
            data = data.drop(columns=[col])

        return data 
        
    # Sort the data following the chronological order
    today = datetime.now().date()
    yesterday = today - timedelta(days=2)
    data['LTScheduledDatetime'] = pd.to_datetime(data['LTScheduledDatetime'])
    data = data[data['LTScheduledDatetime'].dt.date <= yesterday]
    data = data.sort_values(by='LTScheduledDatetime').reset_index(drop=True)
    data = date_columns_creation(data=data)

    ### Add of holidays data
    # Have to be here in the code and not later ! => the numerical encoding of categorical data will lead to error otherwise. 
    data = add_holidays_data(data)

    # Only keep commercial flight
    data = data[data['IdBusinessUnitType'] == 1]

    # Direction - Arrivée -> 1 , Départ -> 0
    #data['Direction'] = data['Direction'].map({'Arrivée': 1, 'Départ': 0})


    ### To comment
    # mappings_filename = os.path.join(config_folder, "mappings_config.json")
    # all_mappings = {}
    # cols_airports = ['SysStopover', 'AirportOrigin', 'AirportPrevious', "IdAircraftType"]
    # for col in cols_airports:
    #     data[col] = data[col].fillna('UNKNOWN').astype(str)
        
    #     # Création du mapping Code -> ID (trié par ordre alphabétique pour la consistance)
    #     unique_values = sorted(data[col].unique())
    #     mapping_dict = {val: i for i, val in enumerate(unique_values)}
        
    #     # Application au DataFrame
    #     data[col] = data[col].map(mapping_dict)
        
    #     # Stockage pour sauvegarde JSON
    #     all_mappings[col] = mapping_dict

    
    # with open(mappings_filename, 'w', encoding='utf-8') as f:
    #     json.dump(all_mappings, f, indent=4, ensure_ascii=False)

    # Additional column creation
    data = add_route_feature(data=data)
    data = add_route_index(data=data)
    data["OccupancyRate"] = data["NbPaxTotal"] / data["NbOfSeats"]
    # OccupancyPreviousYears column creation
    data = add_optimized_historical_features(df=data)
    data = add_optimized_flight_history(df=data)

    # drop useless column
    # data = data.drop(columns=["LTScheduledDatetime"])
    
    # Saving the new dataframe
    data.to_csv(data_new_filename, encoding='utf-8', index=False)


if __name__=='__main__':
    pre_process_main(data_old_filename=main_old_filename, data_new_filename=main_new_filename)