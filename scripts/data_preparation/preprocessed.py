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
import os 
from datetime import datetime, timedelta
from utils.holidays.env_variables import FEATURE_NAME_AIRPORT_CODE

data_folder = os.path.join(Path(__file__).parent.parent.parent, "data")
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
        # data["LTScheduledSecond"] = pd.to_datetime(data['LTScheduledDatetime']).dt.second
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
    data['Direction'] = data['Direction'].map({'Arrivée': 1, 'Départ': 0})

    # SysStopover (destination) / AirportOrigin (escale) / AirportPrevious (origine)
    # 1. Retrieve all airport codes in the dataset
    cols_airports = ['SysStopover', 'AirportOrigin', 'AirportPrevious']
    for col in cols_airports:
        data[col] = data[col].fillna('UNKNOWN').astype(str)

    # 2. Create the mapping dictionary (Code -> Index)
    all_codes = pd.unique(data[cols_airports].values.ravel())
    airport_to_idx = {code: i for i, code in enumerate(all_codes)}

    # 3. Apply the same mapping to all three columns
    for col in cols_airports:
        data[col] = data[col].map(airport_to_idx)

    # OccupancyRate column creation
    data["OccupancyRate"] = data["NbPaxTotal"] / data["NbOfSeats"]

    # drop useless column
    data = data.drop(columns=["LTScheduledDatetime"])
    
    # Saving the new dataframe
    data.to_csv(data_new_filename, encoding='utf-8', index=False)


if __name__=='__main__':
    pre_process_main(data_old_filename=main_old_filename, data_new_filename=main_new_filename)