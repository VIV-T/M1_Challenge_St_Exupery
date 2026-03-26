import pandas as pd
from pathlib import Path
import os 
from datetime import datetime, timedelta

data_folder = os.path.join(Path(__file__).parent.parent.parent, "data")
filename = os.path.join(data_folder, "main.csv")
new_filename = os.path.join(data_folder, "main_preprocessed.csv")


### Main data ###
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
    pre_process_main(data_old_filename=filename, data_new_filename=new_filename)