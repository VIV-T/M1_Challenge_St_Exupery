import pandas as pd
import os
from pathlib import Path


# ------------- Global variables -------------

data_folder = os.path.join(Path(__file__).parent.parent.parent, "data")
main_old_filename = f"{data_folder}/main.csv"
main_new_filename = f"{data_folder}/main_preprocessed.csv"

holidays_old_filename = f"{data_folder}/holidays.csv"
holidays_new_filename = f"{data_folder}/holidays_preprocessed.csv"




# -------------  Utils -------------

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

    data.to_csv(data_new_filename, encoding='utf-8')



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
    df_final = pd.DataFrame({'date': all_dates})


    def find_period(current_date):
        # Looking if the date is between start et end in our dataset
        match = data[(current_date >= data['start_date']) & 
                        (current_date <= data['end_date'])]
        
        if not match.empty:
            return match['name'].values[0]
        return None

    # Apply this function each line of the calendar
    df_final['name'] = df_final['date'].apply(find_period)
    df_final['holiday'] = df_final['name'].notnull().astype(int)
    
    # Save the new df
    df_final.to_csv(data_new_filename, encoding='utf-8')





# -------------  Main execution -------------

if __name__=='__main__':
    pre_process_main(data_old_filename=main_old_filename, data_new_filename=main_new_filename)
    pre_process_holidays(data_new_filename=holidays_new_filename, data_old_filename=holidays_old_filename)