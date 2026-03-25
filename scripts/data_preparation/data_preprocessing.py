import pandas as pd
import numpy as np
import os
from pathlib import Path


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



### Weather data ###
def remove_missing_values(csv_path : str) -> pd.DataFrame:
    """
    Remove columns with missing values.
    First the columns whose are empty. (100% NA).
    Secondly the columns with more than 25% of missings values, except 3 columns
    """

    # remove columns with 100% of missing values
    df = pd.read_csv(csv_path)
    df_no_empty = df.dropna(axis=1, how='all')
    removed_columns = set(df.columns) - set(df_no_empty.columns)
    print(f"{len(removed_columns)} deleted columns (100% empty).")

    # remove columns with > 25 % of missing values
    threshold = int(len(df) * 0.75)
    df_clean = df_no_empty.dropna(axis=1, thresh=threshold)
    removed_columns = set(df_no_empty.columns) - set(df_clean.columns)
    print(f"{len(removed_columns)} deleted columns (>25% empty).")

    return df_clean


def remove_bad_quality(df : pd.DataFrame) -> pd.DataFrame:
    """
    Each data point/column is assigned a quality code (e.g., T;QT):
        9: Filtered data (the data has passed the initial filters/checks)
        0: Protected data (the data has been definitively validated by the climatologist)
        1: validated data (the data has been validated by an automated check or by the climatologist)
        2: questionable data currently being verified (the data has been flagged as questionable by an automated check)
        
        We simply retain the valid data (with a quality code of 0, 1 or 9) and remove the columns with the quality code.
    """

    # Identify all data columns that have an associated “Q” column
    quality_columns = [c for c in df.columns if 'Q' + c in df.columns]

    for col in quality_columns:
        col_q = 'Q' + col
        
        # Define the mask for “bad” data (data that is not 0, 1, or 9)
        # Add fillna(2) to treat missing Q codes as “doubtful” as a precaution
        invalid = ~df[col_q].fillna(2).isin([0, 1, 9])
        
        # Clear the column if the quality is poor
        df.loc[invalid, col] = np.nan

    # Completely remove all columns starting with Q
    removed_columns = [c for c in df.columns if c.startswith('Q')]
    df_clean = df.drop(columns=removed_columns)

    return df_clean


def remove_unnecesserary_columns(df : pd.DataFrame) -> pd.DataFrame :
    """
    Remove the columns that are not relevant.
    """

    columns_to_remove = [
        "Unnamed: 0",
        "NUM_POSTE",
        "NOM_USUEL",
        "LAT",
        "LON",
        "ALTI",
        "HTN",
        "HTX",
        "HXI",
        "HXY",
        "HFXI3S",
        "HUN",
        "HUX",
        "PMER",
        "PSTAT",
        "PMERMIN",
        "CL",
        "VV",
        "DVV200",
        "WW",
        "W1",
        "W2",
        "STATUS_FXI3S",
        "STATUS_DXI3S"
    ]

    df_clean = df.drop(columns=columns_to_remove)

    return df_clean



def pre_process_weather(data_old_filename:str, data_new_filename:str):
    """
    Function used to pre-processed the main dataset get from the Bigquery table.

    Params:
        - data_old_filename: the filename to load the raw dataset.
        - data_new_filename: the filename to save the pre-processed dataset.
    """
    df = remove_missing_values(data_old_filename)
    df_1 = remove_bad_quality(df)
    df_2 = remove_unnecesserary_columns(df_1)
    df_2.to_csv(data_new_filename)
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

    data.to_csv(data_new_filename, encoding='utf-8')







# -------------  Main execution -------------

if __name__=='__main__':
    pre_process_main(data_old_filename=main_old_filename, data_new_filename=main_new_filename)
    pre_process_holidays(data_old_filename=holidays_old_filename, data_new_filename=holidays_new_filename)
    pre_process_weather(data_old_filename=weather_old_filename, data_new_filename=weather_new_filename)