"""
Condition to run this script, run:
    - get_main.py

Purpose of the script:
    Prepare the final dataset, ready for training, with additional columns using or not external data souorces.
"""

### Imports
import pandas as pd
from pathlib import Path
import os 

# Personal imports
from scripts.data_preparation.utils.holidays.env_variables import FEATURE_NAME_AIRPORT_CODE
from scripts.data_preparation.get_holidays_pipeline import main_holiday_pipeline
from scripts.data_preparation.utils.main.add_features import add_features


data_folder = os.path.join(Path(__file__).parent.parent.parent, "data")
config_folder = os.path.join(Path(__file__).parent.parent.parent, "config")
main_old_filename = os.path.join(data_folder, "main.csv")
main_new_filename = os.path.join(data_folder, "main_preprocessed.csv")
main_new_filename_PHMR = os.path.join(data_folder, "main_preprocessed_PHMR.csv")
holidays_filename = Path(os.path.join(data_folder, "holidays.csv"))



def main_preprocessed(data_old_filename = main_old_filename, main_new_filename = main_new_filename, with_holidays : bool =False) -> pd.DataFrame :
    # initialization
    data = pd.read_csv(data_old_filename, encoding='utf-8')


    ### Holidays
    if with_holidays :  # depends if we want to use the holiday preprocessing ? 
        if holidays_filename.exists : 
            data_holidays = pd.read_csv(holidays_filename, encoding='utf-8')
        else:
            data_holidays = main_holiday_pipeline()

        # date conversion - preparation for merge.    
        data['LTScheduledDatetime'] = pd.to_datetime(data['LTScheduledDatetime'])
        data_holidays['LTScheduledDatetime'] = pd.to_datetime(data_holidays['LTScheduledDatetime'])

        ##### Carreful - cf. data.shape (nb rows before/ after the merge)
        # Inner join of the current dataset and the holidays one, we keep only the shared flight.
        # Why? the preprocessed performed on the holdidays dataset also applied to our main data.
        # Number of row: 365005 to 337375       -> find the reason why
        # print(data.shape)
        data = pd.merge(data, data_holidays, on=["LTScheduledDatetime", FEATURE_NAME_AIRPORT_CODE], how="inner")
        # print(data.shape)
    
    ### Splitting the two dataset, one for NbPaxTotal, an other for PHMR
    data_Pax = data.drop(columns=["FarmsNbPaxPHMR"])
    data_PHMR = data.drop(columns=["NbPaxTotal"])

    ### Main - add of features (stats computation) / Only for NbPaxTotal
    data_Pax = add_features(df=data_Pax)  

    # save 
    data_Pax.to_csv(main_new_filename, encoding='utf-8', index=False)
    data_PHMR.to_csv(main_new_filename_PHMR, encoding='utf-8', index=False)

    return data_Pax, data_PHMR


# ## Test
# if __name__=='__main__':
#     main_preprocessed(data_old_filename=main_old_filename, main_new_filename=main_new_filename)