### Imports
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys
import joblib

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))


from scripts.data_preparation.preprocessed import main_preprocessed
from scripts.data_preparation.get_main import column_list

# launch the get_main.py file to query the db

### Initialization & Global variables
## Paths 
root = Path(__file__).parent.parent.parent
DATA_FOLDER_PATH = os.path.join(root, "data")
DATASET_PATH = os.path.join(DATA_FOLDER_PATH, "main_preprocessed.csv")
MODEL_FOLDER_PATH = os.path.join(root, "models")
MODEL_MAIN_FILENAME = os.path.join(MODEL_FOLDER_PATH, "lgbm_regressor_model.pkl")

TARGET = "NbPaxTotal"
COLUMN_LIST = [col.strip() for col in column_list.strip().split(',') if col.strip()]    # col_list built from the get_main columns.

# Today & tomorrow
now = datetime.now()
today = pd.Timestamp(now.date())
tomorrow = pd.Timestamp(today + timedelta(days=1))

LIMIT_DATE_TRAIN = today
LIMIT_DATE_TEST = tomorrow
# output filename, depending on the date.
PREDICTIONS_FILENAME = os.path.join(DATA_FOLDER_PATH, f"prediction_{str(today)}.csv")


if __name__ == '__main__' : 
    # Load the preprocessed data - or build the needed df.
    if os.path.exists(DATASET_PATH) : 
        print('Loading preprocessed file ...')
        df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    else:
        print("Preprocessing ...")
        df = main_preprocessed()

    ### Creation of the Test DataFrame
    # Date filtering
    df['LTScheduledDatetime'] = pd.to_datetime(df['LTScheduledDatetime'])
    test_df = df[(df['LTScheduledDatetime'] >= LIMIT_DATE_TRAIN) & (df['LTScheduledDatetime'] <= LIMIT_DATE_TEST)].copy()
    print(f"Test set: {len(test_df)} rows / From {test_df['LTScheduledDatetime'].min()} to {test_df['LTScheduledDatetime'].max()}")

    # column management :
    #   - remove the target 
    X_new = df[COLUMN_LIST]
    X_new = df.drop(columns=[TARGET])

    
    #   - set categorical variables (same as the one defined during the training)
    #### TO DO - cf. Bapt.


    # load the model(s) - PHMR model need to be added.
    model = joblib.load(MODEL_MAIN_FILENAME)
    # predictions
    predictions = model.predict(X_new)  # issues with categorical variables => must be the same type in the X_new df than in the training df.


    # Results & save
    X_new['NbPaxTotal'] = predictions
    df_final = X_new['FlightNumberNormalized', 'LTScheduledDatetime', 'NbPaxTotal'] # + "PHMR column"
    df_final.to_csv(PREDICTIONS_FILENAME, encoding = 'utf-8', index=False)
