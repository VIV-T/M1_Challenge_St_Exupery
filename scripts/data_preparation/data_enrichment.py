import pandas as pd
from pathlib import Path
import os 


# ------------- Global variables -------------

data_path = os.path.join(Path(__file__).parent.parent.parent, "data")
main_filename = f"{data_path}/main_preprocessed.csv"
holidays_filename = f"{data_path}/holidays_preprocessed.csv"
weather_filename = f"{data_path}/weather_preprocessed.csv"

# Data importation
main_data = pd.read_csv(main_filename)
holidays_data = pd.read_csv(holidays_filename)
weather_data = pd.read_csv(weather_filename)

# Merge with external data sources
data = pd.merge(main_data, holidays_data, left_on='LTScheduledDatetime-day', right_on='holidays_date', how="left")
full_data = pd.merge(data, weather_data, left_on="LTScheduledDatetime-hour-code", right_on="AAAAMMJJHH", how="left")


# Last transformations - remove of the useless columns for the training.
print(full_data.columns)
rmv_col_list = ['holidays_date', 'AAAAMMJJHH', "LTScheduledDatetime-day", "LTScheduledDatetime-hour-code"]
full_data = full_data.drop(columns=rmv_col_list)

# Sort the df following the chronological order
full_data = full_data.sort_values(by='LTScheduledDatetime', ascending=True)

# Save the new dataset
full_data.to_csv(f"{data_path}/dataset_training.csv", encoding='utf-8', index=False)

