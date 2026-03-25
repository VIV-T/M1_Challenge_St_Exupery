import pandas as pd
from pathlib import Path
import os 


# ------------- Global variables -------------

data_path = os.path.join(Path(__file__).parent.parent.parent, "data")
main_filename = f"{data_path}/main_preprocessed.csv"
holidays_filename = f"{data_path}/holidays_preprocessed.csv"
weather_filename = f"{data_path}/weather_preprocessed.csv"



main_data = pd.read_csv(main_filename)
holidays_data = pd.read_csv(holidays_filename)
weather_data = pd.read_csv(weather_filename)

data = pd.merge(main_data, holidays_data, left_on='LTScheduledDatetime-day', right_on='date', how="left")
full_data = pd.merge(data, weather_data, left_on="LTScheduledDatetime-hour-code", right_on="AAAAMMJJHH", how="left")

print(full_data)
full_data.to_csv(f"{data_path}/enriched_data.csv", encoding='utf-8')

