import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pathlib import Path
import os 
from pandasql import sqldf
from datetime import datetime, timedelta

data_folder = os.path.join(Path(__file__).parent.parent.parent, "data")
filename = os.path.join(data_folder, "mouvements_aero_insa.csv")
# corr_matrix_filename = os.path.join(data_folder, "correlation_matrix_dataset_training.csv")

# Charger les données
data = pd.read_csv(filename, encoding='utf-8')

column_list = [
    "FlightNumberNormalized", # ID
    "LTScheduledDatetime", 
    "IdAircraftType",   # To remove
    "Direction", 
    "NbOfSeats",
    "NbPaxTotal",
    "ScheduleType", # To remove 
    "IdBusinessUnitType",  # To remove (after preprocessing !)
    "SysStopover", # Aristide 
    "AirportOrigin", # Aristide
    "AirportPrevious", # Aristide
    ]

data = data[column_list]

def date_columns_creation(data : pd.DataFrame):
    data["LTScheduledYear"] = pd.to_datetime(data['LTScheduledDatetime']).dt.year
    data["LTScheduledMonth"] = pd.to_datetime(data['LTScheduledDatetime']).dt.month
    data["LTScheduledDay"] = pd.to_datetime(data['LTScheduledDatetime']).dt.day
    data["LTScheduledHour"] = pd.to_datetime(data['LTScheduledDatetime']).dt.hour
    data["LTScheduledMinute"] = pd.to_datetime(data['LTScheduledDatetime']).dt.minute
    # data["LTScheduledSecond"] = pd.to_datetime(data['LTScheduledDatetime']).dt.second
    return data 


# Sort the df following the chronological order
today = datetime.now().date()
yesterday = today - timedelta(days=2)
data['LTScheduledDatetime'] = pd.to_datetime(data['LTScheduledDatetime'])
data = data[data['LTScheduledDatetime'].dt.date <= yesterday]
data = data.sort_values(by='LTScheduledDatetime').reset_index(drop=True)


data = data[data['IdBusinessUnitType'] == 1]
data = date_columns_creation(data=data)
data["OccupancyRate"] = data["NbPaxTotal"] / data["NbOfSeats"]
data.to_csv('new_test.csv', index=False, encoding='utf-8')




# define the function to execute SQL code.
pysqldf = lambda q: sqldf(q, globals())


# Exemple de requête SQL
res =  pysqldf("""
    SELECT DISTINCT(AirportPrevious)
    FROM data;
""")
print(res)
# if type(res) is pd.DataFrame :
#     nb_lines = res["COUNT(*)"][0]
# else : 
#     raise ValueError("Invalid query.") 
# print(f"Nmber of total line in this Table : {nb_lines}")


# #### Exploration stats ####
# print("\n\n---- Exploration Statistics ----\n")

# # define the function to execute SQL code.
# pysqldf = lambda q: sqldf(q, globals())


# # Exemple de requête SQL
# res =  pysqldf("""
#     SELECT COUNT(*)
#     FROM data;
# """)
# if type(res) is pd.DataFrame :
#     nb_lines = res["COUNT(*)"][0]
# else : 
#     raise ValueError("Invalid query.") 
# print(f"Nmber of total line in this Table : {nb_lines}")


# res = pysqldf("""
#     SELECT COUNT(*)
#     FROM data
#     WHERE OccupancyRate = 0.0;
# """)
# if type(res) is pd.DataFrame :
#     nb_empty_flight = res["COUNT(*)"][0]
# else : 
#     raise ValueError("Invalid query.") 
# print(f"Number of empty flight : {nb_empty_flight}")


# res = pysqldf("""
#     SELECT COUNT(*)
#     FROM data
#     WHERE OccupancyRate = 1.0;
# """)
# if type(res) is pd.DataFrame :
#     nb_full_flight = res["COUNT(*)"][0]
# else : 
#     raise ValueError("Invalid query.") 
# print(f"Number of full flight : {nb_full_flight}")


# print(f"\nRate of empty flight: {nb_empty_flight / nb_lines}")
# print(f"Rate of full flight: {nb_full_flight / nb_lines}\n")




# res = pysqldf("""
#     SELECT COUNT(*)
#     FROM data
#     WHERE OccupancyRate > 1.0;
# """)
# if type(res) is pd.DataFrame :
#     nb_overfull_flight = res["COUNT(*)"][0]
# else : 
#     raise ValueError("Invalid query.") 
# print(f"Number of overfull flight : {nb_overfull_flight}")
# print(f"Rate of overfull flight: {nb_overfull_flight / nb_lines}\n")    # Check the transit NbPax ? 




