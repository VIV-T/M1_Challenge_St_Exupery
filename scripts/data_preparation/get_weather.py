import pandas as pd
import ssl
from pathlib import Path
import os

### Global variable
data_folder = os.path.join(Path(__file__).parent.parent.parent, "data")

"""
Source : meteo.data.gouv.fr

The data is divided into two separate sets: one covers the period from 2020 to 2024, and the other covers the period from 2025 to the present.
The second dataset is updated daily.
Since the data we are interested in only begins in 2023, this script retrieves weather data starting from 2023.
"""

# ignore SSL error
ssl._create_default_https_context = ssl._create_unverified_context

def get_data_from_2023_to_2024(url : str) -> pd.DataFrame:

    """
    Get the data from 2023 to 2024.
    """
    # Direct reading (Pandas handles .gz decompression and CSV on its own)
    # We use ‘sep=;’ because that's the Météo-France format
    df = pd.read_csv(url, compression='gzip', sep=';', low_memory=False)
    df = df[df['NOM_USUEL'] == 'LYON-ST EXUPERY']
    df = df[(df['AAAAMMJJHH'] >= 2023010100)]
    
    return df

def get_data_from_2025_to_today(url :str) -> pd.DataFrame:

    """
    Get the data from 2025 to today.
    """

    # Direct reading (Pandas handles .gz decompression and CSV on its own)
    # We use ‘sep=;’ because that's the Météo-France format
    df = pd.read_csv(url, compression='gzip', sep=';', low_memory=False)
    df = df[df['NOM_USUEL'] == 'LYON-ST EXUPERY']

    return df

def concat_data(df1 : pd.DataFrame, df2 : pd.DataFrame) -> pd.DataFrame:
    
    """
    Combine the two datasets (since they have the same columns), then sort them by date.
    """
    df = pd.concat([df1, df2], axis=0)

    # sort by date
    df = df.sort_values(by='AAAAMMJJHH')

    # reinitialize index
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":

    weather_2020_2024_hourly = "https://www.data.gouv.fr/api/1/datasets/r/3d560889-526f-479f-9ecd-ecdc43c009be"
    weather_2025_today_hourly = "https://www.data.gouv.fr/api/1/datasets/r/7fa40c33-31cd-40e6-9bd4-64d05ab308a3"
    output_file = os.path.join(data_folder, "weather.csv")
    df_weather_hourly_1 = get_data_from_2023_to_2024(weather_2020_2024_hourly)
    # print(len(df_weather_hourly_1))
    df_weather_hourly_2 = get_data_from_2025_to_today(weather_2025_today_hourly)
    # print(len(df_weather_hourly_2))
    df_weather_hourly = concat_data(df_weather_hourly_1, df_weather_hourly_2)
    print(f"Weather dataset created with {len(df_weather_hourly)} rows.")
    df_weather_hourly.to_csv(output_file)
    print("Weather dataset saved in weather.csv")

    

