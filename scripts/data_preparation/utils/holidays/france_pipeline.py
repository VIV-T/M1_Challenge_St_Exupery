### Import
import pandas as pd
from utils.holidays.france_zones import get_zone_airports
from utils.holidays.france_daily_calendar import get_calendar_scholar_holidays


def add_scholar_holidays(df_flights, df_calendar): 
    """
    IsScholarHolidays variable creation : Binary that indicates if a days is part of an holiday period.
    Params:
        - df_flight : the clean pd.DataFrame with the flight and the corresponding holiday zone of the departure location
        - df_calendar : a pd.DataFrame containing all the holidays date for each holiday zone.
    """
    df_cal_long = df_calendar.melt(id_vars=['date'], var_name='holiday_zone', value_name='IsScholarHolidays')
    
    # Normalisation des formats de date pour la jointure
    df_cal_long['date'] = pd.to_datetime(df_cal_long['date'])
    df_flights['LTScheduledDatetime-day'] = pd.to_datetime(df_flights['LTScheduledDatetime-day'])

    # Merge and matching 
    df_fr = df_flights.merge(
        df_cal_long, 
        left_on=['LTScheduledDatetime-day', 'holiday_zone'], 
        right_on=['date', 'holiday_zone'], 
        how='left'
    )

    # CLeaning
    df_fr['IsScholarHolidays'] = df_fr['IsScholarHolidays'].fillna(0).astype(int)
    return df_fr.drop(columns=['date'])



def pipeline_france(data : pd.DataFrame)-> pd.DataFrame:
    """
    Execute the full pipeline for France data.
    Params:
        - data : the pd.DataFrame filtered to keep only the France data
    """
    data_holiday_zone = get_zone_airports(data["city"])
    data["holiday_zone"] = data_holiday_zone

    df_calendar_scholar_holiday = get_calendar_scholar_holidays()

    df_resultat = add_scholar_holidays(df_flights=data, df_calendar = df_calendar_scholar_holiday)

    rmv_column_list = ["holiday_zone"]      # ["LTScheduledDatetime-day", "OACI_code", "city", "holiday_zone"]
    df_resultat = df_resultat.drop(rmv_column_list, axis=1)
    return df_resultat
