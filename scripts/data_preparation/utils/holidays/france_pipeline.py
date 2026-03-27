### Import
import pandas as pd
from utils.holidays.france_zones import get_zone_airports
from utils.holidays.france_daily_calendar import get_calendar_scholar_holidays


def add_scholar_holidays(df_flights, df_calendar):  
    # On transforme le calendrier pour avoir 3 colonnes : date, zone, est_vacances
    # On retire le préfixe 'Vacances_' pour matcher avec ta colonne 'holiday_zone'
    df_cal_long = df_calendar.melt(id_vars=['date'], var_name='holiday_zone', value_name='IsScholarHolidays')
    # df_cal_long['holiday_zone'] = df_cal_long['holiday_zone'].str.replace('Vacances_', '').str.replace('_', ' ')
    
    # 2. Normalisation des formats de date pour la jointure
    df_cal_long['date'] = pd.to_datetime(df_cal_long['date'])
    df_flights['LTScheduledDatetime-day'] = pd.to_datetime(df_flights['LTScheduledDatetime-day'])

    # 4. Jointure directe (Merge)
    # On fait correspondre (date + zone) du vol avec (date + zone) du calendrier
    df_fr = df_flights.merge(
        df_cal_long, 
        left_on=['LTScheduledDatetime-day', 'holiday_zone'], 
        right_on=['date', 'holiday_zone'], 
        how='left'
    )

    # 5. Nettoyage
    df_fr['IsScholarHolidays'] = df_fr['IsScholarHolidays'].fillna(0).astype(int)
    return df_fr.drop(columns=['date'])






def pipeline_france(data : pd.DataFrame)-> pd.DataFrame:
    data_holiday_zone = get_zone_airports(data["city"])
    data["holiday_zone"] = data_holiday_zone

    df_calendar_scholar_holiday = get_calendar_scholar_holidays()
    df_calendar_scholar_holiday.to_csv("calendar_fr_holidays_zone.csv", encoding='utf-8')

    df_resultat = add_scholar_holidays(df_flights=data, df_calendar = df_calendar_scholar_holiday)

    rmv_column_list = ["holiday_zone"]      #["LTScheduledDatetime-day", "OACI_code", "city", "holiday_zone"]
    df_resultat = df_resultat.drop(rmv_column_list, axis=1)
    return df_resultat
