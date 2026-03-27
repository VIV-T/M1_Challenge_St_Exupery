import pandas as pd
import requests

def add_school_holiday_international(df, date_col="LTScheduledDatetime-day", country_col="country"):
    """Add school holiday information for international flights."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["IsSchoolHoliday"] = 0

    for country in df[country_col].unique():
        country_data = df[df[country_col] == country]
        
        start_date = country_data[date_col].min()
        end_date = country_data[date_col].max()
        
        # Process date ranges within API limit (1095 days)
        for chunk_start, chunk_end in _split_date_range(start_date, end_date):
            holidays = _fetch_school_holidays(country, chunk_start, chunk_end)
            _mark_holidays_in_df(df, country, holidays, date_col, country_col)
    
    return df


def _split_date_range(start_date, end_date, max_days=1095):
    """Split date range into chunks within API limits."""
    chunks = []
    current_start = start_date
    
    while current_start <= end_date:
        current_end = min(current_start + pd.Timedelta(days=max_days), end_date)
        chunks.append((current_start, current_end))
        current_start = current_end + pd.Timedelta(days=1)
    
    return chunks


def _fetch_school_holidays(country, start_date, end_date):
    """Fetch school holidays from API for a specific country and date range."""
    url = "https://openholidaysapi.org/SchoolHolidays"
    params = {
        "countryIsoCode": country,
        "validFrom": start_date.strftime("%Y-%m-%d"),
        "validTo": end_date.strftime("%Y-%m-%d"),
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Handle different response formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'holidays' in data:
            return data['holidays']
        else:
            raise ValueError(f"Unexpected response format for {country}")
            
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error fetching holidays for {country}: {e}")


def _mark_holidays_in_df(df, country, holidays, date_col, country_col):
    """Mark school holidays in the DataFrame."""
    for holiday in holidays:
        if not isinstance(holiday, dict) or "startDate" not in holiday or "endDate" not in holiday:
            continue
            
        start = pd.to_datetime(holiday["startDate"])
        end = pd.to_datetime(holiday["endDate"])
        
        mask = (
            (df[country_col] == country) &
            (df[date_col] >= start) &
            (df[date_col] <= end)
        )
        
        df.loc[mask, "IsSchoolHoliday"] = 1