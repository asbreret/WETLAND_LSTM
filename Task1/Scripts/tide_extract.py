import requests
from datetime import datetime
import pandas as pd
import calendar
import matplotlib.pyplot as plt

def fetch_tide_predictions_daily_mean(station_id, start_year, end_year):
    """
    Fetch tide prediction data and calculate daily mean levels for a given date range in years.
    
    Parameters:
    - station_id: Station ID for which to fetch tide predictions.
    - start_year: The start year of the date range.
    - end_year: The end year of the date range.
    
    Returns:
    - A pandas DataFrame with daily mean tide levels across the specified date range.
    """
    base_url = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'
    all_predictions = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            start_date = datetime(year, month, 1)
            days_in_month = calendar.monthrange(year, month)[1]  # Find the last day of the month
            end_date = datetime(year, month, days_in_month)
            
            params = {
                'product': 'predictions',
                'application': 'NOS.COOPS.TAC.WL',
                'datum': 'MLLW',
                'station': station_id,
                'time_zone': 'lst_ldt',
                'units': 'english',
                'interval': 'hilo',  # For hourly data, this should be adjusted if the API supports it.
                'format': 'json',
                'begin_date': start_date.strftime('%Y%m%d'),
                'end_date': end_date.strftime('%Y%m%d')
            }
            
            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                for pred in predictions:
                    all_predictions.append({
                        'time': datetime.strptime(pred['t'], '%Y-%m-%d %H:%M'),
                        'level': float(pred['v'])
                    })
            else:
                print(f"Failed to retrieve data for {start_date.strftime('%Y-%m')}:", response.status_code)
    
    # Convert collected data to DataFrame
    df_predictions = pd.DataFrame(all_predictions)
    df_predictions['time'] = pd.to_datetime(df_predictions['time'])
    df_predictions.set_index('time', inplace=True)
    
    # Calculate daily mean
    daily_mean = df_predictions.resample('D')['level'].mean()
    
    return daily_mean

def plot_daily_mean_tide_levels(daily_mean, year):
    """
    Plot daily mean tide levels for a specified year.
    
    Parameters:
    - daily_mean: pandas DataFrame with columns 'date' and 'level' representing daily mean tide levels.
    - year: The year for which to plot daily mean tide levels. This parameter is now used only for the title.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(daily_mean.index, daily_mean.values, label='Daily Mean Tide Level', color='green')
    plt.xlabel('Date')
    plt.ylabel('Mean Tide Level (feet)')
    plt.title(f'Daily Mean Tide Levels for {year}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()



# Example usage
station_id = '9415144'  # Replace with the desired station ID
start_year = 2010
end_year = 2021
daily_mean = fetch_tide_predictions_daily_mean(station_id, start_year, end_year)

plot_daily_mean_tide_levels(daily_mean, start_year)
