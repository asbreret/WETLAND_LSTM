import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import calendar

def plot_tide_predictions(df, start_date=None, end_date=None):
    """
    Plot tide predictions from a pandas DataFrame within the specified date range.
    
    Parameters:
    - df: pandas DataFrame with columns 'time' and 'level'.
    - start_date: Optional; The start date for the plot (inclusive).
    - end_date: Optional; The end date for the plot (inclusive).
    """
    # Filter the DataFrame based on the provided date range, if any
    if start_date is not None:
        df = df[df['time'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df['time'] <= pd.to_datetime(end_date)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['level'], label='Tide Level', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Tide Level (feet)')
    plt.title(f'Tide Predictions from {start_date} to {end_date}' if start_date and end_date else 'Tide Predictions for Full Year')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()



def fetch_and_plot_full_year_tide_predictions(station_id, year):
    """
    Fetch tide prediction data for a full year, given a station ID and year, and plot it.
    Returns the data as a pandas DataFrame.
    """
    base_url = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'
    params_template = {
        'product': 'predictions',
        'application': 'NOS.COOPS.TAC.WL',
        'datum': 'MLLW',
        'station': station_id,
        'time_zone': 'lst_ldt',
        'units': 'english',
        'interval': '30',
        'format': 'json'
    }

    all_predictions = []  # Store all predictions here
    
    for month in range(1, 13):
        start_date = datetime(year, month, 1)
        days_in_month = calendar.monthrange(year, month)[1]  # Correctly handles leap years
        end_date = datetime(year, month, days_in_month)
        
        params = params_template.copy()
        params['begin_date'] = start_date.strftime('%Y%m%d')
        params['end_date'] = end_date.strftime('%Y%m%d')
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            predictions = data['predictions']
            for pred in predictions:
                all_predictions.append({
                    'time': datetime.strptime(pred['t'], '%Y-%m-%d %H:%M'),
                    'level': float(pred['v'])
                })
        else:
            print(f"Failed to retrieve data for {start_date.strftime('%Y-%m')}:", response.status_code)
            continue
    
    # Convert collected data to DataFrame
    df_predictions = pd.DataFrame(all_predictions)
    
   
    return df_predictions

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
year = 2023  # Replace with the desired year
df_tide_predictions = fetch_and_plot_full_year_tide_predictions(station_id, year)
print(df_tide_predictions.head())  # Print the first few rows of the DataFrame



# Plotting tide predictions for a specific period
start_date = '2023-03-01'
end_date = '2023-03-31'
plot_tide_predictions(df_tide_predictions, start_date, end_date)

daily_mean = df_tide_predictions.resample('D', on='time')['level'].mean()


# Plot the daily mean tide levels for a specific year
plot_daily_mean_tide_levels(daily_mean, year)