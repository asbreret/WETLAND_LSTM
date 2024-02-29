import pandas as pd
import numpy as np
import netCDF4 as nc
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os  # To handle directory operations

# Function to read data from a NetCDF file
def read_data_from_netcdf(file_path):
    """
    Read data from a NetCDF file into a DataFrame.
    """
    with nc.Dataset(file_path, 'r') as dataset:
        times = dataset.variables['time'][:]
        data_values = dataset.variables['values'][:]
        time_units = dataset.variables['time'].units
        calendar_used = dataset.variables['time'].calendar if 'calendar' in dataset.variables['time'].ncattrs() else 'gregorian'
        variable_names = dataset.getncattr('variable_names').split(',')

    dates = nc.num2date(times, units=time_units, calendar=calendar_used)
    dates = [datetime.datetime(d.year, d.month, d.day) for d in dates]
    df = pd.DataFrame(data_values, index=dates, columns=variable_names)
    return df

# Define the site code
site_code = 'US-Myb'

# Paths to the NetCDF files
drivers_file_path = f'processed_netcdf/{site_code}_drivers_data.nc'
ecosystem_file_path = f'processed_netcdf/{site_code}_ecosystem_data.nc'

# Load the data into dataframes
drivers_df = read_data_from_netcdf(drivers_file_path)
ecosystem_df = read_data_from_netcdf(ecosystem_file_path)

# Perform Pearson correlation between driver and ecosystem variables
correlation_results = pd.DataFrame(index=drivers_df.columns)
for eco_var in ecosystem_df.columns:
    correlation_results[eco_var] = drivers_df.corrwith(ecosystem_df[eco_var])

# Transpose the correlation_results for plotting
correlation_results = correlation_results.T

# Plotting the correlations as vertical bars in 2 rows of subplots, 1 column
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Loop through each ecosystem variable to create a subplot
for i, eco_var in enumerate(correlation_results.index):
    sns.barplot(x=correlation_results.columns, y=correlation_results.loc[eco_var], ax=axs[i])
    axs[i].set_title(f'Correlation with {eco_var}')
    axs[i].set_ylabel('Correlation Coefficient')
    axs[i].set_xlabel('Driver Variables')
    axs[i].tick_params(axis='x', rotation=45)
    axs[i].set_ylim(-1, 1)  # Set y-axis limits

plt.tight_layout()

# Create the 'Pearson' folder if it doesn't exist
output_folder = 'Pearson'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save the figure
plt.savefig(f'{output_folder}/{site_code}_pearson.png', dpi=300)  # Saving at higher resolution
plt.show()
