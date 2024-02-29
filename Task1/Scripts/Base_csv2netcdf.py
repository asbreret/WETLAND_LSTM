import os
import glob
import numpy as np
import netCDF4 as nc
import pandas as pd
from variable_selection import primary_variables_metadata
import re

def read_filenames_and_sites(base_directory):
    """
    Reads filenames in the given directory and extracts site names from them.
    """
    csv_files = glob.glob(os.path.join(base_directory, '*.csv'))
    site_files_dict = {}

    for filename in csv_files:
        base_filename = os.path.basename(filename)
        start_index = base_filename.find('US-')
        end_index = base_filename.find('_', start_index)
        site_name = base_filename[start_index:end_index]
        site_files_dict[site_name] = filename  # Since only one file per site, directly assign

    return site_files_dict

def load_and_filter_data(file_path):
    """
    Load data from a CSV file and filter for specific variables, including patterns like TS_PI_1.
    """
    df = pd.read_csv(file_path, skiprows=2)  # Adjust skiprows if needed

    try:
        df['TIMESTAMP_START'] = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')
    except ValueError:
        df['TIMESTAMP_START'] = pd.to_datetime(df['TIMESTAMP_START'].apply(lambda x: '{:.0f}'.format(x)), format='%Y%m%d%H%M')

    df.set_index('TIMESTAMP_START', inplace=True)

    # Generate a regex pattern that matches the variable keys exactly or with a pattern extension (e.g., TS_PI_1)
    variable_keys_pattern = r'(' + '|'.join(primary_variables_metadata.keys()) + r')(_\w+)*$'
    
    # Filter columns based on the regex pattern
    df_filtered = df[[col for col in df.columns if re.match(variable_keys_pattern, col)]].copy()

    return df_filtered

def clean_and_aggregate_data(df_filtered):
    """
    Clean the filtered DataFrame, compute daily averages, and apply a rolling average.
    """
    df_filtered.replace([-9999, 2.0], np.nan, inplace=True)
    df_filtered.dropna(how='all', inplace=True)

    daily_avg_df = df_filtered.resample('D').mean()
    daily_avg_df = daily_avg_df.rolling(window=7, center=True, min_periods=1).mean()

    return daily_avg_df

def save_to_netcdf(df, file_path):
    """
    Save DataFrame to a NetCDF file.
    """
    folder_path = os.path.dirname(file_path)
    os.makedirs(folder_path, exist_ok=True)

    with nc.Dataset(file_path, 'w', format='NETCDF4') as dataset:
        dataset.createDimension('time', len(df))
        dataset.createDimension('variable', len(df.columns))
        times = dataset.createVariable('time', 'f8', ('time',))
        values = dataset.createVariable('values', 'f4', ('time', 'variable'))
        times[:] = nc.date2num(df.index.to_pydatetime(), units='days since 1970-01-01')
        values[:, :] = df.values
        times.units = 'days since 1970-01-01'
        times.calendar = 'gregorian'
        values.description = 'Processed data'
        dataset.setncattr_string('variable_names', ','.join(df.columns))

    print(f"File saved as {file_path}")

# Example usage
base_directory = r'C:\Users\asbre\OneDrive\Desktop\AI_WETLAND\Task1\Data\Raw\BASE'
processed_directory = r'C:\Users\asbre\OneDrive\Desktop\AI_WETLAND\Task1\Processed\Base_Processed'

site_files = read_filenames_and_sites(base_directory)

for site_name, file_path in site_files.items():
    filtered_df = load_and_filter_data(file_path)
    processed_df = clean_and_aggregate_data(filtered_df)
    processed_file_path = os.path.join(processed_directory, f"BASE_{site_name}.nc")
    save_to_netcdf(processed_df, processed_file_path)
