import os
import glob
from Read_fluxnet import process_and_select_variables
import numpy as np
import netCDF4 as nc

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

def process_file(file_path):
    """
    Process a single file and return the processed DataFrame.
    """
    print(f"Processing {file_path}...")
    filtered_df = process_and_select_variables(file_path)
    filtered_df = filtered_df.replace([-9999, 2.0], np.nan)
    filtered_df = filtered_df.dropna(how='all')
    daily_avg_df = filtered_df.resample('D').mean()
    daily_avg_df = daily_avg_df.rolling(window=7, center=True, min_periods=1).mean()
    
    return daily_avg_df

def save_to_netcdf(df, file_path):
    """
    Save a DataFrame to a NetCDF file.
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
base_directory = r'C:\Users\asbre\OneDrive\Desktop\AI_WETLAND\Task1\Data\Raw\FULLSET'
processed_directory = r'C:\Users\asbre\OneDrive\Desktop\AI_WETLAND\Task1\Processed\Fullset_Processed'


site_files = read_filenames_and_sites(base_directory)

for site_name, file_path in site_files.items():
    processed_df = process_file(file_path)
    netcdf_file_path = os.path.join(processed_directory, f'FULLSET_{site_name}.nc')
    save_to_netcdf(processed_df, netcdf_file_path)
