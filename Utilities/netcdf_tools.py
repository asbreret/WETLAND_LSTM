import os
import glob
import netCDF4 as nc
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas.plotting import table

from matplotlib.colors import to_rgba
from scipy.signal import detrend

import pywt


def read_filenames(base_directory):
    """
    Reads filenames in the given directory for CSV files.
    """
    return glob.glob(os.path.join(base_directory, '*.csv'))



def load_data(file_path):
    """
    Load data from a CSV file, dynamically handling the number of header rows based on the filename.
    Converts 'TIMESTAMP_START' to a datetime index named 'time' and removes the original 'TIMESTAMP_START' and 'TIMESTAMP_END' columns.
    
    Args:
    file_path (str): The path to the CSV file to load.
    
    Returns:
    pd.DataFrame: A DataFrame with TIMESTAMP_START converted to datetime index and TIMESTAMP_END removed.
    """
    # Determine the number of rows to skip based on whether the file contains 'BASE' in its name,
    # assuming 'BASE' files include metadata in the first two rows
    skip_rows = 2 if 'BASE' in file_path else 0

    # Read the CSV file, skipping the appropriate number of header rows based on file type
    df = pd.read_csv(file_path, skiprows=skip_rows)

    # Identify all columns that contain 'TIMESTAMP' in their names
    timestamp_columns = [col for col in df.columns if 'TIMESTAMP' in col]

    # Iterate over the list of timestamp columns
    for col in timestamp_columns:
        # If the column name includes 'START', use it to set the DataFrame's datetime index
        if 'START' in col:
            df['time'] = pd.to_datetime(df[col], format='%Y%m%d%H%M', errors='coerce')  # Convert column to datetime
            df.set_index('time', inplace=True)  # Set the new datetime column as the index
            # Warn if any datetime conversions resulted in NaT (not-a-time), which indicates parsing issues
            if df.index.isna().any():
                print("Warning: Some 'time' index values could not be parsed.")
        # Drop the current timestamp column, whether it's 'START' or 'END'
        df.drop(columns=[col], inplace=True)

    return df




def daily_interp(df):
    """
    Interpolate data to daily frequency, handling missing values.
    """
    df.replace([-9999,2.0], np.nan, inplace=True)
    df.dropna(how='all', inplace=True)
    return df.resample('D').mean()


def save_to_netcdf(df, file_path):
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

def load_site_data(directory_path):
    """
    Scans a directory for NetCDF files, loads both FULLSET and BASE if both exist,
    and merges them prioritizing FULLSET.

    Args:
    directory_path (str): The path to the directory containing the NetCDF files.

    Returns:
    dict: A dictionary containing data and metadata for each site, including merged data if applicable.
    """
    site_files = {}
    site_info = {}

    # Scan all files in the directory and categorize them
    for filename in os.listdir(directory_path):
        if filename.endswith('.nc'):
            parts = filename.split('_')
            site_code = parts[0]
            dataset_type = parts[1].split('.')[0]

            if site_code not in site_files:
                site_files[site_code] = {}
            site_files[site_code][dataset_type] = os.path.join(directory_path, filename)

    # Process and merge data if both FULLSET and BASE exist
    for site, files in site_files.items():
        fullset_df = read_data_from_netcdf(files['FULLSET']) if 'FULLSET' in files else None
        base_df = read_data_from_netcdf(files['BASE']) if 'BASE' in files else None

        if fullset_df is not None and base_df is not None:
            # Merge FULLSET and BASE, prioritizing FULLSET
            df = fullset_df.combine_first(base_df)
        elif fullset_df is not None:
            df = fullset_df
        else:
            df = base_df

        if df is not None:
            start_year = df.index.year.min()
            full_years = df.index.year.max() - start_year + 1
            site_info[site] = {
                'dataframe': df,
                'start_year': start_year,
                'full_years': full_years,
                'variable_names': df.columns.tolist(),
                'file_path': files['FULLSET'] if 'FULLSET' in files else files['BASE']
            }

    return site_info


def site_data_table(site_data, output_figure_path):
    """
    Creates a table from the site data, displays it, and saves it as an image file. Rows with all variables present are highlighted in lime,
    and rows with any missing variables are highlighted in light red.

    Args:
    site_data (dict): Dictionary containing site data with necessary details.
    output_figure_path (str): Path where the figure image will be saved.
    """
    # Prepare data for the DataFrame
    data = {
        'Site Code': [],
        'Start Year': [],
        'Full Years': [],
        'All Driver Vars Present': [],
        'Missing Driver Vars': [],
        'All Target Vars Present': [],
        'Missing Target Vars': []
    }

    for site_code, info in site_data.items():
        data['Site Code'].append(site_code)
        data['Start Year'].append(info['start_year'])
        data['Full Years'].append(info['full_years'])
        data['All Driver Vars Present'].append(info['all_driver_vars_present'])
        data['Missing Driver Vars'].append(', '.join(info['missing_driver_vars']))
        data['All Target Vars Present'].append(info['all_target_vars_present'])
        data['Missing Target Vars'].append(', '.join(info['missing_target_vars']))

    # Create DataFrame from the data
    df = pd.DataFrame(data)

    # Filter to get only the rows with all variables present
    lime_df = df[df['All Driver Vars Present'] & df['All Target Vars Present']]

    # Calculate total and lime_rows * total
    total_trainable_full_years = lime_df['Full Years'].sum()
    lime_row_count = lime_df.shape[0]

    # Summary DataFrame
    summary_df = pd.DataFrame({
        'Total Trainable Full Years': [total_trainable_full_years],
        'Lime Rows * Total Full Years': [lime_row_count]
    })

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, len(df) * 0.35 + 0.5), gridspec_kw={'height_ratios': [len(df), 1]})

    # Main data table
    tab1 = table(ax1, df, loc='upper center', cellLoc='center', colWidths=[0.12]*7)
    tab1.auto_set_font_size(False)
    tab1.set_fontsize(14)  # Set font size
    tab1.scale(1.2, 1.2)
    ax1.axis('off')

    # Summary table
    summary_df = pd.DataFrame({
        'Total Trainable Full Years': [total_trainable_full_years],
        'Number of Lime Rows': [lime_row_count]
    })
    tab2 = table(ax2, summary_df, loc='upper center', cellLoc='center', colWidths=[0.3, 0.3])
    tab2.auto_set_font_size(False)
    tab2.set_fontsize(14)  # Set font size
    tab2.scale(1.2, 1.2)
    tab2.get_celld()[(0, 0)].set_facecolor(to_rgba('lime', alpha=0.5))
    tab2.get_celld()[(0, 1)].set_facecolor(to_rgba('lime', alpha=0.5))
    ax2.axis('off')

    # Apply coloring
    for (i, key), cell in tab1.get_celld().items():
        if i > 0:  # Skip header row
            if df.iloc[i - 1]['All Driver Vars Present'] and df.iloc[i - 1]['All Target Vars Present']:
                cell.set_facecolor(to_rgba('lime', alpha=0.5))
            else:
                cell.set_facecolor(to_rgba('lightcoral', alpha=0.5))

    # Save the table as an image
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(output_figure_path, bbox_inches='tight', dpi=300)
    plt.show()




def check_variables(df, driver_vars, target_vars):
    """
    Check the presence of driver and target variables in the DataFrame, either exactly or with an underscore.
    Also, list any missing variables.

    Args:
    df (pd.DataFrame): The DataFrame to check.
    driver_vars (list): List of driver variables to check for.
    target_vars (list): List of target variables to check for.

    Returns:
    dict: Dictionary containing boolean results and lists of missing variables.
    """
    results = {
        'all_driver_vars_present': True, 
        'all_target_vars_present': True,
        'missing_driver_vars': [],
        'missing_target_vars': []
    }

    # Check for driver variables
    for var in driver_vars:
        var_present = f"{var}" in df.columns or any(col.startswith(f"{var}_") for col in df.columns)
        results['all_driver_vars_present'] &= var_present
        if not var_present:
            results['missing_driver_vars'].append(var)

    # Check for target variables
    for var in target_vars:
        var_present = f"{var}" in df.columns or any(col.startswith(f"{var}_") for col in df.columns)
        results['all_target_vars_present'] &= var_present
        if not var_present:
            results['missing_target_vars'].append(var)

    return results




def update_checks_site_data(site_data, driver_variables, target_variables):
    """
    Updates the provided site_data dictionary with variable check results.
    
    Parameters:
        site_data (dict): A dictionary containing site data, with each key being a site code.
        driver_variables (list): List of driver variables to check in the data.
        target_variables (list): List of target variables to check in the data.
    """
    # Process each site's data
    for site_code, info in site_data.items():
        # Check for the presence of required variables in the data
        variable_check = check_variables(info['dataframe'], driver_variables, target_variables)
        info.update({
            'variable_check': variable_check,
            'all_driver_vars_present': variable_check['all_driver_vars_present'],
            'all_target_vars_present': variable_check['all_target_vars_present'],
            'missing_driver_vars': variable_check['missing_driver_vars'],
            'missing_target_vars': variable_check['missing_target_vars']
        })

        # Optionally print summary information for each site
        # print(f"Data for {site_code}:")
        # print(info['dataframe'].head())
        # print(f"Start Year: {info['start_year']}")
        # print(f"Full Years: {info['full_years']}")
        # print(f"All driver variables present: {info['all_driver_vars_present']}")
        # print(f"Missing driver variables: {info['missing_driver_vars']}")
        # print(f"All target variables present: {info['all_target_vars_present']}")
        # print(f"Missing target variables: {info['missing_target_vars']}")

    # You can also create a visual table or other outputs here
    site_data_table(site_data, 'site_data_table.png')



def find_variable_forms(df, variables, suffix_priority):
    best_forms = {}
    for var in variables:
        found = False
        
        # Special handling for the "TS" variable to capture all its variants
        if var == 'TS':
            # Find all columns that start with "TS"
            ts_variants = [col for col in df.columns if col.startswith('TS')]
            if ts_variants:
                best_forms[var] = ts_variants
                found = True

        # For variables other than "TS", follow the usual process
        if not found:
            for suffix in suffix_priority:
                full_var = f"{var}{suffix}"
                if full_var in df.columns:
                    best_forms[var] = full_var
                    found = True
                    break

        # Fallback to base variable if no prefixed form is found
        if not found:
            best_forms[var] = var

    return best_forms





def update_training_sites(training_sites, df_columns, suffix_priority_driver, suffix_priority_target):
    for site_code, info in training_sites.items():
        df = info['dataframe']
        info['best_driver_forms'] = find_variable_forms(df, df_columns['driver_variables'], suffix_priority_driver)
        info['best_target_forms'] = find_variable_forms(df, df_columns['target_variables'], suffix_priority_target)

        # Get lists of the best forms of driver and target variables
        best_driver_forms = list(info['best_driver_forms'].values())
        best_target_forms = list(info['best_target_forms'].values())

        # Process TS variables separately if they are listed
        if 'TS' in info['best_driver_forms']:
            ts_columns = [col for col in info['best_driver_forms']['TS'] if not col.endswith('_QC')]
            if ts_columns:  # Ensure there are valid TS columns to process
                ts_mean = df[ts_columns].mean(axis=1)
                df['TS_mean'] = ts_mean  # Add mean as a new column
                best_driver_forms.append('TS_mean')  # Include this in the driver forms
            # Remove the original TS list entry to avoid errors in subsequent DataFrame operations
            best_driver_forms = [col for col in best_driver_forms if not isinstance(col, list)]
            info['best_driver_forms'] = best_driver_forms  # Update info with the new best_driver_forms

        # Check and filter the dataframe for driver variables
        if all(col in df.columns for col in best_driver_forms):
            drivers_df = df[best_driver_forms]
        else:
            missing_cols = [col for col in best_driver_forms if col not in df.columns]
            raise ValueError(f"Missing driver columns in dataframe for site {site_code}: {missing_cols}")

        # Check and filter the dataframe for target variables
        if all(col in df.columns for col in best_target_forms):
            targets_df = df[best_target_forms]
        else:
            missing_cols = [col for col in best_target_forms if col not in df.columns]
            raise ValueError(f"Missing target columns in dataframe for site {site_code}: {missing_cols}")

        # Add these filtered dataframes as new entries in the site info
        info['drivers_df'] = drivers_df
        info['targets_df'] = targets_df

    return training_sites



def plot_variables_across_sites(site_dataframes_list, driver_vars, target_vars):
    total_plots = len(driver_vars) + len(target_vars)
    cols = 3  
    rows = (total_plots + cols - 1) // cols  

    plt.figure(figsize=(30, 5 * rows))  

    # Plot driver variables in the first two columns
    for i, var in enumerate(driver_vars):
        column_position = (i % (cols - 1)) + 1  
        row_position = (i // (cols - 1)) + 1
        ax_index = (row_position - 1) * cols + column_position
        ax = plt.subplot(rows, cols, ax_index)
        print(f"Plotting driver variable: {var}")
        for site_data in site_dataframes_list:
            if len(site_data) == 3:
                site_code, drivers_df, _ = site_data
                ax.plot(drivers_df.index, drivers_df.iloc[:, i], label=site_code)  # Use the first column
        ax.set_title(f"Driver: {var}")
        ax.set_xlabel('Date')
        ax.set_ylabel(var)

    # Plot target variables in the last column
    
    fallback_vars = {
    'FCH4': 'FCH4_PI_F'
    }
    
    for j, var in enumerate(target_vars):
        column_position = cols
        row_position = (j // 1) + 1
        ax_index = (row_position - 1) * cols + column_position
        ax = plt.subplot(rows, cols, ax_index)
        print(f"Plotting target variable: {var}")
        for site_data in site_dataframes_list:
            if len(site_data) == 3:
                site_code, _, targets_df = site_data
                try:
                    ax.plot(targets_df.index, targets_df[var], label=site_code)
                except KeyError:
                    if var in fallback_vars:
                        fallback_var = fallback_vars[var]
                        if fallback_var in targets_df.columns:
                            ax.plot(targets_df.index, targets_df[fallback_var], label=site_code)
                        else:
                            print(f"Warning: {var} and fallback {fallback_var} not found in {site_code}")
                    else:
                        print(f"Warning: {var} not found in {site_code}")
        ax.set_title(f"Target: {var}")
        ax.set_xlabel('Date')
        ax.set_ylabel(var)

    plt.subplots_adjust(right=0.85)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()

def create_site_dataframes(training_sites):
    site_dataframes = []
    for site_code, site_info in training_sites.items():
        # Assuming drivers_df and targets_df are your dataframes
        drivers_df = site_info['drivers_df']
        targets_df = site_info['targets_df']
        
        # Ensure both DataFrames have the same date index before joining
        if not drivers_df.index.equals(targets_df.index):
            print(f"Warning: Index mismatch for site {site_code}")
            print(f"Drivers_df index: {drivers_df.index}")
            print(f"Targets_df index: {targets_df.index}")
        
        # Assign name attributes to the DataFrames
        drivers_df.name = 'drivers_df'
        targets_df.name = 'targets_df'
        
        site_dataframes.append((site_code, drivers_df, targets_df))

    return site_dataframes



def process_site_dataframes(site_dataframes_list):
    # Create a new list variable to store the processed dataframes
    processed_site_dataframes_list = []
    
    # Loop through each site's dataframes
    for site_data in site_dataframes_list:
        site_code, drivers_df, targets_df = site_data
        
        # Interpolate NaNs within existing data regions for drivers_df, no extrapolation
        drivers_df = drivers_df.interpolate(method='time', limit_area='inside')
        
        # Interpolate NaNs within existing data regions for targets_df, no extrapolation
        targets_df = targets_df.interpolate(method='time', limit_area='inside')
        
        # Find rows that have any NaNs and remove
        combined_df = pd.concat([drivers_df, targets_df], axis=1)
        nan_rows = combined_df.isnull().any(axis=1)
        combined_df = combined_df[~nan_rows]
        
        # Split dataframes back
        drivers_df = combined_df.iloc[:, :len(drivers_df.columns)]
        targets_df = combined_df.iloc[:, len(drivers_df.columns):]
        
        # Append the processed dataframes to the new list variable
        processed_site_dataframes_list.append((site_code, drivers_df, targets_df))
    
    return processed_site_dataframes_list



def apply_rolling_mean(processed_site_dataframes_list, window_days):
    # Updated list to store dataframes with applied rolling mean
    rolled_site_dataframes_list = []
    
    # Loop through each site's processed dataframes
    for site_data in processed_site_dataframes_list:
        site_code, drivers_df, targets_df = site_data
        
        # Apply rolling mean with a specified window and minimum number of observations as 1
        drivers_rolled = drivers_df.rolling(window=f'{window_days}D', min_periods=1).mean()
        targets_rolled = targets_df.rolling(window=f'{window_days}D', min_periods=1).mean()
        
        # Append the rolled dataframes to the new list
        rolled_site_dataframes_list.append((site_code, drivers_rolled, targets_rolled))
    
    return rolled_site_dataframes_list



def apply_detrend(processed_site_dataframes_list):
    # List to store detrended dataframes
    detrended_site_dataframes_list = []

    # Loop through each site's processed dataframes
    for site_data in processed_site_dataframes_list:
        site_code, drivers_df, targets_df = site_data
        
        # Apply detrending to each column of both dataframes
        drivers_detrended = drivers_df.apply(detrend, axis=0)
        targets_detrended = targets_df.apply(detrend, axis=0)
        
        # Append the detrended dataframes to the new list
        detrended_site_dataframes_list.append((site_code, drivers_detrended, targets_detrended))
    
    return detrended_site_dataframes_list



    
def apply_wavelet_approx(processed_site_dataframes_list, wavelet='db4', levels=4):
    # List to store wavelet approximated dataframes
    approx_site_dataframes_list = []

    # Loop through each site's processed dataframes
    for site_data in processed_site_dataframes_list:
        site_code, drivers_df, targets_df = site_data
        
        # Function to apply wavelet approximation to each column
        def wavelet_approx(series):
            nonlocal levels
            series = series.dropna()  # Ensure there are no NaNs, or handle them as you see fit
            
            # Automatically determine the maximum level of decomposition if not specified
            if levels is None:
                levels = pywt.dwt_max_level(len(series), pywt.Wavelet(wavelet).dec_len)
            
            coeffs = pywt.wavedec(series, wavelet, level=levels)
            
            # Directly reconstruct the approximation component
            approximation = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], wavelet)[:len(series)]
            
            # Create a DataFrame with the approximation component
            return pd.Series(approximation, index=series.index)
        
        # Apply wavelet approximation to each column of both dataframes
        drivers_approx = drivers_df.apply(wavelet_approx, axis=0)
        targets_approx = targets_df.apply(wavelet_approx, axis=0)

        # Append the wavelet approximated dataframes to the new list
        approx_site_dataframes_list.append((site_code, drivers_approx, targets_approx))
    
    return approx_site_dataframes_list