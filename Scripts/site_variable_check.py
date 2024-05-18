import os
import sys

def add_path(path):
    """Adds the specified path to the Python module search path (sys.path)."""
    absolute_path = os.path.abspath(path)
    if absolute_path not in sys.path:
        sys.path.append(absolute_path)

# Add the ../Utilities directory to the Python path
utilities_dir = '../Utilities'
add_path(utilities_dir)

from netcdf_tools import (
    load_site_data, update_checks_site_data, update_training_sites,
    plot_variables_across_sites, create_site_dataframes, process_site_dataframes,
    apply_rolling_mean, apply_detrend, apply_wavelet_approx, save_to_netcdf
)


# Configuration settings
directory = r'C:\Users\asbre\OneDrive\Desktop\LSTM_Wetland_Model\Data\Raw\Netcdf'
save_directory = r'C:\Users\asbre\OneDrive\Desktop\LSTM_Wetland_Model\Data\Processed'
# driver_variables = ['PA', 'P', 'SW_IN', 'TA', 'VPD', 'WS', 'WTD', 'TS']
driver_variables = ['PA', 'P', 'SW_IN', 'TA', 'VPD', 'WS', 'TS']
target_variables = ['GPP_NT_VUT_REF', 'RECO_NT_VUT_REF', 'NEE_VUT_REF', 'FCH4']
suffix_priority_driver = ['_F']
suffix_priority_target = ['_PI_F', '_F']

# Option flags
rolling_mean_option = False  # Set to True to apply rolling mean
detrend_option = False        # Set to False if detrending should not be applied
wavelet_option = False        # Set to False if wavelet approximation should not be applied

# Step 1: Load site data
site_data = load_site_data(directory)

# Step 2: Perform initial checks on site data and generate table of available variables
update_checks_site_data(site_data, driver_variables, target_variables)

# Step 3: Filter sites where all required variables are present
training_sites = {
    site_code: info for site_code, info in site_data.items()
    if info['variable_check']['all_driver_vars_present'] and info['variable_check']['all_target_vars_present']
}

# Step 4: Define column settings for the update routine
df_columns = {
    'driver_variables': driver_variables,
    'target_variables': target_variables
}

# Step 5: Update training sites using the subroutine
training_sites = update_training_sites(training_sites, df_columns, suffix_priority_driver, suffix_priority_target)

# Step 6: Get the list of combined DataFrames
site_dataframes_list = create_site_dataframes(training_sites)

# Step 7: Process site dataframes
processed_site_dataframes_list = process_site_dataframes(site_dataframes_list)

# Step 8: Optionally apply data processing steps
processing_steps = []

if detrend_option:
    processed_site_dataframes_list = apply_detrend(processed_site_dataframes_list)
    processing_steps.append('detrend')

if rolling_mean_option:
    roll_window = 30  # Specify the rolling window in days
    processed_site_dataframes_list = apply_rolling_mean(processed_site_dataframes_list, roll_window)
    processing_steps.append('rolling')

if wavelet_option:
    processed_site_dataframes_list = apply_wavelet_approx(processed_site_dataframes_list)
    processing_steps.append('wavelet')

# Helper function to create filenames with processing steps
def create_filename(site_code, data_type):
    steps = '_'.join(processing_steps)
    return f"{site_code}_{data_type}.nc" if steps else f"{site_code}_{data_type}.nc"

# Step 9: Save processed dataframes to NetCDF
for site_code, driver_df, target_df in processed_site_dataframes_list:
    # Define file paths
    driver_file_path = os.path.join(save_directory, create_filename(site_code, 'drivers'))
    target_file_path = os.path.join(save_directory, create_filename(site_code, 'targets'))
    
    # Save to NetCDF
    save_to_netcdf(driver_df, driver_file_path)
    save_to_netcdf(target_df, target_file_path)

# Step 10: Plot the processed data
plot_variables_across_sites(processed_site_dataframes_list, driver_variables, target_variables)
