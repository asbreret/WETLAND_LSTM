import netCDF4 as nc
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os

site_code = 'US-Myb'  # Define the site code here

# File paths constructed dynamically using the site_code
base_file_path = f'C:\\Users\\asbre\\OneDrive\\Desktop\\AI_WETLAND\\Task1\\Processed\\Base_Processed\\BASE_{site_code}.nc'
fullset_file_path = f'C:\\Users\\asbre\\OneDrive\\Desktop\\AI_WETLAND\\Task1\\Processed\\Fullset_Processed\\FULLSET_{site_code}.nc'


def plot_variables(df, site_code, filename_segment):
    """
    Plot each variable in the DataFrame in a subplot layout of 3 columns.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the data to plot.
    site_code (str): The site code, used for titling and naming the saved plot.
    filename_segment (str): A string to include in the filename for the saved plot.
    """
    # Determine the number of variables and calculate rows needed for subplots
    num_variables = len(df.columns)
    num_rows = -(-num_variables // 3)  # Ceiling division to ensure enough rows
    
    # Create a figure with subplots
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, num_rows * 3))  # Adjust figure size as needed
    fig.suptitle(f'Variables for {site_code}', fontsize=16)
    
    # Flatten the axs array for easy indexing
    axs = axs.flatten()
    
    # Loop over variables to plot each
    for i, col in enumerate(df.columns):
        axs[i].plot(df.index, df[col])
        axs[i].set_title(col)
        axs[i].tick_params(labelrotation=45)  # Rotate labels to prevent overlap
    
    # Hide unused subplots if the number of variables is not a multiple of 3
    for ax in axs[num_variables:]:
        ax.set_visible(False)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust the top to accommodate suptitle
    
    # Create directory for saving the plots if it doesn't exist
    plot_directory = 'variable_plots'
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    
    # Save the figure with high resolution
    plot_filename = os.path.join(plot_directory, f'{site_code}_{filename_segment}_variables.png')
    plt.savefig(plot_filename, dpi=300)  # Save with high resolution
    plt.close(fig)  # Close the figure to free memory


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

def merge_dataframes(base_df, fullset_df):
    """
    Merge two DataFrames with time as index, prioritizing fullset_df variables.
    """
    intersecting_times = base_df.index.intersection(fullset_df.index)
    base_df_aligned = base_df.loc[intersecting_times]
    fullset_df_aligned = fullset_df.loc[intersecting_times]
    merged_df = base_df_aligned.combine_first(fullset_df_aligned)
    return merged_df

def remove_sparse_variables(df, threshold=400):
    """
    Remove variables with NaN counts above a specified threshold.
    """
    variables_to_keep = df.isnull().sum() < threshold
    filtered_df = df.loc[:, variables_to_keep]
    return filtered_df

def interpolate_and_clean(df):
    """
    Interpolate NaN values and remove rows still containing NaNs.
    """
    interpolated_df = df.interpolate()
    cleaned_df = interpolated_df.dropna()
    return cleaned_df

def remove_corresponding_variables(df):
    """
    Remove variables without '_F' if their '_F' flagged counterpart exists.
    """
    flag_variables = [col for col in df.columns if col.endswith('_F')]
    for flag_var in flag_variables:
        corresponding_var = flag_var[:-2]
        if corresponding_var in df.columns:
            df.drop(columns=[corresponding_var], inplace=True)
    return df

# Read data
base_df = read_data_from_netcdf(base_file_path)
fullset_df = read_data_from_netcdf(fullset_file_path)

# Process data
merged_df = merge_dataframes(base_df, fullset_df)

# Calculate TS_F here, after merging and before filtering or cleaning
ts_pi_columns = [col for col in merged_df.columns if col.startswith('TS_PI_')]
merged_df['TS_F'] = merged_df[ts_pi_columns].mean(axis=1, skipna=True)

#
merged_df_filtered = remove_sparse_variables(merged_df)
merged_df_interpolated = interpolate_and_clean(merged_df_filtered)
final_df = remove_corresponding_variables(merged_df_interpolated)

# `final_df` is now ready for further analysis or export.
# Example usage with your final DataFrame and site code
plot_variables(final_df, site_code, 'ALL')


# Assuming final_df is your initial DataFrame

# Define the columns for the drivers DataFrame
driver_columns = ['PA_F', 'PPFD_IN', 'P_F', 'RH', 'SW_IN_F', 'TA_F', 'T_SONIC', 'VPD_F', 'WS_F', 'TS_F']

# Filter out the driver columns to create the drivers DataFrame
drivers_df = final_df[driver_columns]

# Use the columns not in driver_columns for the ecosystem DataFrame
ecosystem_columns = [col for col in final_df.columns if col not in driver_columns]
ecosystem_df = final_df[['GPP_PI_F','RECO_PI_F']]

plot_variables(drivers_df, site_code, 'DRIVERS')
plot_variables(ecosystem_df, site_code, 'ECOSYSTEM')


# Save drivers_df and ecosystem_df with site_code included in the filename
save_to_netcdf(drivers_df, f'processed_netcdf/{site_code}_drivers_data.nc')
save_to_netcdf(ecosystem_df, f'processed_netcdf/{site_code}_ecosystem_data.nc')