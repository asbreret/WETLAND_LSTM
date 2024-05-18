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


from netcdf_tools import read_filenames, load_data, save_to_netcdf, daily_interp

# Define the directories
data_directory = r'C:\Users\asbre\OneDrive\Desktop\LSTM_Wetland_Model\Data\Raw\CSV'
netcdf_directory = r'C:\Users\asbre\OneDrive\Desktop\LSTM_Wetland_Model\Data\Raw\Netcdf'

# Ensure the NetCDF directory exists
os.makedirs(netcdf_directory, exist_ok=True)

# Usage Example
csv_files = read_filenames(data_directory)
if not csv_files:
    print("No CSV files found.")

for file_path in csv_files:
    print(f"Processing {file_path}...")
    df = load_data(file_path)
    df_daily = daily_interp(df)
    # Construct the new file path in the NetCDF directory
    new_file_name = os.path.splitext(os.path.basename(file_path))[0] + '_daily.nc'
    new_file_path = os.path.join(netcdf_directory, new_file_name)
    save_to_netcdf(df_daily, new_file_path)
    print(f"Saved daily interpolated NetCDF file at {new_file_path}")
