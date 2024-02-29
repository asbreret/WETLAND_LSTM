import pandas as pd
import netCDF4 as nc
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.model_selection import train_test_split


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

# Define the site code
site_code = 'US-Myb'

# Paths to the NetCDF files
drivers_file_path = f'processed_netcdf/{site_code}_drivers_data.nc'
ecosystem_file_path = f'processed_netcdf/{site_code}_ecosystem_data.nc'

# Load the data into dataframes
drivers_df = read_data_from_netcdf(drivers_file_path)
ecosystem_df = read_data_from_netcdf(ecosystem_file_path)

# Perform Random Forest analysis to determine the importance of driver variables
feature_importances = pd.DataFrame(index=drivers_df.columns)
# Adjusted part of the script for training the Random Forest model
for eco_var in ecosystem_df.columns:
    X = drivers_df
    y = ecosystem_df[eco_var].dropna()
    X = X.loc[y.index]  # Align the drivers with available ecosystem variable data

    # Splitting data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the Random Forest model with adjusted parameters
    rf = RandomForestRegressor(n_estimators=200,  # Increased from 100 to 200
                               max_depth=30,  # Set max depth of each tree
                               min_samples_split=4,  # Minimum number of samples required to split an internal node
                               min_samples_leaf=2,  # Minimum number of samples required to be at a leaf node
                               max_features='sqrt',  # Number of features to consider at every split
                               bootstrap=True,  # Use bootstrap samples when building trees
                               random_state=42)
    rf.fit(X_train, y_train)
    feature_importances[eco_var] = rf.feature_importances_









# Select the top N=2 drivers based on importance for GPP and RECO
N = 2
top_drivers_gpp = feature_importances['GPP_PI_F'].nlargest(N).index
top_drivers_reco = feature_importances['RECO_PI_F'].nlargest(N).index

# Prepare the dataframes for saving
drivers_gpp_df = drivers_df[top_drivers_gpp]
drivers_reco_df = drivers_df[top_drivers_reco]
ecosystem_gpp_df = ecosystem_df[['GPP_PI_F']]
ecosystem_reco_df = ecosystem_df[['RECO_PI_F']]

# Paths for saving
output_folder = 'RandomForest'
drivers_gpp_path = f'{output_folder}/{site_code}_drivers_gpp.nc'
drivers_reco_path = f'{output_folder}/{site_code}_drivers_reco.nc'
ecosystem_gpp_path = f'{output_folder}/{site_code}_ecosystem_gpp.nc'
ecosystem_reco_path = f'{output_folder}/{site_code}_ecosystem_reco.nc'

# Save to NetCDF
save_to_netcdf(drivers_gpp_df, drivers_gpp_path)
save_to_netcdf(drivers_reco_df, drivers_reco_path)
save_to_netcdf(ecosystem_gpp_df, ecosystem_gpp_path)
save_to_netcdf(ecosystem_reco_df, ecosystem_reco_path)



# Transpose the feature_importances for plotting
feature_importances = feature_importances.T



# Plotting the feature importances as vertical bars in 2 rows of subplots, 1 column
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Loop through each ecosystem variable to create a subplot
for i, eco_var in enumerate(feature_importances.index):
    sns.barplot(x=feature_importances.columns, y=feature_importances.loc[eco_var], ax=axs[i])
    axs[i].set_title(f'Random Forest Feature Importance for {eco_var}')
    axs[i].set_ylabel('Importance Score')
    axs[i].set_xlabel('Driver Variables')
    axs[i].tick_params(axis='x', rotation=45)

plt.tight_layout()

# Create the 'RandomForest' folder if it doesn't exist
output_folder = 'RandomForest'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save the figure
plt.savefig(f'{output_folder}/{site_code}_random_forest.png', dpi=300)  # Saving at higher resolution
plt.show()
    

