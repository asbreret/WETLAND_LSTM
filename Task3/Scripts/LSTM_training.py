import pandas as pd
import netCDF4 as nc
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Assuming ftnn is a module that contains the Fish_tank_NN function. Ensure it's imported or defined in your script.
from NN_model import Wetland_NN
#from tensorflow.keras.models import load_model
from hyperparameter_optimization import my_optimisation




def prepare_LSTM_data(features_df, target_df, sequence_length, start_index=0, end_index=None):
    """
    Prepare LSTM input and output data for a specified range, combining features and target for the input,
    and using the next time step of the target as the output. Outputs are formatted as 3D arrays.
    
    Args:
        features_df (DataFrame): The input features data with time series for each variable.
        target_df (DataFrame): The target data with time series for each variable.
        sequence_length (int): The number of time steps in each sequence.
        start_index (int): The starting index for data preparation.
        end_index (int): The ending index for data preparation. If None, uses the end of the DataFrame.
    
    Returns:
        LSTM_input_data (np.array): 3D array for LSTM input, shape (samples, sequence, features + target).
        LSTM_output_data (np.array): 3D array for LSTM output, shape (samples, 1, target).
    """
    if end_index is None:
        end_index = len(features_df)
    
    LSTM_input_data = []
    LSTM_output_data = []
    
    for i in range(start_index, end_index - sequence_length):
        # Combine features and target for the input sequence
        sequence_input_features = features_df.iloc[i:i+sequence_length].values
        sequence_input_target = target_df.iloc[i:i+sequence_length].values
        sequence_input = np.hstack((sequence_input_features, sequence_input_target))
        
        # The output sequence is the target at the next time step after the input sequence, reshaped to maintain 3D structure
        sequence_output = target_df.iloc[i + sequence_length].values.reshape(1, -1)  # Reshape to (1, target) for 3D
        
        LSTM_input_data.append(sequence_input)
        LSTM_output_data.append(sequence_output)
    
    LSTM_input_data = np.array(LSTM_input_data)
    LSTM_output_data = np.array(LSTM_output_data)
    
    return LSTM_input_data, LSTM_output_data

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

def find_year_indices(df, start_year, end_year):
    """
    Find the indices for the first occurrence in the start_year and the last occurrence in the end_year.
    """
    df.index = pd.to_datetime(df.index)
    start_index = df.index.searchsorted(pd.to_datetime(f'{start_year}-01-01'))
    end_index = df.index.searchsorted(pd.to_datetime(f'{end_year}-12-31'), side='right') - 1
    return start_index, end_index

# Scaling and preparing data
def scale_and_prepare_data(features_df, target_df, sequence_length, start_year, end_year):
    # Find year indices
    start_index, end_index = find_year_indices(features_df, start_year, end_year)

    # Scale features
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    features_scaled = pd.DataFrame(scaler_features.fit_transform(features_df), columns=features_df.columns, index=features_df.index)
    
    # Scale targets
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    target_scaled = pd.DataFrame(scaler_target.fit_transform(target_df), columns=target_df.columns, index=target_df.index)
    
    # Prepare LSTM data
    LSTM_input_data, LSTM_output_data = prepare_LSTM_data(features_scaled, target_scaled, sequence_length, start_index, end_index)
    
    return LSTM_input_data, LSTM_output_data, scaler_features, scaler_target

# Model training and saving function
def train_and_save_model(LSTM_input_data, LSTM_output_data, param_dict, model_filename):
    # Splitting the dataset into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(LSTM_input_data, LSTM_output_data, test_size=0.2)
    
    # Train the model
    net, history = Wetland_NN(X_train, X_val, Y_train, Y_val, **param_dict)
    
    # Save the model
    net.save(model_filename)
    return net, history







# Base path for all files
base_path = "../../Task2/Scripts/RandomForest/"

# Specific file names
file_names = [
    "US-Myb_drivers_gpp.nc",
    "US-Myb_drivers_reco.nc",
    "US-Myb_ecosystem_gpp.nc",
    "US-Myb_ecosystem_reco.nc"
]

# Reading each file into a separate DataFrame
df_gpp_drivers = read_data_from_netcdf(base_path + file_names[0])
df_reco_drivers = read_data_from_netcdf(base_path + file_names[1])
df_gpp_ecosystem = read_data_from_netcdf(base_path + file_names[2])
df_reco_ecosystem = read_data_from_netcdf(base_path + file_names[3])






# Example usage
sequence_length = 14
start_year = 2012
end_year = 2016
model_filename = 'model_GPP.h5'

# Adjust the DataFrame names as per your requirement
LSTM_input_data, LSTM_output_data, scaler_features, scaler_target = scale_and_prepare_data(df_gpp_drivers, df_gpp_ecosystem, sequence_length, start_year, end_year)



# X_train, X_val, Y_train, Y_val = train_test_split(LSTM_input_data, LSTM_output_data, test_size=0.2)
# # # Perform optimization
# best_hyperparams, all_runs = my_optimisation(X_train, X_val, Y_train, Y_val)

# # # Output the results
# print("Best Hyperparameters:", best_hyperparams)

# # Best Hyperparameters: {'batch_size': 72.0341124514471, 'dense_activation': 0.7203244934421581, 'dropout_rate': 5.718740867244332e-05, 'initial_learning_rate': 0.003093092469055214, 'lstm_activation': 0.14675589081711304, 'num_hidden_units': 127.70157843063934, 'num_layers': 1.3725204227553418}

#   Best Hyperparameters: {'batch_size': 72.0341124514471, 'dense_activation': 0.0,                'dropout_rate': 5.718740867244332e-05, 'initial_learning_rate': 0.003093092469055214, 'lstm_activation': 0.0, 'num_hidden_units': 127.70157843063934, 'num_layers': 1.1862602113776708}



# Define your model parameters in a dictionary as before
param_dict = {
    'numHiddenUnits': 130,
    'initial_learning_rate': 0.003,
    'num_layers': 1,
    'dropout_rate': 0.0,
    'batch_size': 64,
    'num_epochs': 1200,
    'learning_rate_schedule': 'exponential',
    'optimizer': 'adam',
    'early_stopping_patience': 50,
    'lstm_activation': 'tanh',
    'dense_activation': 'linear'
}

# Train and save the model
net, history = train_and_save_model(LSTM_input_data, LSTM_output_data, param_dict, model_filename)
