import pandas as pd
import netCDF4 as nc
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# Assuming NN_model.py contains the Wetland_NN function and hyperparameter_optimization.py contains the my_optimisation function
from NN_model import Wetland_NN
#from hyperparameter_optimization import my_optimisation


# ----- Function Definitions -----


def rolling_forecast(model, LSTM_input_train, LSTM_input_predict, forecast_days=10):
    """
    Perform a rolling forecast for a specified number of days using an LSTM model.

    The function uses the last training sequence to make the initial prediction, 
    then updates the input sequence with actual features and predicted targets,
    and repeats this process for the desired number of forecast days.
    
    Args:
        model: Trained LSTM model.
        LSTM_input_train (np.array): Training data, shape (samples, 14, features).
        LSTM_input_predict (np.array): Data for prediction phase, shape (samples, 14, features).
        forecast_days (int): Number of days to forecast.

    Returns:
        np.array: Forecasted target values, shape (forecast_days,).
    """
    # Initialize the sequence with the last sequence from the training data
    last_sequence = LSTM_input_train[-1:, :, :]  # Shape: (1, 14, features)

    # Initialize an empty list to store the forecasted target values
    forecasted_targets = []

    for i in range(forecast_days):
        # Predict the next target value
        predicted_target = model.predict(last_sequence)
        forecasted_targets.append(predicted_target.flatten()[0])

        # Prepare the next input sequence
        if i < forecast_days - 1:
            # Shift the sequence by one day, dropping the oldest day
            next_sequence = np.roll(last_sequence, -1, axis=1)
            # Update the last day with actual features (except target) and the predicted target
            # If available, use actual features for the next day, otherwise use the last known features
            if i < len(LSTM_input_predict) - 1:
                next_features = LSTM_input_predict[i, -1, :-1]  # Actual features for the next day
            else:
                next_features = last_sequence[0, -1, :-1]  # Last known features
            # Combine next_features with the predicted target to form the new last day
            new_last_day = np.concatenate((next_features, predicted_target.flatten()), axis=None)
            # Update the sequence with this new last day
            next_sequence[0, -1, :] = new_last_day
            last_sequence = next_sequence

    return np.array(forecasted_targets)



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

def scale_and_prepare_data(features_df, target_df, sequence_length, start_year, end_year):
    # Find year indices
    start_index, end_index = find_year_indices(features_df, start_year, end_year)

    # Scale features
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    features_scaled = pd.DataFrame(scaler_features.fit_transform(features_df), 
                                   columns=features_df.columns, 
                                   index=features_df.index)
    
    # Scale targets
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    target_scaled = pd.DataFrame(scaler_target.fit_transform(target_df), 
                                 columns=target_df.columns, 
                                 index=target_df.index)
    
    # Prepare LSTM data
    LSTM_input_data, LSTM_output_data = prepare_LSTM_data(features_scaled, target_scaled, sequence_length, start_index, end_index)
    
    # Extract datetime index (assuming the index is the same for both features_df and target_df)
    datetime_index = features_scaled.index[start_index:end_index+1]  # Adjust the slicing as needed
    
    return LSTM_input_data, LSTM_output_data, scaler_features, scaler_target, datetime_index

# Model training and saving function
def train_and_save_model(LSTM_input_data, LSTM_output_data, param_dict, model_filename):
    # Splitting the dataset into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(LSTM_input_data, LSTM_output_data, test_size=0.2)
    
    # Train the model
    net, history = Wetland_NN(X_train, X_val, Y_train, Y_val, **param_dict)
    
    # Save the model
    net.save(model_filename)
    return net, history




# ----- Data Preparation -----

base_path = "../../Task2/Scripts/RandomForest/"
file_names = ["US-Myb_drivers_gpp.nc", "US-Myb_drivers_reco.nc", "US-Myb_ecosystem_gpp.nc", "US-Myb_ecosystem_reco.nc"]

df_gpp_drivers = read_data_from_netcdf(base_path + file_names[0])
df_reco_drivers = read_data_from_netcdf(base_path + file_names[1])
df_gpp_ecosystem = read_data_from_netcdf(base_path + file_names[2])
df_reco_ecosystem = read_data_from_netcdf(base_path + file_names[3])

sequence_length = 14
start_year = 2012
split_year = 2019
end_year = 2020
model_filename = 'model_GPP.h5'

LSTM_input_data, LSTM_output_data, scaler_features, scaler_target, datetime_indices = scale_and_prepare_data(df_gpp_drivers, df_gpp_ecosystem, sequence_length, start_year, end_year)

split_index = datetime_indices.get_loc(datetime_indices[datetime_indices.year == split_year][0])
LSTM_input_train = LSTM_input_data[:split_index]
LSTM_input_predict = LSTM_input_data[split_index:]
LSTM_output_train = LSTM_output_data[:split_index]
LSTM_output_predict = LSTM_output_data[split_index:]



# ----- Model Training (Optional) -----

# Define your model parameters in a dictionary as before
param_dict = {
    'numHiddenUnits': 130,
    'initial_learning_rate': 0.003,
    'num_layers': 1,
    'dropout_rate': 0.0,
    'batch_size': 64,
    'num_epochs': 600,
    'learning_rate_schedule': 'exponential',
    'optimizer': 'adam',
    'early_stopping_patience': 50,
    'lstm_activation': 'tanh',
    'dense_activation': 'linear'
}


# Uncomment and customize this section as needed for training and saving your model
# net, history = train_and_save_model(LSTM_input_train, LSTM_output_train, param_dict, model_filename)

# ----- Forecasting and Plotting -----

net = load_model(model_filename)  # Load your trained model

forecast_days = 30  # Adjust based on your forecasting needs
predicted_targets = rolling_forecast(net, LSTM_input_train, LSTM_input_predict, forecast_days)

actual_targets = LSTM_input_predict[:forecast_days, -1, -1]  # Extract actual targets for comparison





# Assuming 'predicted_targets' are your model's predictions and 'actual_targets' are the true values
predicted_targets_reshaped = predicted_targets.reshape(-1, 1)  # Reshape predictions to match the scaler's expected input
actual_targets_reshaped = actual_targets.reshape(-1, 1)  # Reshape actual targets for inverse transformation

# Inverse transform to original scale
predicted_targets_original = scaler_target.inverse_transform(predicted_targets_reshaped)
actual_targets_original = scaler_target.inverse_transform(actual_targets_reshaped)

# Extract datetime indices for the forecast period
forecast_datetime_indices = datetime_indices[split_index:split_index + forecast_days]

# Ensure that the length of forecast_datetime_indices matches that of your predictions and actual targets
# This might not be necessary if you're sure they align, but it's a good practice to check
if len(forecast_datetime_indices) != len(predicted_targets_original):
    forecast_datetime_indices = forecast_datetime_indices[:len(predicted_targets_original)]

# Create the 'Figures' subdirectory if it doesn't exist
figures_directory = 'Figures'
if not os.path.exists(figures_directory):
    os.makedirs(figures_directory)

# Plotting the results after inverse transformation with time on the x-axis
plt.figure(figsize=(10, 6), dpi=300)  # Increase dpi for higher resolution
plt.plot(forecast_datetime_indices, actual_targets_original, label='Observed GPP', color='blue', marker='o')
plt.plot(forecast_datetime_indices, predicted_targets_original, label='Predicted GPP', linestyle='--', color='red', marker='x')
plt.title('Comparison of Predicted and Observed GPP')
plt.xlabel('Date')
plt.ylabel('GPP Value')
plt.legend()
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability

# Save the figure in the 'Figures' subdirectory with the filename 'GPP_forecast.png'
plt.savefig(os.path.join(figures_directory, 'GPP_forecast.png'))

# Display the plot
plt.show()