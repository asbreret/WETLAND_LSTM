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


import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from LSTM_helper import process_site_data, scale_data_3D, calculate_error_metrics, plot_predictions, plot_metrics
from netcdf_tools import read_data_from_netcdf
from pathlib import Path
import joblib


# Define the subdirectory name
SUBDIR = 'Wavelet'

# Define the base directories where the data and models are located using relative paths
BASE_DIR = Path('../Data/Processed') / SUBDIR
MODEL_DIR = Path('../Data/Models') / SUBDIR
PLOT_DIR = Path('../Data/Plots') / SUBDIR

# Create plot directory if it does not exist
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Get a list of all files in the directory
files = os.listdir(BASE_DIR)

# Filter out driver and target files and extract site codes dynamically
driver_files = [f for f in files if 'drivers.nc' in f]
target_files = [f for f in files if 'targets.nc' in f]
site_codes = {f.split('_')[0] for f in driver_files + target_files}

# Define the sequence length (weekly data with a 4-week sequence length)
SEQUENCE_LENGTH = 7 * 4
variables = ['GPP', 'RECO', 'NEE', 'FCH4']
variable_indices = {v: i for i, v in enumerate(variables)}  # Make sure this order matches the training data

# Initialize accumulators for all metrics
all_yearly_metrics = pd.DataFrame()

for site_code in site_codes:
    
 
    # Load scalers
    input_scaler_filename = MODEL_DIR / f'{site_code}_input_scaler.pkl'
    output_scaler_filename = MODEL_DIR / f'{site_code}_output_scaler.pkl'
    input_scaler = joblib.load(input_scaler_filename)
    output_scaler = joblib.load(output_scaler_filename)

    all_predictions = pd.DataFrame()  # Initialize predictions for each site

    for variable in variables:
        print(f"Processing site {site_code} for variable {variable}")

        LSTM_input_data, LSTM_output_data = process_site_data(site_code, BASE_DIR, SEQUENCE_LENGTH)
        LSTM_input_data_scaled, _ = scale_data_3D(LSTM_input_data, input_scaler)

        # Load models and make predictions
        models = []
        N = 10  # Number of models to load and predict
        predictions = []
        for j in range(N):
            model_filename = MODEL_DIR / f'model_{site_code}_{variable}_{j+1}.h5'
            model = load_model(model_filename)
            models.append(model)
            predictions.append(model.predict(LSTM_input_data_scaled))

        predictions = np.array(predictions)
        predicted_mean = np.mean(predictions, axis=0)
        predicted_std = np.std(predictions, axis=0)

        # Extract the relevant mean and scale for the variable
        idx = variable_indices[variable]
        variable_mean = output_scaler.mean_[idx]
        variable_scale = output_scaler.scale_[idx]

        # Apply inverse scaling
        predicted_mean_rescaled = predicted_mean * variable_scale + variable_mean
        predicted_std_rescaled = predicted_std * variable_scale

        # Read the time data from the NetCDF file
        driver_file = [f for f in driver_files if site_code in f][0]
        time_data = read_data_from_netcdf(BASE_DIR / driver_file).index

        # Filter time_data to include only full years
        years = time_data.year
        year_counts = years.value_counts()
        full_years = year_counts[(year_counts == 365) | (year_counts == 366)].index
        time_data = time_data[time_data.year.isin(full_years)]

        min_length = min(len(time_data), predicted_mean.shape[0])
        time_data = time_data[:min_length]

        df_predictions = pd.DataFrame({
            'Date': pd.to_datetime(time_data),
            'Predicted': predicted_mean_rescaled.squeeze()[:min_length],
            'Actual': LSTM_output_data[:min_length, 0, variables.index(variable)],
            'Predicted_STD': predicted_std_rescaled.squeeze()[:min_length],
            'Variable': variable,
            'Site': site_code
        })

        df_predictions.set_index('Date', inplace=True)
        metrics = calculate_error_metrics(df_predictions)
        metrics['Variable'] = variable
        metrics['Site'] = site_code
        all_yearly_metrics = pd.concat([all_yearly_metrics, metrics])
        all_predictions = pd.concat([all_predictions, df_predictions])

    plot_predictions(all_predictions,PLOT_DIR, site_code)
    
    
# Call the function with your DataFrame
plot_metrics(all_yearly_metrics, PLOT_DIR)  