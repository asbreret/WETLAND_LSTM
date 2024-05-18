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
from tensorflow.keras.models import load_model
from LSTM_helper import process_site_data, train_model, scale_data_3D
from pathlib import Path
import joblib


# Define the base directories where the data and models are located using relative paths
BASE_DIR = Path('../Data/Processed/Roll_noWTD')
MODEL_DIR = Path('../Data/Models')
    
# Get a list of all files in the directory
files = os.listdir(BASE_DIR)

# Filter out driver and target files and extract site codes dynamically
driver_files = [f for f in files if 'drivers.nc' in f]
target_files = [f for f in files if 'targets.nc' in f]
site_codes = {f.split('_')[0] for f in driver_files + target_files}

# Define the sequence length (weekly data with a 4-week sequence length)
SEQUENCE_LENGTH = 7 * 4

# Define training parameters
param_dict = {
    'numHiddenUnits': 100,
    'initial_learning_rate': 0.001,
    'num_layers': 1,
    'dropout_rate': 0.0,
    'batch_size': 1024,
    'num_epochs': 100,
    'learning_rate_schedule': 'exponential',
    'optimizer': 'adam',
    'early_stopping_patience': 50,
    'lstm_activation': 'tanh'
}

# Loop over each site to exclude it from the training data
for site_code in site_codes:
    all_LSTM_input_data = []
    all_LSTM_output_data = []
    
    for exclude_site_code in site_codes:
        if exclude_site_code != site_code:
            LSTM_input_data, LSTM_output_data = process_site_data(exclude_site_code, BASE_DIR, SEQUENCE_LENGTH)
            
            if LSTM_input_data.size > 0 and LSTM_output_data.size > 0:
                all_LSTM_input_data.append(LSTM_input_data)
                all_LSTM_output_data.append(LSTM_output_data)
    
    if all_LSTM_input_data and all_LSTM_output_data:
        all_LSTM_input_data = np.concatenate(all_LSTM_input_data, axis=0)
        all_LSTM_output_data = np.concatenate(all_LSTM_output_data, axis=0)
    else:
        raise ValueError("No valid LSTM input or output data available for training.")

    # Scale the input and output data
    all_LSTM_input_data, input_scaler = scale_data_3D(all_LSTM_input_data)
    all_LSTM_output_data, output_scaler = scale_data_3D(all_LSTM_output_data)


    # Save the scalers
    input_scaler_filename = MODEL_DIR / f'{site_code}_input_scaler.pkl'
    output_scaler_filename = MODEL_DIR / f'{site_code}_output_scaler.pkl'
    joblib.dump(input_scaler, input_scaler_filename)
    joblib.dump(output_scaler, output_scaler_filename)


    # Loop through each variable (GPP, RECO, NEE, FCH4)
    variables = ['GPP', 'RECO', 'NEE', 'FCH4']
    
    for i, variable in enumerate(variables):
        # Extract the output data for the current variable and retain its 3D shape
        all_LSTM_output_data_var = all_LSTM_output_data[:, :, i:i+1]

        # Train and save models
        models = []
        model_filenames = []
        N = 10  # Define the number of models to train
        
        for j in range(N):
            model_filename = os.path.join(MODEL_DIR, f'model_{site_code}_{variable}_{j+1}.h5')
            net, history = train_model(all_LSTM_input_data, all_LSTM_output_data_var, param_dict, model_filename)
#            net = load_model(model_filename)
            net.save(model_filename)
            
            models.append(load_model(model_filename))
            model_filenames.append(model_filename)

        print(f'Models trained and saved for site {site_code} and variable {variable}: {model_filenames}')
