import pandas as pd
import netCDF4 as nc
import datetime
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Assuming NN_model.py contains the Wetland_NN function and hyperparameter_optimization.py contains the my_optimisation function
from NN_model import Wetland_NN
from hyperparameter_optimization import my_optimisation
from sklearn.metrics import r2_score
import os
import seaborn as sns



from sklearn.decomposition import PCA




# ----- Function Definitions -----



def plot_metrics(all_yearly_metrics, plot_dir):
    # Define the metrics and variables to plot
    metrics_to_plot = ['MAE', 'RMSE', 'R-squared', 'Pearson correlation']
    variables = all_yearly_metrics['Variable'].unique()
    sites = all_yearly_metrics['Site'].unique()

    # Use a color palette with a different color for each site
    colors = sns.color_palette('tab10', len(sites))
    color_map = dict(zip(sites, colors))

    # Create the 4x4 grid of subplots
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(24, 20))

    # Loop through each variable and each metric
    for i, variable in enumerate(variables):
        for j, metric in enumerate(metrics_to_plot):
            ax = axes[i, j]
            subset = all_yearly_metrics[all_yearly_metrics['Variable'] == variable]

            # Create an empty DataFrame to store the data to plot
            data_to_plot = pd.DataFrame(index=sites, columns=[metric])

            for site in sites:
                site_data = subset[subset['Site'] == site]
                data_to_plot.loc[site, metric] = site_data[metric].values[0]

            # Plot the metric for each site with specific colors
            data_to_plot[metric].plot(kind='bar', ax=ax, color=[color_map[site] for site in data_to_plot.index], edgecolor='black', alpha=0.7)

            # Set plot labels and titles
            ax.set_title(f'{variable} - {metric}', fontweight='bold', fontsize=14)
            ax.set_ylabel(metric)
            ax.set_xlabel('Site')
            ax.tick_params(labelrotation=45)

            # Set y-axis limit based on the metric type
            if metric in ['MAE', 'RMSE']:
                ax.set_ylim(bottom=0)
            elif metric == 'R-squared' or metric == 'Pearson correlation':
                ax.set_ylim(0, 1)

    # Set the main title for the entire figure
    plt.suptitle('Yearly Metrics Comparison Across Sites and Variables', fontweight='bold', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Create a custom legend
    custom_lines = [plt.Line2D([0], [0], color=color_map[site], lw=4) for site in sites]
    fig.legend(custom_lines, sites, loc='upper center', ncol=len(sites))
    
    # Save the plot to the specified directory
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, 'yearly_metrics_comparison.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()
    
    


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def perform_pca_on_drivers(drivers_df, N):
    """
    Perform PCA on the drivers DataFrame after scaling the data, and return the first N principal components,
    retaining the original datetime index.

    Args:
        drivers_df (DataFrame): DataFrame containing the drivers data, with a datetime index.
        N (int): Number of principal components to retain.

    Returns:
        DataFrame: A DataFrame with the first N principal components and the original datetime index.
    """
    # Scaling the data
    scaler = StandardScaler()
    drivers_scaled = scaler.fit_transform(drivers_df)
    
    # Applying PCA with n_components set to N
    pca = PCA(n_components=N)
    principalComponents = pca.fit_transform(drivers_scaled)
    
    # Generating column names based on N
    pc_columns = [f'principal component {i+1}' for i in range(N)]
    
    # Creating a DataFrame for the principal components
    # Ensuring it retains the original datetime index from drivers_df
    drivers_pca_df = pd.DataFrame(data=principalComponents, 
                                  columns=pc_columns, 
                                  index=drivers_df.index)
    
    # Optionally, print the explained variance ratio and cumulative explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()
    
    print(f"Explained variance by component: {explained_variance_ratio}")
    print(f"Cumulative explained variance: {cumulative_explained_variance}")
    
    return drivers_pca_df


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






def scale_data(features_df, target_df, scaler_features=None, scaler_target=None):
    """
    Scale features and target data frames using StandardScaler.

    Args:
        features_df (pd.DataFrame): Data frame containing the features.
        target_df (pd.DataFrame): Data frame containing the target values.
        scaler_features (StandardScaler, optional): Pre-fitted scaler for the features.
        scaler_target (StandardScaler, optional): Pre-fitted scaler for the target.

    Returns:
        tuple: (scaled features DataFrame, scaled target DataFrame, scaler for features, scaler for target)
    """
    # Scale features
    if scaler_features is None:
        scaler_features = StandardScaler()
        features_scaled = pd.DataFrame(scaler_features.fit_transform(features_df),
                                       columns=features_df.columns, index=features_df.index)
    else:
        features_scaled = pd.DataFrame(scaler_features.transform(features_df),
                                       columns=features_df.columns, index=features_df.index)

    # Scale target
    if scaler_target is None:
        scaler_target = StandardScaler()
        target_scaled = pd.DataFrame(scaler_target.fit_transform(target_df),
                                     columns=target_df.columns, index=target_df.index)
    else:
        target_scaled = pd.DataFrame(scaler_target.transform(target_df),
                                     columns=target_df.columns, index=target_df.index)

    return features_scaled, target_scaled, scaler_features, scaler_target




def prepare_LSTM_data(features_df, target_df, sequence_length, years):
    """
    Prepare LSTM input and output data for specified years, taking into account the sequence_length
    that may require data from the end of the previous year to complete sequences at the start of a specified year.
    """
    if isinstance(years, int):
        years = [years]  # Ensure years is always a list
    
    concatenated_input_data = []
    concatenated_output_data = []
    
    for year in years:
        # No need to extend the dates into the next year
        start_date = pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=sequence_length - 1)
        end_date = pd.Timestamp(year=year, month=12, day=31)
        
        filtered_features_df = features_df.loc[start_date:end_date]
        filtered_target_df = target_df.loc[start_date:end_date]
        
        year_input_data = []
        year_output_data = []
        
        for i in range(sequence_length - 1, len(filtered_features_df)):
            sequence_input = filtered_features_df.iloc[i - (sequence_length - 1):i + 1].values
            sequence_output = filtered_target_df.iloc[i].values.reshape(1, -1)
            
            year_input_data.append(sequence_input)
            year_output_data.append(sequence_output)
        
        if year_input_data:
            concatenated_input_data.append(np.array(year_input_data))
            concatenated_output_data.append(np.array(year_output_data))
    
    LSTM_input_data = np.concatenate(concatenated_input_data, axis=0) if concatenated_input_data else np.array([])
    LSTM_output_data = np.concatenate(concatenated_output_data, axis=0) if concatenated_output_data else np.array([])
    
    return LSTM_input_data, LSTM_output_data







def prepare_LSTM_data_by_dates(features_df, target_df, sequence_length, start_date, end_date):
    """
    Prepare LSTM input and output data for a specified date range, adjusting the start_date
    backwards by sequence_length - 1 days to ensure complete sequences from the beginning.
    
    Args:
        features_df (pd.DataFrame): Scaled features DataFrame with a DateTimeIndex.
        target_df (pd.DataFrame): Scaled target DataFrame with a DateTimeIndex.
        sequence_length (int): Number of days to look back for each sequence.
        start_date (str or pd.Timestamp): Start date for the data preparation.
        end_date (str or pd.Timestamp): End date for the data preparation.
    
    Returns:
        np.array: 3D array of LSTM input sequences for the specified date range.
        np.array: 3D array of LSTM output sequences for the specified date range.
    """
    # Convert start_date and end_date to pd.Timestamp if they are strings
    start_date = pd.Timestamp(start_date) if isinstance(start_date, str) else start_date
    end_date = pd.Timestamp(end_date) if isinstance(end_date, str) else end_date
    
    # Adjust the start_date backwards to account for the sequence length
    adjusted_start_date = start_date - pd.Timedelta(days=sequence_length - 1)
    
    # Filter the data for the adjusted date range
    filtered_features_df = features_df.loc[adjusted_start_date:end_date]
    filtered_target_df = target_df.loc[adjusted_start_date:end_date]
    
    input_data = []
    output_data = []
    
    for i in range(sequence_length - 1, len(filtered_features_df)):
        sequence_input = filtered_features_df.iloc[i - (sequence_length - 1):i + 1].values
        if filtered_features_df.index[i] <= end_date:
            sequence_output = filtered_target_df.iloc[i].values.reshape(1, -1)
            input_data.append(sequence_input)
            output_data.append(sequence_output)
    
    # Convert to numpy arrays
    LSTM_input_data = np.array(input_data)
    LSTM_output_data = np.array(output_data)
    
    return LSTM_input_data, LSTM_output_data



def plot_all_variables(all_predictions):
    # Extract unique variables
    variables = all_predictions['Variable'].unique()
    
    # Set up the figure for subplots, using 4 rows and 2 columns
    fig, axes = plt.subplots(5, 2, figsize=(40, 20), dpi=200)
    
    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    # Prepare additional ecosystem prediction for CO2 if necessary
    if 'GPP' in variables and 'RECO' in variables and 'CO2' in variables:
        # Get predictions for GPP and RECO to calculate ecosystem prediction
        gpp_pred = all_predictions[all_predictions['Variable'] == 'GPP']['Predicted'].values
        reco_pred = all_predictions[all_predictions['Variable'] == 'RECO']['Predicted'].values
        # Create a temporary DataFrame for the ecosystem prediction
        ecosystem_pred = reco_pred - gpp_pred
        # Add ecosystem prediction to the main DataFrame for CO2
        all_predictions.loc[all_predictions['Variable'] == 'CO2', 'Ecosystem Prediction'] = ecosystem_pred

    for i, variable in enumerate(variables):
        # Get the current axes for plotting
        ax_pred = axes[i * 2]     # For prediction plot
        ax_cumul = axes[i * 2 + 1]  # For cumulative plot
        
        # Filter the DataFrame for the current variable
        df_plot = all_predictions[all_predictions['Variable'] == variable]

        # Plot actual vs predicted values
        ax_pred.plot(df_plot.index, df_plot['Predicted'], label='Predicted', color='blue')
        ax_pred.plot(df_plot.index, df_plot['Actual'], label='Actual', color='red')
        if variable == 'CO2' and 'Ecosystem Prediction' in df_plot.columns:
            ax_pred.plot(df_plot.index, df_plot['Ecosystem Prediction'], label='Ecosystem Prediction', color='green')
        ax_pred.set_title(f'{variable} Predictions vs Actual', fontsize=14, fontweight='bold')
        ax_pred.set_xlabel('Date')
        ax_pred.set_ylabel('Value')
        ax_pred.legend()

        # Yearly cumulative sum for the second subplot
        df_yearly_cumulative = df_plot[['Predicted', 'Actual']].groupby(df_plot.index.year).cumsum()
        ax_cumul.plot(df_yearly_cumulative.index, df_yearly_cumulative['Predicted'], label='Cumulative Predicted', color='blue')
        ax_cumul.plot(df_yearly_cumulative.index, df_yearly_cumulative['Actual'], label='Cumulative Actual', color='red')
        if variable == 'CO2' and 'Ecosystem Prediction' in df_plot.columns:
            eco_cumul = df_plot[['Ecosystem Prediction']].groupby(df_plot.index.year).cumsum()
            ax_cumul.plot(eco_cumul.index, eco_cumul['Ecosystem Prediction'], label='Cumulative Ecosystem Prediction', color='green')
        ax_cumul.set_title(f'{variable} Yearly Cumulative Sum', fontsize=14, fontweight='bold')
        ax_cumul.set_xlabel('Year')
        ax_cumul.set_ylabel('Cumulative Sum')
        ax_cumul.legend()

        # Calculate and annotate year-end percentage difference
        # for both original and ecosystem predictions if applicable
        df_year_end = df_plot.resample('A').sum()
        if 'Predicted' in df_year_end.columns and 'Actual' in df_year_end.columns:
            for year_index, values in df_year_end.iterrows():
                year = year_index.year  # Extract the year from the index
                predicted = values['Predicted']
                actual = values['Actual']
                if predicted != 0:  # Avoid division by zero
                    percentage_change = ((actual - predicted) / predicted) * 100
                    position = pd.Timestamp(year, 12, 31)  # Use the last day of the year for the position
                    ax_cumul.annotate(f'{percentage_change:+.2f}%',
                                      xy=(position, actual),
                                      xytext=(0,5), 
                                      textcoords="offset points",
                                      ha='center', 
                                      fontsize=10,
                                      fontweight='bold',
                                      arrowprops=dict(arrowstyle="->", color='black'))

                # Ecosystem prediction percentage difference for CO2
                if variable == 'CO2' and 'Ecosystem Prediction' in values:
                    ecosystem_predicted = values['Ecosystem Prediction']
                    if ecosystem_predicted != 0:  # Avoid division by zero
                        ecosystem_percentage_change = ((actual - ecosystem_predicted) / ecosystem_predicted) * 100
                        # Adjust position slightly for visibility
                        position_eco = pd.Timestamp(year, 12, 15)  # Use mid-December for position
                        ax_cumul.annotate(f'Eco: {ecosystem_percentage_change:+.2f}%',
                                          xy=(position_eco, actual),
                                          xytext=(0,15),
                                          textcoords="offset points",
                                          ha='center',
                                          fontsize=10,
                                          fontweight='bold',
                                          arrowprops=dict(arrowstyle="->", color='green'))
                        
    plt.tight_layout()
    plt.show()


# Example usage:
    # plot_all_variables(all_predictions)



def plot_predictions(df_plot, plot_dir, site_code):
    """
    Plot Predicted, Actual data with standard deviation bounds and include a subplot
    for the yearly cumulative sum for each variable in df_plot, covering both Predicted and Actual data,
    and include NEE_VUT_REF as RECO_PI_F - GPP_PI_F.
    
    Parameters:
        df_plot (DataFrame): DataFrame containing data to be plotted, indexed by date.
        plot_dir (Path): Directory where the plot will be saved.
        site_code (str): Site code used in the plot filename.
    """
    variables = df_plot['Variable'].unique()  # Extract unique variables
    num_variables = len(variables)
    fig, axes = plt.subplots(num_variables, 2, figsize=(30, 6 * num_variables), dpi=100)  # Adjust layout size

    if num_variables == 1:
        axes = np.array([axes]).reshape(-1, 2)  # Make it 2D array even for a single variable

    for i, variable in enumerate(variables):
        data = df_plot[df_plot['Variable'] == variable]
        ax = axes[i, 0]  # First column for Predicted vs. Actual
        ax.plot(data.index, data['Predicted'], label='Predicted', color='blue')
        ax.fill_between(data.index, data['Predicted'] - data['Predicted_STD'], data['Predicted'] + data['Predicted_STD'],
                        color='blue', alpha=0.2, label='Predicted ± STD')
        ax.scatter(data.index, data['Actual'], label='Actual', color='red', alpha=0.5)
        ax.set_title(f'Predictions vs Actual for {variable}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Values')
        ax.legend()
        ax.grid(True)

        # Second column for cumulative sum
        ax_right = axes[i, 1]
        df_yearly_cumulative_predicted = data['Predicted'].groupby(data.index.year).cumsum()
        df_yearly_cumulative_actual = data['Actual'].groupby(data.index.year).cumsum()
        ax_right.plot(df_yearly_cumulative_predicted.index, df_yearly_cumulative_predicted, label='Yearly Cumulative Predicted', color='blue')
        ax_right.plot(df_yearly_cumulative_actual.index, df_yearly_cumulative_actual, label='Yearly Cumulative Actual', color='red')

        # Special handling for NEE_VUT_REF
        if variable == 'NEE_VUT_REF':
            reco_data = df_plot[df_plot['Variable'] == 'RECO_PI_F']['Predicted']
            gpp_data = df_plot[df_plot['Variable'] == 'GPP_PI_F']['Predicted']
            nee_vut_ref_pred = reco_data.values - gpp_data.values
            ax.plot(data.index, nee_vut_ref_pred, label='Predicted NEE_VUT_REF', color='green')

            # Cumulative plots for RECO_PI_F and GPP_PI_F are already annual cumulative, sum them directly
            reco_cumulative = df_plot[df_plot['Variable'] == 'RECO_PI_F']['Predicted'].groupby(data.index.year).cumsum()
            gpp_cumulative = df_plot[df_plot['Variable'] == 'GPP_PI_F']['Predicted'].groupby(data.index.year).cumsum()
            nee_vut_ref_cumulative = reco_cumulative - gpp_cumulative
            ax_right.plot(nee_vut_ref_cumulative.index, nee_vut_ref_cumulative, label='Cumulative NEE_VUT_REF', color='green')
            ax_right.legend()

        ax_right.set_title(f'Yearly Cumulative Values for {variable}')
        ax_right.set_xlabel('Year')
        ax_right.set_ylabel('Cumulative Values')
        ax_right.grid(True)
        
        # Create a twin axis for the right subplot to plot Predicted_STD
        ax_right2 = ax_right.twinx()
        ax_right2.plot(data.index, data['Predicted_STD'], label='Predicted STD', color='black', linestyle='--')
        ax_right2.set_ylabel('Standard Deviation')
        ax_right2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(plot_dir / f'{site_code}_predictions.png')  # Save the plot
    plt.show()  # Display the plot
    
    




# Model training and saving function
def train_model(LSTM_input_data, LSTM_output_data, param_dict, model_filename):
    
    # split = 0.2   - standard
    split = 0.2
    # Splitting the dataset into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(LSTM_input_data, LSTM_output_data, test_size=split)
    
    # Train the model
    net, history = Wetland_NN(X_train, X_val, Y_train, Y_val, **param_dict)
    
    # Save the model
    #net.save(model_filename)
    
    return net, history










def optimize_train_save_model(LSTM_input_data, LSTM_output_data, model_filename):
    """
    Optimizes hyperparameters, trains, and saves an LSTM model.
    
    Args:
        LSTM_input_data (np.array): Input data for the LSTM model.
        LSTM_output_data (np.array): Output data for the LSTM model.
        model_filename (str): Filename to save the trained model.
    """
    # Splitting the dataset into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(LSTM_input_data, LSTM_output_data, test_size=0.2)
    
    # Step 1: Run Hyperparameter Optimization
    # Note: Ensure that my_optimisation is correctly defined to work with your dataset
    # You might need to adjust the function signature or the way it's called based on your data preparation
    best_params, all_runs_details = my_optimisation(X_train, X_val, Y_train, Y_val)
    
    # Step 2: Use the best parameters from optimization to define param_dict for training
    param_dict = {
        'numHiddenUnits': int(best_params['num_hidden_units']),
        'initial_learning_rate': best_params['initial_learning_rate'],
        'num_layers': int(best_params['num_layers']),
        'dropout_rate': best_params['dropout_rate'],
        'batch_size': int(best_params['batch_size']),
        'num_epochs': 600,  # or adjust based on your optimization results/preferences
        'learning_rate_schedule': 'exponential',  # Assuming this is fixed
        'optimizer': 'adam',  # Assuming this is fixed
        'early_stopping_patience': 50  # Assuming this is fixed
    }
    
    # Print the param_dict to display the optimized parameters
    print("Optimized parameters for training:")
    for param, value in param_dict.items():
        print(f"{param}: {value}")
    
    
    # Step 3: Train the model with optimized parameters
    net, history = Wetland_NN(X_train, X_val, Y_train, Y_val, **param_dict)
    
    # Step 4: Save the model
    net.save(model_filename)
    
    return net, history






def calculate_error_metrics(df_plot):
    """
    Calculate error metrics and other metrics for the 'Actual' and 'Predicted' columns in df_plot.

    Parameters:
        df_plot (DataFrame): DataFrame containing data for analysis.

    Returns:
        DataFrame: DataFrame containing error metrics and other metrics for the 'Actual' and 'Predicted' columns.
    """
    # Calculate errors
    errors = df_plot['Actual'] - df_plot['Predicted']

    # Calculate absolute errors
    absolute_errors = np.abs(errors)

    # Calculate squared errors
    squared_errors = errors ** 2

    # Calculate error metrics
    mae = absolute_errors.mean()  # Mean Absolute Error (MAE)
    mse = squared_errors.mean()   # Mean Squared Error (MSE)
    rmse = np.sqrt(mse)           # Root Mean Squared Error (RMSE)
    mape = (absolute_errors / df_plot['Actual']).mean() * 100  # Mean Absolute Percentage Error (MAPE)

    # Calculate R-squared (coefficient of determination)
    r_squared = r2_score(df_plot['Actual'], df_plot['Predicted'])

    # Calculate Pearson correlation coefficient
    correlation_coefficient = df_plot['Actual'].corr(df_plot['Predicted'])

    # Create a dictionary of metrics
    metrics_dict = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R-squared': r_squared,
        'Pearson correlation': correlation_coefficient
    }

    # Convert dictionary to DataFrame
    metrics_df = pd.DataFrame(metrics_dict, index=['Metrics'])

    return metrics_df




def optimize_and_train_model(LSTM_input_data, LSTM_output_data):
    # Define hyperparameters ranges
    hidden_units_options = range(40, 201,20)  # Example range for hidden units
    num_layers_options = [1,2]  # Example for the number of layers
    learning_rate_options = [0.001]  # Example for learning rates
    
    results = []

    for num_hidden_units in hidden_units_options:
        for num_layers in num_layers_options:
            for learning_rate in learning_rate_options:
                # Construct parameter dictionary
                param_dict = {
                    'numHiddenUnits': num_hidden_units,
                    'num_layers': num_layers,
                    'initial_learning_rate': learning_rate,
                    'dropout_rate': 0.0,
                    'batch_size': 512,
                    'num_epochs': 600,
                    'learning_rate_schedule': 'exponential',
                    'optimizer': 'adam',
                    'early_stopping_patience': 50
                }

                # Split dataset into training and validation sets (80/20 split)
                X_train, X_val, Y_train, Y_val = train_test_split(LSTM_input_data, LSTM_output_data, test_size=0.2, random_state=42)

                # Train the model
                net, history = train_model(X_train, Y_train, param_dict, "temporary_model.h5")  # Use a temporary filename or manage filenames as needed

                # Collect training and validation losses
                train_loss = history.history['loss']
                val_loss = history.history['val_loss']
                
                # Store results including the final loss values and all parameters
                results.append({
                    'num_hidden_units': num_hidden_units,
                    'num_layers': num_layers,
                    'initial_learning_rate': learning_rate,
                    'dropout_rate': param_dict['dropout_rate'],
                    'batch_size': param_dict['batch_size'],
                    'num_epochs': param_dict['num_epochs'],
                    'learning_rate_schedule': param_dict['learning_rate_schedule'],
                    'optimizer': param_dict['optimizer'],
                    'early_stopping_patience': param_dict['early_stopping_patience'],
                    'train_loss': train_loss[-1],
                    'val_loss': val_loss[-1]
                })
                
    results_df = pd.DataFrame(results)

    return results_df






def plot_training(df):
    # Find the index of the row with the best (minimum) val_loss
    best_val_loss_index = df['val_loss'].idxmin()
    
    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5 + 0.5))  # Adjust the figure size as necessary
    
    # Remove axes
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center', 
                     loc='center',
                     rowLoc='center')
    
    # Adjust header color, optional
    for j, col in enumerate(df.columns):
        table[(0, j)].set_facecolor("palegreen")
        table[(0, j)].set_text_props(weight='bold')
    
    # Highlight the best val_loss row
    for j in range(len(df.columns)):
        table[(best_val_loss_index + 1, j)].set_facecolor("lightgreen")  # +1 to account for header row
        table[(best_val_loss_index + 1, j)].set_text_props(weight='bold')
    
    plt.show()





def full_years_with_buffer(df, sequence_length):
    """
    Determine which years have full data with a trailing sequence_length day buffer.
    """
    full_years = []
    for year in range(df.index.year.min(), df.index.year.max() + 1):
        start_date = pd.Timestamp(year=year, month=1, day=1) - pd.Timedelta(days=sequence_length - 1)
        end_date = pd.Timestamp(year=year, month=12, day=31)
        
        if start_date in df.index and end_date in df.index:
            full_years.append(year)
    
    return full_years

def process_site_data(site_code, base_dir, sequence_length):
    """
    Process data for a single site, returning the LSTM input and output data.
    """
    driver_file = os.path.join(base_dir, f'{site_code}_drivers.nc')
    target_file = os.path.join(base_dir, f'{site_code}_targets.nc')
    
    drivers_df = read_data_from_netcdf(driver_file) if os.path.exists(driver_file) else None
    targets_df = read_data_from_netcdf(target_file) if os.path.exists(target_file) else None
    
    if drivers_df is None or targets_df is None:
        return None, None
    
    common_dates = drivers_df.index.intersection(targets_df.index)
    drivers_df = drivers_df.loc[common_dates]
    targets_df = targets_df.loc[common_dates]
    
    full_years = full_years_with_buffer(drivers_df, sequence_length)
    print(f"Site {site_code} has full data for years: {full_years}")
    
    return prepare_LSTM_data(drivers_df, targets_df, sequence_length, full_years)



def scale_data_3D(data, scaler=None):
    """
    Scale the data by averaging over the sequence length dimension and applying StandardScaler.
    If a scaler is provided, use it to transform the data. Otherwise, fit a new scaler.
    
    Args:
    - data (numpy array): The 3D data to be scaled.
    - scaler (StandardScaler, optional): A pre-fitted scaler to use for transformation.
    
    Returns:
    - scaled_data (numpy array): The scaled data.
    - scaler (StandardScaler): The scaler used for transformation.
    """
    # Averaging over the sequence length dimension
    avg_data = np.mean(data, axis=1)

    # If a scaler is provided, use it; otherwise, create and fit a new one
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(avg_data)

    # Transform the original sequences using the scaler
    scaled_data = np.array([scaler.transform(seq) for seq in data])
    
    return scaled_data, scaler
