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

from Site_info import extract_site_codes, update_site_lat_lon, plot_site_data

Data_Directory      = r'C:\Users\asbre\OneDrive\Desktop\LSTM_Wetland_Model\Data\Raw\CSV'
Site_Plot_Directory = r'C:\Users\asbre\OneDrive\Desktop\LSTM_Wetland_Model\Outputs\Site_locations'
# Assuming 'site_dict' is already defined as shown in your previous message

# Extract the basic site dictionary with BASE and FULLSET data
Site_Dict = extract_site_codes(Data_Directory)

# Update the dictionary with latitude and longitude information
update_site_lat_lon(Site_Dict)

# Assuming 'site_dict' is already defined as shown in your previous message
plot_site_data(Site_Dict, Site_Plot_Directory)


