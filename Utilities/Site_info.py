import os
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.lines import Line2D
from mpl_toolkits.basemap import Basemap


def extract_lat_lon(code):
    """Fetch latitude and longitude from AmeriFlux for a given site code."""
    url = f'https://ameriflux.lbl.gov/sites/siteinfo/{code}'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'lxml')
    lat_lon_cell = soup.find('td', string='Lat, Long:')
    if lat_lon_cell:
        lat_lon_values = lat_lon_cell.find_next_sibling('td').text.split(',')
        return float(lat_lon_values[0].strip()), float(lat_lon_values[1].strip())
    return None, None

def extract_site_codes(directory):
    """Extract unique site codes and their data types from filenames in the specified directory."""
    site_codes = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            parts = filename.split('_')
            site_code = parts[0]
            data_type = parts[1]
            if site_code not in site_codes:
                site_codes[site_code] = {'has_base': False, 'has_fullset': False}
            if 'BASE' in data_type:
                site_codes[site_code]['has_base'] = True
            if 'FULLSET' in data_type:
                site_codes[site_code]['has_fullset'] = True
    return site_codes

def update_site_lat_lon(site_dict):
    """Update site dictionary with latitude and longitude."""
    for site_code in site_dict.keys():
        latitude, longitude = extract_lat_lon(site_code)
        site_dict[site_code].update({'latitude': latitude, 'longitude': longitude})
    return site_dict


def plot_site_data(site_dict, output_directory):
    """Plot site data on a map with a topographic relief background and color-coded site types using Basemap."""
    
    # Initialize lists for data
    sites = []
    longitudes = []
    latitudes = []
    data_types = []
    
    # Process site data for plotting
    for site, data in site_dict.items():
        sites.append(site)
        longitudes.append(data['longitude'])
        latitudes.append(data['latitude'])
        data_types.append('both' if data['has_base'] and data['has_fullset'] else 'base_only')
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({'Site': sites, 'Data Type': data_types}, 
                           geometry=gpd.points_from_xy(longitudes, latitudes))
    gdf.set_crs(epsg=4326, inplace=True)

    # Determine the latitude and longitude bounds
    min_lat = min(latitudes)
    max_lat = max(latitudes)
    min_lon = min(longitudes)
    max_lon = max(longitudes)

    # Set figure size based on the longitude and latitude span
    width =  max(5, (max_lon - min_lon) * 0.25 )  # Scale width by longitude span
    height = max(5, (max_lat - min_lat) * 0.25 )  # Scale height by latitude span
    
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Initialize Basemap with optimized parameters
    margin = 2  # Margin around the data points
    m = Basemap(projection='merc', llcrnrlat=min_lat - margin, urcrnrlat=max_lat + margin,
                llcrnrlon=min_lon - margin, urcrnrlon=max_lon + margin, resolution='f', ax=ax)
    m.shadedrelief()  # Adjust scale to balance detail and memory usage
    
    # Convert geographic coordinates to map coordinates and plot
    map_x, map_y = m(longitudes, latitudes)
    colors = {'base_only': 'yellow', 'both': 'lime'}
    ax.scatter(map_x, map_y, c=[colors[dt] for dt in data_types], s=100)

    # Create a custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', label='Both BASE and FULLSET', markersize=10),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', label='BASE only', markersize=10)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
    
    # Setting plot title
    plt.title('Site Data Locations', fontsize='x-large', fontweight='bold')
    
    # Optimize layout
    plt.tight_layout(pad=1.0)
    
    # Save the plot
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_path = os.path.join(output_directory, 'site_map_topographic_basemap.png')
    plt.savefig(output_path, dpi=150)  # Adjusted DPI
    plt.close()


