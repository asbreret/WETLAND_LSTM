# Define the path to the directory containing the zip files
zip_dir="/mnt/c/Users/asbre/OneDrive/Desktop/LSTM_Wetland_Model/Data/Raw/Ameriflux"

# Loop through each zip file in the specified directory
for zip_file in "$zip_dir"/*.zip; do
    # Get the base name of the zip file (without the path)
    zip_file_base=$(basename "$zip_file")
    
    # Remove the .zip extension from the file name to create the directory name
    dir_name="${zip_file_base%.*}"
    
    # Create the directory in the same location as the zip files if it doesn't exist
    mkdir -p "$zip_dir/$dir_name"
    
    # Unzip the file into the created directory
    unzip -o "$zip_file" -d "$zip_dir/$dir_name"
done

