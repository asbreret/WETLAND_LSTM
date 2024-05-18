#!/bin/bash

# Define the paths
input_dir="/mnt/c/Users/asbre/OneDrive/Desktop/LSTM_Wetland_Model/Data/Raw/Ameriflux"
output_dir="/mnt/c/Users/asbre/OneDrive/Desktop/LSTM_Wetland_Model/Data/Raw/CSV"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Function to copy files to proposed new locations
copy_to_proposed_new() {
    find "$input_dir" -type f -name "*FULLSET_HH*.csv" -exec bash -c '
        for file_path; do
            # Extract the site code directly using awk
            site_code=$(basename "$file_path" | awk -F "_" '"'"'{print $2}'"'"' | cut -d'-' -f2)
            # Construct the new filename without additional tags or dates
            new_filename="US-${site_code}_FULLSET.csv"
            # Copy the original file to the proposed new location
            cp "$file_path" "'"$output_dir"'/$new_filename"
            echo "Copied $file_path to '"$output_dir"'/$new_filename"
        done
    ' bash {} +
}

# Call the function to copy files to proposed new locations
copy_to_proposed_new

