# Import necessary functions and metadata from separate modules
from data_preprocessing import load_FULLSET_data
from variable_selection import primary_variables_metadata, qualifier_metadata, ecosystem_dict
from variable_categorization import summarize_variables,summarize_ecosystem_variables, select_vars

def process_and_select_variables(file_path):
    # Load and preprocess the dataset
    df = load_FULLSET_data(file_path)
    print("Dataset loaded successfully.")
    
    # Generate dynamic ecosystem variables metadata based on DataFrame columns
    ecosystem_vars = ecosystem_dict(df.columns)
    
    # Select primary variables based on their availability and predefined priorities
    primary_selected_variables = select_vars(primary_variables_metadata, df.columns, qualifier_metadata)
    print(f"Primary variables selected: {len(primary_selected_variables)}")
    
    # Additionally, include ecosystem variables by checking their presence in df.columns
    ecosystem_selected_variables = [var for var in ecosystem_vars if var in df.columns]
    print(f"Ecosystem variables selected: {len(ecosystem_selected_variables)}")
    
    # Combine both sets of variables, ensuring uniqueness
    final_variables = list(set(primary_selected_variables + ecosystem_selected_variables))
    filtered_df = df[final_variables]
    print(f"Total variables selected and DataFrame filtered: {len(final_variables)} variables.")
    
    # Optionally, summarize the available variables within the DataFrame after filtering
    summarize_variables(primary_variables_metadata, filtered_df.columns, qualifier_metadata)
    summarize_ecosystem_variables(ecosystem_vars, final_variables)
    
    return filtered_df


# Example usage:
# file_path = 'AMF_US-EDN_FLUXNET_FULLSET_HH_2018-2019_3-5.csv'
# filtered_df = process_and_select_variables('AMF_US-Tw1_FLUXNET_FULLSET_HH_2011-2020_3-5.csv')
