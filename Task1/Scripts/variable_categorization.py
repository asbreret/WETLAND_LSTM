def summarize_variables(metadata, columns, qualifiers):
    """
    Summarizes variable availability and details based on DataFrame columns and qualifier priorities.

    Parameters:
    - metadata: Dict containing variable metadata like description and unit.
    - columns: List of column names available in the DataFrame.
    - qualifiers: Dict containing qualifier priorities and descriptions.

    Prints a summary table of variables with their description and unit, only if they are present in the DataFrame columns.
    """
    print("")
    print(f"{'Variable':<20} | {'Status':<20} | {'Description':<40} | {'Unit':<15}")
    print("-" * 105)
    
    for var, attr in metadata.items():
        for qual, info in sorted(qualifiers.items(), key=lambda item: item[1]['priority']):
            qualified = f"{var}{qual}"
            if qualified in columns:
                # Ensure the description is suitably truncated for display
                description = attr['description'][:37] + '...' if len(attr['description']) > 40 else attr['description']
                status = info['description']  # Using the description from qualifier metadata as status
                
                print(f"{qualified:<20} | {status:<20} | {description:<40} | {attr['unit']:<15}")
                break  # Found a match, no need to check further qualifiers for this variable


def summarize_ecosystem_variables(ecosystem_variables, columns):
    """
    Summarizes ecosystem variable availability and details based on DataFrame columns.

    Parameters:
    - ecosystem_variables: Dict containing ecosystem variables' metadata including names, descriptions, and units.
    - columns: List of column names available in the DataFrame.

    Prints a summary table of available variables with their description and unit.
    """
    print("")
    print(f"{'Variable':<30} | {'Description':<50} | {'Unit':<15}")
    print("-" * 100)

    for var, attr in ecosystem_variables.items():
        if var in columns:  # Check if the variable is present in the DataFrame columns
            # Ensure the description is suitably truncated for display
            description = (attr['description'][:47] + '...') if len(attr['description']) > 50 else attr['description']
            # Replace 'Unknown' unit with 'N/A' for clarity
            unit = attr['unit'] if attr['unit'] != 'Unknown' else 'N/A'
            
            print(f"{var:<30} | {description:<50} | {unit:<15}")
            


def select_vars(metadata, columns, qualifiers):
    """
    Selects variables from DataFrame columns based on their qualifiers and priority.

    Parameters:
    - metadata: Dict containing variables' metadata.
    - columns: List of DataFrame column names.
    - qualifiers: Dict of qualifier priorities and descriptions.

    Returns:
    - selected: List of selected variables based on availability and priority.
    """
    selected = []
    # Iterate through each variable to find the highest-priority qualifier available
    for var in metadata:
        for qual, info in sorted(qualifiers.items(), key=lambda i: i[1]['priority']):
            qualified = f"{var}{qual}"
            if qualified in columns:
                selected.append(qualified)
                break  # Stop after finding the highest-priority match
    return selected


def select_ecosystem_vars(ecosystem_variables, columns):
    """
    Selects ecosystem variables from DataFrame columns based on generated metadata.

    Parameters:
    - ecosystem_variables: Dict containing ecosystem variables' metadata including names, descriptions, and units.
    - columns: List of DataFrame column names.

    Returns:
    - selected: List of selected ecosystem variables based on availability in the DataFrame columns.
    """
    selected = []
    # Iterate through each ecosystem variable to see if it's present in the DataFrame columns
    for var in ecosystem_variables:
        if var in columns:
            selected.append(var)  # Add the variable to the selected list if it's present

    return selected