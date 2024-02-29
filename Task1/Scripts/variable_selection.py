primary_variables_metadata = {
    'TA': {'description': 'Air temperature', 'unit': 'deg C'},
    'P': {'description': 'Precipitation', 'unit': 'mm'},
    'SW_IN': {'description': 'Shortwave radiation, incoming', 'unit': 'W m-2'},
    'LW_IN': {'description': 'Longwave radiation, incoming', 'unit': 'W m-2'},
    'SWC': {'description': 'Soil water content (volumetric), range 0-100', 'unit': '%'},
    'WS': {'description': 'Wind speed', 'unit': 'm s-1'},
    'GPP': {'description': 'Gross Primary Productivity', 'unit': 'µmolCO2 m-2 s-1'},
    'NEE': {'description': 'Net Ecosystem Exchange', 'unit': 'µmolCO2 m-2 s-1'},
    'RECO': {'description': 'Ecosystem Respiration', 'unit': 'µmolCO2 m-2 s-1'},
    'VPD': {'description': 'Vapor Pressure Deficit', 'unit': 'hPa'},
    'PA': {'description': 'Atmospheric pressure', 'unit': 'kPa'},
    'RH': {'description': 'Relative humidity, range 0-100', 'unit': '%'},
    'T_SONIC': {'description': 'Sonic temperature', 'unit': 'deg C'},
    'PPFD_IN': {'description': 'Photosynthetic photon flux density, incoming', 'unit': 'µmolPhoton m-2 s-1'},
    'TAU': {'description': 'Momentum flux', 'unit': 'kg m-1 s-2'},
    'ALB': {'description': 'Albedo, range 0-100', 'unit': '%'},
    'MCRI': {'description': 'Carotenoid Reflectance Index (Gitelson et al., 2002)', 'unit': 'nondimensional'},
    'MTCI': {'description': 'Meris Terrestrial Chlorophyll Index (Dash and Curran, 2004)', 'unit': 'nondimensional'},
    'CO2': {'description': 'Carbon Dioxide (CO2) mole fraction in wet air', 'unit': 'µmolCO2 mol-1'},
    'H2O': {'description': 'Water (H2O) vapor in mole fraction of wet air', 'unit': 'mmolH2O mol-1'},
    'SWP': {'description': 'Soil water potential', 'unit': 'kPa'},
    'TS': {'description': 'Soil temperature', 'unit': 'deg C'},
    'TSN': {'description': 'Snow temperature', 'unit': 'deg C'},
    'WTD': {'description': 'Water table depth', 'unit': 'm'},
    # AQUATIC
    'COND_WATER': {'description': 'Conductivity (i.e., electrical conductivity) of water', 'unit': 'µS cm-1'},
    'DO': {'description': 'Dissolved oxygen in water', 'unit': 'µmol L-1'},
    'PCH4': {'description': 'Dissolved methane (CH4) in water', 'unit': 'nmolCH4 mol-1'},
    'PCO2': {'description': 'Dissolved carbon dioxide (CO2) in water', 'unit': 'µmolCO2 mol-1'},
    'PN2O': {'description': 'Dissolved nitrous oxide (N2O) in water', 'unit': 'nmolN2O mol-1'},
    'PPFD_UW_IN': {'description': 'Photosynthetic photon flux density, underwater, incoming', 'unit': 'µmolPhotons m-2 s-1'},
    'TW': {'description': 'Water temperature', 'unit': 'deg C'},
    # GASES
    'CH4': {'description': 'Methane (CH4) mole fraction in wet air', 'unit': 'nmolCH4 mol-1'},
    'CH4_MIXING_RATIO': {'description': 'Methane (CH4) in mole fraction of dry air', 'unit': 'nmolCH4 mol-1'},
    'CO': {'description': 'Carbon Monoxide (CO) mole fraction in wet air', 'unit': 'nmolCO mol-1'},
    'CO2_MIXING_RATIO': {'description': 'Carbon Dioxide (CO2) in mole fraction of dry air', 'unit': 'µmolCO2 mol-1'},
    'CO2_SIGMA': {'description': 'Standard deviation of carbon dioxide mole fraction in wet air', 'unit': 'µmolCO2 mol-1'},
    'CO2C13': {'description': 'Stable isotopic composition of CO2 - C13 (i.e., d13C of CO2)', 'unit': '‰ (permil)'},
    'FC': {'description': 'Carbon Dioxide (CO2) turbulent flux (no storage correction)', 'unit': 'µmolCO2 m-2 s-1'},
    'FCH4': {'description': 'Methane (CH4) turbulent flux (no storage correction)', 'unit': 'nmolCH4 m-2 s-1'},
    'FN2O': {'description': 'Nitrous oxide (N2O) turbulent flux (no storage correction)', 'unit': 'nmolN2O m-2 s-1'},
    'FNO': {'description': 'Nitric oxide (NO) turbulent flux (no storage correction)', 'unit': 'nmolNO m-2 s-1'},
    'FNO2': {'description': 'Nitrogen dioxide (NO2) turbulent flux (no storage correction)', 'unit': 'nmolNO2 m-2 s-1'},
    'FO3': {'description': 'Ozone (O3) turbulent flux (no storage correction)', 'unit': 'nmolO3 m-2 s-1'},
    'H2O_MIXING_RATIO': {'description': 'Water (H2O) vapor in mole fraction of dry air', 'unit': 'mmolH2O mol-1'},
    'H2O_SIGMA': {'description': 'Standard deviation of water vapor mole fraction', 'unit': 'mmolH2O mol-1'},
    'N2O': {'description': 'Nitrous Oxide (N2O) mole fraction in wet air', 'unit': 'nmolN2O mol-1'},
    'N2O_MIXING_RATIO': {'description': 'Nitrous Oxide (N2O) in mole fraction of dry air', 'unit': 'nmolN2O mol-1'},
    'NO': {'description': 'Nitric oxide (NO) mole fraction in wet air', 'unit': 'nmolNO mol-1'},
    'NO2': {'description': 'Nitrogen dioxide (NO2) mole fraction in wet air', 'unit': 'nmolNO2 mol-1'},
    'O3': {'description': 'Ozone (O3) mole fraction in wet air', 'unit': 'nmolO3 mol-1'},
    'SC': {'description': 'Carbon Dioxide (CO2) storage flux', 'unit': 'µmolCO2 m-2 s-1'},
    'SCH4': {'description': 'Methane (CH4) storage flux', 'unit': 'nmolCH4 m-2 s-1'},
    'SN2O': {'description': 'Nitrous oxide (N2O) storage flux', 'unit': 'nmolN2O m-2 s-1'},
    'SNO': {'description': 'Nitric oxide (NO) storage flux', 'unit': 'nmolNO m-2 s-1'},
    'SNO2': {'description': 'Nitrogen dioxide (NO2) storage flux', 'unit': 'nmolNO2 m-2 s-1'},
    'SO2': {'description': 'Sulfur Dioxide (SO2) mole fraction in wet air', 'unit': 'nmolSO2 mol-1'},
    'SO3': {'description': 'Ozone (O3) storage flux', 'unit': 'nmolO3 m-2 s-1'},
    
    # HEAT
    'FH2O': {'description': 'Water vapor (H2O) turbulent flux (no storage correction)', 'unit': 'mmolH2O m-2 s-1'},
    'G': {'description': 'Soil heat flux', 'unit': 'W m-2'},
    'H': {'description': 'Sensible heat turbulent flux (no storage correction)', 'unit': 'W m-2'},
    'LE': {'description': 'Latent heat turbulent flux (no storage correction)', 'unit': 'W m-2'},
    'SB': {'description': 'Heat storage flux in biomass', 'unit': 'W m-2'},
    'SG': {'description': 'Heat storage flux in the soil above the soil heat fluxes measurement', 'unit': 'W m-2'},
    'SH': {'description': 'Sensible heat (H) storage flux', 'unit': 'W m-2'},
    'SLE': {'description': 'Latent heat (LE) storage flux', 'unit': 'W m-2'}    
    
}



# Metadata dictionary for qualifiers
qualifier_metadata = {
    '_F': {'priority': 3, 'description': 'Gap-filled data'},
    '_PI': {'priority': 4, 'description': 'Provided by Principal Investigator'},
    '_QC': {'priority': 5, 'description': 'Quality controlled'},
    '_IU': {'priority': 6, 'description': 'In Use'},
    '': {'priority': 7, 'description': 'Raw data'}
}



def ecosystem_dict(df_columns):
    metadata = {}
    prefixes = ['GPP', 'RECO', 'NEE']
    
    # Initialize priorities directly in mappings
    partitioning_mapping = {'DT': {'priority': 1}, 'NT': {'priority': 2}}
    method_mapping = {'CUT': {'priority': 2}, 'VUT': {'priority': 1}}
    qualifiers_mapping = {
        'REF': {'priority': 1}, 'QC': {'priority': 2}, 'MEAN': {'priority': 3}, 'SE': {'priority': 4},
        'RANDUNC': {'priority': 5}, 'RANDUNC_METHOD': {'priority': 6}, 'RANDUNC_N': {'priority': 7},
        'JOINTUNC': {'priority': 8}, 'USTAR50': {'priority': 9}
    }

    # Adjusting partitioning priority for RECO to favor NT over DT
    partitioning_mapping['RECO'] = {'DT': {'priority': 2}, 'NT': {'priority': 1}}

    best_vars = {prefix: {'variable': '', 'priority': float('inf')} for prefix in prefixes}

    for col in df_columns:
        parts = col.split('_')
        prefix = parts[0]

        # Adjust priorities based on variable prefix (GPP, RECO, NEE)
        if prefix in prefixes:
            overall_priority = 0

            for part in parts[1:]:
                # Apply specific partitioning priority for RECO
                if prefix == 'RECO' and part in partitioning_mapping['RECO']:
                    overall_priority += partitioning_mapping['RECO'][part]['priority']
                elif part in partitioning_mapping:
                    overall_priority += partitioning_mapping[part]['priority']
                elif part in method_mapping:
                    overall_priority += method_mapping[part]['priority']
                elif part in qualifiers_mapping:
                    overall_priority += qualifiers_mapping[part]['priority']
                else:
                    overall_priority += 10  # Default priority for unmapped parts

            if overall_priority < best_vars[prefix]['priority']:
                best_vars[prefix] = {'variable': col, 'priority': overall_priority}

    # Extract the best variables into metadata
    for prefix, data in best_vars.items():
        if data['variable']:  # Check if a best variable was found
            metadata[data['variable']] = {
                'description': f"{prefix} best variable based on priority",
                'unit': "µmolCO2 m-2 s-1",
                'overall_priority': data['priority']
            }

    return metadata
