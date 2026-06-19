import os
import pandas as pd
import numpy as np

raw_dir = r"data/raw"
cleaned_dir = r"data/cleaned"
os.makedirs(cleaned_dir, exist_ok=True)

print("=== STARTING GEOLOGY DATASETS CLEANING TEST ===")

# 1. USGS MRDS (minerals)
mrds_path = os.path.join(raw_dir, "mrds.csv")
if os.path.exists(mrds_path):
    print("Cleaning mrds.csv...")
    try:
        # Load the dataset with low_memory=False
        df_mrds = pd.read_csv(mrds_path, low_memory=False)
        cols = ['site_name', 'latitude', 'longitude', 'commod1', 'prod_size', 'dev_stat']
        df_mrds = df_mrds[[c for c in cols if c in df_mrds.columns]]
        df_mrds = df_mrds.dropna(subset=['latitude', 'longitude'])
        df_mrds = df_mrds.drop_duplicates()
        
        # Rename columns to standard format
        df_mrds = df_mrds.rename(columns={
            'site_name': 'Nom',
            'latitude': 'Latitude',
            'longitude': 'Longitude',
            'commod1': 'Commodite',
            'prod_size': 'Taille_Production',
            'dev_stat': 'Statut_Developpement'
        })
        
        mrds_out = os.path.join(cleaned_dir, "mrds_cleaned.csv")
        df_mrds.to_csv(mrds_out, index=False)
        print(f"  Saved {mrds_out} - {df_mrds.shape[0]} rows, {df_mrds.shape[1]} cols")
    except Exception as e:
        print(f"  Error loading mrds.csv: {e}")
else:
    print("  mrds.csv not found!")

# 2. Coal Mines
coal_path = os.path.join(raw_dir, "geologie", "Global Coal Mine Tracker, May 2026__.xlsx")
if os.path.exists(coal_path):
    print("Cleaning coal mines...")
    df_coal = pd.read_excel(coal_path, sheet_name="Non-closed mines")
    df_coal = df_coal.dropna(subset=['Latitude', 'Longitude'])
    cols_keep = {
        'GEM Mine ID': 'Mine_ID',
        'Mine Name': 'Nom',
        'Country / Area': 'Pays',
        'Status': 'Statut',
        'Capacity (Mtpa)': 'Capacite_Mtpa',
        'Production (Mtpa)': 'Production_Mtpa',
        'Coal Type': 'Type_Charbon',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude'
    }
    df_coal = df_coal[[c for c in cols_keep.keys() if c in df_coal.columns]].rename(columns=cols_keep)
    df_coal = df_coal.drop_duplicates()
    coal_out = os.path.join(cleaned_dir, "coal_mines_cleaned.csv")
    df_coal.to_csv(coal_out, index=False)
    print(f"  Saved {coal_out} - {df_coal.shape[0]} rows, {df_coal.shape[1]} cols")
else:
    print("  Coal mines tracker not found!")

# 3. Iron Mines
iron_path = os.path.join(raw_dir, "geologie", "Global-Iron-Ore-Mines-Tracker-August-2025-V1.xlsx")
if os.path.exists(iron_path):
    print("Cleaning iron mines...")
    df_iron = pd.read_excel(iron_path, sheet_name="Main Data")
    df_iron = df_iron.dropna(subset=['Coordinates'])
    
    # Parse Coordinates (format: "lat, lon")
    def parse_coords(coord_str):
        try:
            parts = str(coord_str).split(",")
            if len(parts) == 2:
                return float(parts[0].strip()), float(parts[1].strip())
        except Exception:
            pass
        return np.nan, np.nan
        
    coords = df_iron['Coordinates'].apply(parse_coords)
    df_iron['Latitude'] = [c[0] for c in coords]
    df_iron['Longitude'] = [c[1] for c in coords]
    df_iron = df_iron.dropna(subset=['Latitude', 'Longitude'])
    
    cols_keep = {
        'GEM Asset ID': 'Asset_ID',
        'Asset name (English)': 'Nom',
        'Country/Area': 'Pays',
        'Operating status': 'Statut',
        'Production 2024 (ttpa)': 'Production_2024_ttpa',
        'Design capacity (ttpa)': 'Capacite_ttpa',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude'
    }
    df_iron = df_iron[[c for c in cols_keep.keys() if c in df_iron.columns]].rename(columns=cols_keep)
    df_iron = df_iron.drop_duplicates()
    iron_out = os.path.join(cleaned_dir, "iron_mines_cleaned.csv")
    df_iron.to_csv(iron_out, index=False)
    print(f"  Saved {iron_out} - {df_iron.shape[0]} rows, {df_iron.shape[1]} cols")
else:
    print("  Iron mines tracker not found!")

# 4. Oil & Gas
oil_path = os.path.join(raw_dir, "geologie", "Global-Oil-and-Gas-Extraction-Tracker-March-2026.xlsx")
if os.path.exists(oil_path):
    print("Cleaning oil and gas...")
    df_oil = pd.read_excel(oil_path, sheet_name="Field-level main data")
    df_oil = df_oil.dropna(subset=['Latitude', 'Longitude'])
    cols_keep = {
        'Unit ID': 'Unit_ID',
        'Unit Name': 'Nom',
        'Fuel type': 'Type_Combustible',
        'Country/Area': 'Pays',
        'Production Type': 'Type_Production',
        'Status': 'Statut',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude'
    }
    df_oil = df_oil[[c for c in cols_keep.keys() if c in df_oil.columns]].rename(columns=cols_keep)
    df_oil = df_oil.drop_duplicates()
    oil_out = os.path.join(cleaned_dir, "oil_gas_cleaned.csv")
    df_oil.to_csv(oil_out, index=False)
    print(f"  Saved {oil_out} - {df_oil.shape[0]} rows, {df_oil.shape[1]} cols")
else:
    print("  Oil and gas tracker not found!")

print("=== GEOLOGY CLEANING TEST SUCCESS ===")
