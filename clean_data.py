"""
Script de nettoyage global des datasets agricoles et physiques.
Prend en compte la nouvelle arborescence de dossiers sous data/raw
et nettoie également les nouveaux datasets géologiques et écologiques.
"""
import os
import pandas as pd
import numpy as np

raw_dir = r"data/raw"
cleaned_dir = r"data/cleaned"
os.makedirs(cleaned_dir, exist_ok=True)

# Helper function to load with correct encoding
def load_csv(filepath):
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            return pd.read_csv(filepath, encoding=enc, low_memory=False, on_bad_lines='skip')
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Could not read file {filepath} with any of the standard encodings.")

def find_col(df, *names):
    # Exact match first
    for name in names:
        if name in df.columns:
            return name
            
    # Case-insensitive match
    for name in names:
        for col in df.columns:
            if name.lower() == col.lower():
                return col
                
    # Normalize unicode/strip accents and non-alphanumeric characters for robust matching
    import unicodedata
    def norm(s):
        s = str(s).lower()
        # strip accents
        s = "".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        # strip non-alphanumeric characters (like replacement character \ufffd)
        s = "".join(c for c in s if c.isalnum())
        # common corruptions in these datasets:
        s = s.replace('element', 'lment').replace('annee', 'anne').replace('unite', 'unit')
        return s

    for name in names:
        norm_name = norm(name)
        for col in df.columns:
            if norm_name == norm(col):
                return col

    for name in names:
        norm_name = norm(name)
        for col in df.columns:
            norm_col = norm(col)
            if norm_name in norm_col or norm_col in norm_name:
                # Avoid matching a 'code' column if the search target does not have 'code'
                if 'code' in norm_col and 'code' not in norm_name:
                    continue
                return col
                
    # Fallback to the original matching logic
    for name in names:
        base = name.lower().replace('é', '').replace('è', '').replace('ê', '').replace('à', '')
        for col in df.columns:
            col_clean = col.lower().replace('é', '').replace('è', '').replace('ê', '').replace('à', '')
            if len(base) >= 4 and (base[:4] in col_clean or base[:4] in col.lower()):
                return col
    return None

def clean_common(df, value_col='Valeur'):
    df = df.dropna(subset=[value_col])
    df = df.drop_duplicates()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=[value_col])
    # clean Annee if present
    annee_col = find_col(df, 'Année', 'Annee', 'Year')
    if annee_col:
        df[annee_col] = pd.to_numeric(df[annee_col], errors='coerce')
        df = df.dropna(subset=[annee_col])
        df[annee_col] = df[annee_col].astype(int)
    return df

def save(df, name):
    filepath = os.path.join(cleaned_dir, name)
    df.to_csv(filepath, index=False, encoding='utf-8')
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  [OK] Saved {name} - {df.shape[0]} rows, {df.shape[1]} cols ({size_mb:.2f} MB)")

def main():
    # ==============================================================================
    # PHASE 1: Clean reorganized historical datasets
    # ==============================================================================
    print("=" * 80)
    print("PHASE 1: CLEANING HISTORICAL FAO/WORLD BANK DATASETS")
    print("=" * 80)

    # 1. Bilan Nutritif Sols & Terres Cultivées
    print("\nCleaning Bilan Nutritif Sols...")
    sol_path = os.path.join(raw_dir, "pedologie_sols", "sol_nutritif", "Environnement_Bilan_nutritif_des_sols_F_Toutes_les_Données_(Normalisé).csv")
    if os.path.exists(sol_path):
        df_sol = load_csv(sol_path)
        
        col_unite = find_col(df_sol, 'Unité', 'Unit')
        col_element = find_col(df_sol, 'Élément', 'Element')
        col_zone = find_col(df_sol, 'Zone', 'Area')
        col_annee = find_col(df_sol, 'Année', 'Year')
        col_valeur = find_col(df_sol, 'Valeur', 'Value')
        col_produit = find_col(df_sol, 'Produit', 'Item')
        
        # Generate Bilan Nutritif Sols (kg/ha)
        df_sols_kgha = df_sol[df_sol[col_unite] == 'kg/ha'].copy()
        cols_keep_sols = {col_zone: 'Pays', col_produit: 'Produit', col_annee: 'Annee', col_valeur: 'Valeur'}
        df_sols_kgha = df_sols_kgha[[c for c in cols_keep_sols.keys()]].copy().rename(columns=cols_keep_sols)
        df_sols_kgha = clean_common(df_sols_kgha)
        save(df_sols_kgha, "bilan_nutritif_sols.csv")
        
        # Generate Bilan Nutritif Terres Cultivées (kg/ha + %)
        df_terres = df_sol[df_sol[col_unite].isin(['kg/ha', '%'])].copy()
        cols_keep_terres = {col_zone: 'Pays', col_produit: 'Produit', col_element: 'Element', col_annee: 'Annee', col_unite: 'Unite', col_valeur: 'Valeur'}
        df_terres = df_terres[[c for c in cols_keep_terres.keys()]].copy().rename(columns=cols_keep_terres)
        df_terres = clean_common(df_terres)
        save(df_terres, "bilan_nutritif_terres_cultivees.csv")
        
        # Extract Fertilizers proxy (from Item: 'Engrais synthétiques')
        df_fert = df_sol[df_sol[col_produit].astype(str).str.contains('Engrais', case=False, na=False)].copy()
        df_fert = df_fert[df_fert[col_unite] == 'kg/ha'].copy()
        cols_keep_fert = {col_zone: 'Pays', col_produit: 'Produit', col_annee: 'Annee', col_valeur: 'Valeur'}
        df_fert = df_fert[[c for c in cols_keep_fert.keys()]].copy().rename(columns=cols_keep_fert)
        df_fert = clean_common(df_fert)
        save(df_fert, "fertilizers_nutrient.csv")
    else:
        print("  Bilan Nutritif Sols raw file not found!")

# 2. Variation Température
    print("\nCleaning Variation Température...")
    temp_path = os.path.join(raw_dir, "atmosphere_climat", "temperature", "Environnement_variation_temperature_F_Toutes_les_Données_(Normalisé).csv")
    if os.path.exists(temp_path):
        df_temp = load_csv(temp_path)
        col_zone = find_col(df_temp, 'Zone')
        col_annee = find_col(df_temp, 'Année', 'Annee', 'Year')
        col_valeur = find_col(df_temp, 'Valeur')
        col_element = find_col(df_temp, 'Élément', 'Element')
        col_mois = find_col(df_temp, 'Mois')
        col_code_mois = find_col(df_temp, 'Code Mois')
        
        cols_keep = {col_zone: 'Pays', col_code_mois: 'Code Mois', col_mois: 'Mois', col_element: 'Element', col_annee: 'Annee', col_valeur: 'Valeur'}
        df_temp = df_temp[[c for c in cols_keep.keys()]].copy().rename(columns=cols_keep)
        df_temp = clean_common(df_temp)
        save(df_temp, "variation_temperature.csv")
    else:
        print("  Variation Température raw file not found!")

# 3. Intrants / Utilisation Terres
    print("\nCleaning Intrants / Utilisation Terres...")
    terres_path = os.path.join(raw_dir, "socio_economie_demographie", "faostat", "terres_utilisation", "Intrants_TerresUtilisation_F_Toutes_les_Données_(Normalisé).csv")
    if os.path.exists(terres_path):
        df_terres_util = load_csv(terres_path)
        col_unite = find_col(df_terres_util, 'Unité', 'Unite')
        col_element = find_col(df_terres_util, 'Élément', 'Element')
        col_zone = find_col(df_terres_util, 'Zone')
        col_annee = find_col(df_terres_util, 'Année', 'Annee', 'Year')
        col_valeur = find_col(df_terres_util, 'Valeur')
        col_produit = find_col(df_terres_util, 'Produit')
        
        df_terres_util = df_terres_util[df_terres_util[col_unite].isin(['1000 ha', '%'])].copy()
        cols_keep = {col_zone: 'Pays', col_produit: 'Produit', col_element: 'Element', col_annee: 'Annee', col_unite: 'Unite', col_valeur: 'Valeur'}
        df_terres_util = df_terres_util[[c for c in cols_keep.keys()]].copy().rename(columns=cols_keep)
        df_terres_util = clean_common(df_terres_util)
        save(df_terres_util, "intrants_utilisation_terres.csv")
    else:
        print("  Intrants / Utilisation Terres raw file not found!")

# 4. Production Cultures et Animaux
    print("\nCleaning Production Cultures et Animaux...")
    prod_path = os.path.join(raw_dir, "socio_economie_demographie", "faostat", "production_agricole", "Production_Cultures_ProduitsAnimaux_F_Toutes_les_Données_(Normalisé).csv")
    if os.path.exists(prod_path):
        df_prod = load_csv(prod_path)
        col_unite = find_col(df_prod, 'Unité', 'Unite')
        col_element = find_col(df_prod, 'Élément', 'Element')
        col_zone = find_col(df_prod, 'Zone')
        col_annee = find_col(df_prod, 'Année', 'Annee', 'Year')
        col_valeur = find_col(df_prod, 'Valeur')
        col_produit = find_col(df_prod, 'Produit')
        
        mask_cultures = (
            df_prod[col_element].isin(['Production', 'Rendement', 'Superficie récoltée']) &
            df_prod[col_unite].isin(['tonnes', 'kg/ha', 'ha'])
        )
        
        cols_keep = {col_zone: 'Pays', col_produit: 'Produit', col_element: 'Element', col_annee: 'Annee', col_unite: 'Unite', col_valeur: 'Valeur'}
        
        df_cultures = df_prod[mask_cultures].copy()
        df_cultures = df_cultures[[c for c in cols_keep.keys()]].copy().rename(columns=cols_keep)
        df_cultures = clean_common(df_cultures)
        save(df_cultures, "production_cultures.csv")
        
        df_animaux = df_prod[~mask_cultures].copy()
        df_animaux = df_animaux[[c for c in cols_keep.keys()]].copy().rename(columns=cols_keep)
        df_animaux = clean_common(df_animaux)
        save(df_animaux, "production_animaux.csv")
    else:
        print("  Production raw file not found!")

# 5. Pesticides
    print("\nCleaning Pesticides...")
    pest_path = os.path.join(raw_dir, "socio_economie_demographie", "faostat", "pesticides", "Environnement_Pesticides_F_Toutes_les_Données_(Normalisé).csv")
    if os.path.exists(pest_path):
        df_pest = load_csv(pest_path)
        col_zone = find_col(df_pest, 'Zone')
        col_annee = find_col(df_pest, 'Année', 'Year')
        col_valeur = find_col(df_pest, 'Valeur')
        col_produit = find_col(df_pest, 'Produit')
        col_element = find_col(df_pest, 'Élément', 'Element')
        
        # Filter use per area of cropland if present
        df_pest = df_pest[df_pest[col_element].astype(str).str.contains('surface|capita|habit', case=False, na=False)].copy()
        cols_keep = {col_zone: 'Pays', col_produit: 'Produit', col_annee: 'Annee', col_valeur: 'Valeur'}
        df_pest = df_pest[[c for c in cols_keep.keys()]].copy().rename(columns=cols_keep)
        df_pest = clean_common(df_pest)
        save(df_pest, "pesticides.csv")
    else:
        print("  Pesticides raw file not found!")

    # ==============================================================================
    # PHASE 2: Clean newly downloaded physical/geological datasets
    # ==============================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: CLEANING NEW PHYSICAL & GEOLOGICAL DATASETS")
    print("=" * 80)

    # 1. Volcanoes
    print("\nCleaning volcanoes.csv...")
    volc_path = os.path.join(raw_dir, "geologie", "volcanoes.csv")
    if os.path.exists(volc_path):
        df_v = load_csv(volc_path)
        # Drop rows with missing lat/lon
        df_v = df_v.dropna(subset=['latitude', 'longitude'])
        # Drop useless cols
        cols_to_keep = ['volcano_number', 'volcano_name', 'primary_volcano_type', 
                        'last_eruption_year', 'latitude', 'longitude', 'elevation', 'population_within_100_km']
        df_v = df_v[[c for c in cols_to_keep if c in df_v.columns]].copy()
        
        # Handle elevation outliers
        df_v['elevation'] = pd.to_numeric(df_v['elevation'], errors='coerce')
        df_v = df_v[(df_v['elevation'] >= -5000) & (df_v['elevation'] <= 7000)]
        
        # Convert last_eruption_year to numeric and replace unknown/ancient values
        df_v['last_eruption_year'] = pd.to_numeric(df_v['last_eruption_year'], errors='coerce')
        save(df_v, "volcanoes_cleaned.csv")
    else:
        print("  volcanoes.csv not found!")

    # 2. Earthquakes
    print("\nCleaning earthquakes_usgs.csv...")
    eq_path = os.path.join(raw_dir, "geologie", "earthquakes_usgs.csv")
    if os.path.exists(eq_path):
        df_eq = load_csv(eq_path)
        df_eq = df_eq.dropna(subset=['latitude', 'longitude', 'mag'])
        
        # Only keep essential columns
        cols_to_keep = ['time', 'latitude', 'longitude', 'depth', 'mag', 'place', 'type']
        df_eq = df_eq[[c for c in cols_to_keep if c in df_eq.columns]].copy()
        
        # Filter outliers
        df_eq['mag'] = pd.to_numeric(df_eq['mag'], errors='coerce')
        df_eq['depth'] = pd.to_numeric(df_eq['depth'], errors='coerce')
        df_eq = df_eq[(df_eq['mag'] > 0) & (df_eq['mag'] <= 10)]
        df_eq = df_eq[(df_eq['depth'] >= 0) & (df_eq['depth'] <= 800)]
        
        save(df_eq, "earthquakes_cleaned.csv")
    else:
        print("  earthquakes_usgs.csv not found!")

    # 3. Global Power Plants
    print("\nCleaning global_power_plants.csv...")
    pp_path = os.path.join(raw_dir, "geologie", "global_power_plants.csv")
    if os.path.exists(pp_path):
        df_pp = load_csv(pp_path)
        df_pp = df_pp.dropna(subset=['latitude', 'longitude', 'capacity_mw', 'primary_fuel'])
        
        # Only keep key columns
        cols_to_keep = ['country_long', 'name', 'capacity_mw', 'latitude', 'longitude', 'primary_fuel', 'commissioning_year']
        df_pp = df_pp[[c for c in cols_to_keep if c in df_pp.columns]].copy()
        
        # Handle outliers
        df_pp['capacity_mw'] = pd.to_numeric(df_pp['capacity_mw'], errors='coerce')
        df_pp = df_pp[df_pp['capacity_mw'] > 0]
        
        df_pp['commissioning_year'] = pd.to_numeric(df_pp['commissioning_year'], errors='coerce')
        df_pp.loc[(df_pp['commissioning_year'] < 1800) | (df_pp['commissioning_year'] > 2026), 'commissioning_year'] = np.nan
        
        save(df_pp, "global_power_plants_cleaned.csv")
    else:
        print("  global_power_plants.csv not found!")

    # 4. Wood Density
    print("\nCleaning wood_density.csv...")
    wd_path = os.path.join(raw_dir, "ecologie_biomasse", "wood_density.csv")
    if os.path.exists(wd_path):
        df_wd = pd.read_csv(wd_path, sep=';', encoding='latin-1')
        
        for col in ['Latitude', 'Longitude', 'Wood Density']:
            if col in df_wd.columns:
                df_wd[col] = df_wd[col].astype(str).str.replace(',', '.')
                df_wd[col] = pd.to_numeric(df_wd[col], errors='coerce')
                
        df_wd = df_wd.dropna(subset=['Latitude', 'Longitude', 'Wood Density'])
        
        # Keep only target columns
        cols_to_keep = ['Species', 'Latitude', 'Longitude', 'Plant Growth Form', 'Wood Density']
        df_wd = df_wd[[c for c in cols_to_keep if c in df_wd.columns]].copy()
        
        # Handle wood density outliers
        df_wd = df_wd[(df_wd['Wood Density'] >= 0.1) & (df_wd['Wood Density'] <= 1.5)]
        
        save(df_wd, "wood_density_cleaned.csv")
    else:
        print("  wood_density.csv not found!")

    # 5. USGS MRDS Minerals
    print("\nCleaning mrds.csv...")
    mrds_path = os.path.join(raw_dir, "mrds.csv")
    if os.path.exists(mrds_path):
        df_mrds = pd.read_csv(mrds_path, low_memory=False)
        cols = ['site_name', 'latitude', 'longitude', 'commod1', 'prod_size', 'dev_stat']
        df_mrds = df_mrds[[c for c in cols if c in df_mrds.columns]].copy()
        df_mrds = df_mrds.dropna(subset=['latitude', 'longitude'])
        df_mrds = df_mrds.drop_duplicates()
        df_mrds = df_mrds.rename(columns={
            'site_name': 'Nom',
            'latitude': 'Latitude',
            'longitude': 'Longitude',
            'commod1': 'Commodite',
            'prod_size': 'Taille_Production',
            'dev_stat': 'Statut_Developpement'
        })
        save(df_mrds, "mrds_cleaned.csv")
    else:
        print("  mrds.csv not found!")

    # 6. Coal Mines
    print("\nCleaning coal mines...")
    coal_path = os.path.join(raw_dir, "geologie", "Global Coal Mine Tracker, May 2026__.xlsx")
    if os.path.exists(coal_path):
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
        save(df_coal, "coal_mines_cleaned.csv")
    else:
        print("  Coal mines tracker not found!")

    # 7. Iron Mines
    print("\nCleaning iron mines...")
    iron_path = os.path.join(raw_dir, "geologie", "Global-Iron-Ore-Mines-Tracker-August-2025-V1.xlsx")
    if os.path.exists(iron_path):
        df_iron = pd.read_excel(iron_path, sheet_name="Main Data")
        df_iron = df_iron.dropna(subset=['Coordinates'])
        
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
        save(df_iron, "iron_mines_cleaned.csv")
    else:
        print("  Iron mines tracker not found!")

    # 8. Oil & Gas
    print("\nCleaning oil and gas...")
    oil_path = os.path.join(raw_dir, "geologie", "Global-Oil-and-Gas-Extraction-Tracker-March-2026.xlsx")
    if os.path.exists(oil_path):
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
        save(df_oil, "oil_gas_cleaned.csv")
    else:
        print("  Oil and gas tracker not found!")

    # ==============================================================================
    # PHASE 3: Clean Hydrology and Socio-Economic datasets
    # ==============================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: CLEANING HYDROLOGY & SOCIO-ECONOMIC DATASETS")
    print("=" * 80)

    # 1. Hydrology (Freshwater withdrawal)
    print("\nCleaning Hydrology freshwater withdrawal...")
    hydro_path = os.path.join(raw_dir, "hydrologie_eau", "wb_freshwater_withdrawal_pct.csv")
    if os.path.exists(hydro_path):
        df_h = load_csv(hydro_path)
        col_zone = find_col(df_h, 'country_name', 'country_long', 'Zone', 'Area')
        col_annee = find_col(df_h, 'year', 'Year', 'Année')
        col_valeur = find_col(df_h, 'value', 'Value', 'Valeur')
        
        cols_keep = {col_zone: 'Pays', col_annee: 'Annee', col_valeur: 'Valeur'}
        df_h = df_h[[c for c in cols_keep.keys()]].copy().rename(columns=cols_keep)
        df_h = clean_common(df_h)
        save(df_h, "wb_freshwater_withdrawal_pct.csv")
    else:
        print("  wb_freshwater_withdrawal_pct.csv not found!")

    # 2. World Bank Socio-Economic indicators
    wb_dir = os.path.join(raw_dir, "socio_economie_demographie", "worldbank")
    if os.path.exists(wb_dir):
        print("\nCleaning World Bank indicators...")
        for filename in os.listdir(wb_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(wb_dir, filename)
                df_wb = load_csv(file_path)
                col_zone = find_col(df_wb, 'country_name', 'country_long', 'Zone', 'Area')
                col_annee = find_col(df_wb, 'year', 'Year', 'Année')
                col_valeur = find_col(df_wb, 'value', 'Value', 'Valeur')
                
                if col_zone and col_annee and col_valeur:
                    cols_keep = {col_zone: 'Pays', col_annee: 'Annee', col_valeur: 'Valeur'}
                    df_wb = df_wb[[c for c in cols_keep.keys()]].copy().rename(columns=cols_keep)
                    df_wb = clean_common(df_wb)
                    save(df_wb, filename)
                else:
                    print(f"  Skipping {filename} - cols not found")
    else:
        print("  World Bank directory not found!")

    # 3. OWID Socio-Economic datasets
    owid_dir = os.path.join(raw_dir, "socio_economie_demographie", "owid")
    if os.path.exists(owid_dir):
        print("\nCleaning OWID indicators...")
        for filename in os.listdir(owid_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(owid_dir, filename)
                df_ow = load_csv(file_path)
                
                # OWID headers are Entity, Code, Year, <value_col>
                # Let's map column 0 to Pays, column 2 to Annee, column 3 to Valeur
                if len(df_ow.columns) >= 4:
                    col_zone = df_ow.columns[0]
                    col_annee = df_ow.columns[2]
                    col_valeur = df_ow.columns[3]
                    
                    cols_keep = {col_zone: 'Pays', col_annee: 'Annee', col_valeur: 'Valeur'}
                    df_ow = df_ow[[col_zone, col_annee, col_valeur]].copy().rename(columns=cols_keep)
                    df_ow = clean_common(df_ow)
                    save(df_ow, filename)
                else:
                    print(f"  Skipping {filename} - not enough columns")
    else:
        print("  OWID directory not found!")

    # 4. ACLED / UCDP Conflict dataset
    print("\nCleaning ACLED conflict dataset...")
    acled_path = os.path.join(raw_dir, "socio_economie_demographie", "acled", "UcdpPrioConflict_v24_1.csv")
    if os.path.exists(acled_path):
        df_ac = load_csv(acled_path)
        col_zone = find_col(df_ac, 'location', 'Zone')
        col_annee = find_col(df_ac, 'year', 'Year')
        col_valeur = find_col(df_ac, 'intensity_level', 'Value')
        
        if col_zone and col_annee and col_valeur:
            cols_keep = {col_zone: 'Pays', col_annee: 'Annee', col_valeur: 'Valeur'}
            df_ac = df_ac[[c for c in cols_keep.keys()]].copy().rename(columns=cols_keep)
            df_ac = clean_common(df_ac)
            save(df_ac, "UcdpPrioConflict_v24_1.csv")
        else:
            print("  ACLED columns not found!")
    else:
        print("  ACLED dataset not found!")

    # 5. WHO / OWID health datasets
    who_dir = os.path.join(raw_dir, "socio_economie_demographie", "who")
    if os.path.exists(who_dir):
        print("\nCleaning WHO indicators...")
        for filename in os.listdir(who_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(who_dir, filename)
                df_who = load_csv(file_path)
                
                # Try finding columns or mapping by index
                col_zone = find_col(df_who, 'Entity', 'country_name', 'Zone', 'Area')
                col_annee = find_col(df_who, 'Year', 'year', 'Année')
                col_valeur = find_col(df_who, 'Value', 'value', 'Valeur', df_who.columns[3] if len(df_who.columns) >= 4 else None)
                
                if not col_zone and len(df_who.columns) >= 1:
                    col_zone = df_who.columns[0]
                if not col_annee and len(df_who.columns) >= 3:
                    col_annee = df_who.columns[2]
                if not col_valeur and len(df_who.columns) >= 4:
                    col_valeur = df_who.columns[3]
                    
                if col_zone and col_annee and col_valeur:
                    cols_keep = {col_zone: 'Pays', col_annee: 'Annee', col_valeur: 'Valeur'}
                    df_who = df_who[[col_zone, col_annee, col_valeur]].copy().rename(columns=cols_keep)
                    df_who = clean_common(df_who)
                    save(df_who, filename)
                else:
                    print(f"  Skipping {filename} - WHO cols not found")
    else:
        print("  WHO directory not found!")

    print("\n[SUCCESS] All datasets cleaned successfully!")

if __name__ == "__main__":
    main()
