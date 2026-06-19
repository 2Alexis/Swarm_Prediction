import os
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "data/cleaned"

datasets = {
    'prod': "production_cultures.csv",
    'temp': "mean_temperature.csv",
    'precip': "precipitations.csv",
    'fert': "fertilizers_nutrient.csv",
    'pest': "pesticides.csv",
    'sols': "bilan_nutritif_sols.csv",
    'gdp': "wb_gdp_per_capita.csv",
    'life': "wb_life_expectancy.csv",
    'child': "wb_child_mortality.csv",
    'hdi': "owid_hdi.csv",
    'malaria': "wb_malaria_incidence.csv",
    'hydro': "wb_freshwater_withdrawal_pct.csv"
}

for name, filename in datasets.items():
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n==================== {name} ({filename}) ====================")
        print("Columns:", df.columns.tolist())
        
        # Check for columns representing country codes or country names
        code_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['code', 'iso', 'area'])]
        pays_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['pays', 'country', 'zone', 'area'])]
        
        print(f"Potential Code Columns: {code_cols}")
        print(f"Potential Name Columns: {pays_cols}")
        
        # Print first 5 country names safely
        pays_col = pays_cols[0] if pays_cols else None
        if pays_col:
            raw_vals = df[pays_col].dropna().unique()[:5]
            safe_vals = [str(v).encode('ascii', errors='replace').decode('ascii') for v in raw_vals]
            print(f"Sample values from '{pays_col}': {safe_vals}")
            
        # Print first 5 code names safely
        code_col = code_cols[0] if code_cols else None
        if code_col:
            raw_codes = df[code_col].dropna().unique()[:5]
            safe_codes = [str(c).encode('ascii', errors='replace').decode('ascii') for c in raw_codes]
            print(f"Sample values from '{code_col}': {safe_codes}")
            
        print("Sample Row:", {k: str(v).encode('ascii', errors='replace').decode('ascii') for k, v in df.iloc[0].to_dict().items()})
    else:
        print(f"\nFile {filename} NOT found")
