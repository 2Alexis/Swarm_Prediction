import os
import pandas as pd

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
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Years range:", df['Annee'].min(), "to", df['Annee'].max() if 'Annee' in df.columns else "NO ANNEE")
        print("Sample raw country names:", df['Pays'].unique()[:5])
        if 'Produit' in df.columns:
            print("Sample products:", df['Produit'].unique()[:5])
        print(df.head(2))
    else:
        print(f"\nFile {filename} NOT found")
