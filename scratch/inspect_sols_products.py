import pandas as pd
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "data/cleaned"

df_sols = pd.read_csv(os.path.join(DATA_DIR, "bilan_nutritif_sols.csv"))
df_prod = pd.read_csv(os.path.join(DATA_DIR, "production_cultures.csv"))

def safe_print(arr, label):
    safe_arr = [str(x).encode('ascii', errors='replace').decode('ascii') for x in arr]
    print(f"{label} (first 20):")
    print(safe_arr[:20])

safe_print(df_sols['Produit'].unique(), "Unique products in df_sols")
safe_print(df_prod['Produit'].unique(), "Unique products in df_prod")

common = set(df_sols['Produit'].unique()).intersection(set(df_prod['Produit'].unique()))
print(f"\nCommon products count: {len(common)}")
safe_print(list(common), "Common products")
