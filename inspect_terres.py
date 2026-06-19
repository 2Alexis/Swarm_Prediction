import pandas as pd
import os

filepath = os.path.join("data", "cleaned", "intrants_utilisation_terres.csv")
if os.path.exists(filepath):
    df = pd.read_csv(filepath, nrows=10)
    print("Columns in intrants_utilisation_terres.csv:")
    print(df.columns)
    print("\nUnique values in 'Element' or 'Produit' column if they exist:")
    df_full = pd.read_csv(filepath)
    for col in df_full.columns:
        if col in ['Element', 'Produit', 'Item', 'Indicator']:
            print(f"--- Unique values in {col} ---")
            print(df_full[col].unique()[:20])
else:
    print("File not found!")
