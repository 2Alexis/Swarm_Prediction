import pandas as pd
import os

filepath = os.path.join("data", "cleaned", "dataset_final_modelisation.csv")
if os.path.exists(filepath):
    df = pd.read_csv(filepath)
    print(f"Shape: {df.shape}")
    print("\n--- Summary of missing/populated columns ---")
    for col in df.columns:
        not_null = df[col].notnull().sum()
        nulls = df[col].isnull().sum()
        print(f"  {col}: {not_null} non-null, {nulls} nulls ({nulls/len(df)*100:.2f}% null)")
else:
    print(f"File not found: {filepath}")
