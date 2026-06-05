import os
import pandas as pd

cleaned_dir = r"c:\Users\alexi\Desktop\Swarm_Prediction\data\cleaned"
print("Cleaned files in:", cleaned_dir)

for f in sorted(os.listdir(cleaned_dir)):
    if f.endswith('.csv'):
        path = os.path.join(cleaned_dir, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print("=" * 80)
        print(f"File: {f} ({size_mb:.2f} MB)")
        try:
            df = pd.read_csv(path, nrows=5)
            print("Columns:", list(df.columns))
            print("First row:", df.iloc[0].to_dict() if len(df) > 0 else "Empty")
            # Get count and range of years/countries if columns exist
            df_full = pd.read_csv(path)
            print(f"Total Rows: {len(df_full):,}")
            pays_col = [c for c in df_full.columns if 'pays' in c.lower() or 'area' in c.lower()]
            annee_col = [c for c in df_full.columns if 'annee' in c.lower() or 'year' in c.lower() or 'time' in c.lower()]
            if pays_col:
                print(f"Unique Countries: {df_full[pays_col[0]].nunique()}")
            if annee_col:
                print(f"Year Range: {df_full[annee_col[0]].min()} to {df_full[annee_col[0]].max()}")
        except Exception as e:
            print("Error:", str(e))
