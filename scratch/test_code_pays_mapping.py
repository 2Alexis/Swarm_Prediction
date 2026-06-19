import pandas as pd
import pycountry
import os

DATA_DIR = "data/cleaned"

df_temp = pd.read_csv(os.path.join(DATA_DIR, "mean_temperature.csv"))
df_precip = pd.read_csv(os.path.join(DATA_DIR, "precipitations.csv"))

unique_codes = set(df_temp['Code_Pays'].unique()).union(set(df_precip['Code_Pays'].unique()))

resolved = {}
unresolved = []

for code3 in unique_codes:
    if len(str(code3)) == 3:
        country = pycountry.countries.get(alpha_3=str(code3).upper())
        if country:
            resolved[code3] = country.alpha_2
        else:
            unresolved.append(code3)
    else:
        unresolved.append(code3)

print(f"Resolved {len(resolved)} / {len(unique_codes)} 3-letter codes.")
print("\nUnresolved codes:")
for u in sorted(unresolved):
    # Print the country name associated with this code from the dataset
    temp_matches = df_temp[df_temp['Code_Pays'] == u]['Pays'].unique()
    precip_matches = df_precip[df_precip['Code_Pays'] == u]['Pays'].unique()
    print(f"  - {u} (Names in datasets: {list(temp_matches)} / {list(precip_matches)})")
