import pandas as pd

# Load centroids with keep_default_na=False
centroids = pd.read_csv('data/cleaned/country_centroids.csv', keep_default_na=False)

print("Total rows:", len(centroids))
print("Columns:", centroids.columns.tolist())

# Check for specific ISO codes
test_codes = ['BY', 'CV', 'TW', 'CD', 'CG', 'CI', 'SV', 'IQ', 'KG', 'LR', 'NA', 'NG', 'KP', 'KR', 'TZ', 'TL', 'TR']
print("\nChecking test codes in country_centroids.csv:")
for code in test_codes:
    matches = centroids[centroids['ISO'] == code]
    if not matches.empty:
        print(f"  Code {code}: found country '{matches.iloc[0]['COUNTRY']}' (Affiliation: {matches.iloc[0]['COUNTRYAFF']})")
    else:
        print(f"  Code {code}: NOT found")

# Let's search for some countries by name in the centroids file
print("\nSearching for Taiwan, Tanzania, Korea, Congo, etc. by name:")
for name in ['taiwan', 'tanzania', 'korea', 'congo', 'ivory', 'coast', 'salvador', 'iraq', 'kyrgyz', 'liberia', 'nigeria', 'timor', 'turkey']:
    matches = centroids[centroids['COUNTRY'].str.lower().str.contains(name)]
    if not matches.empty:
        print(f"  Search '{name}':")
        for idx, row in matches.iterrows():
            print(f"    - {row['COUNTRY']} (ISO: {row['ISO']}, Lat: {row['latitude']}, Lon: {row['longitude']})")
    else:
        print(f"  Search '{name}': NOT found")
