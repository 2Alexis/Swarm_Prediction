import pandas as pd

# Load datasets
df_prod = pd.read_csv('data/cleaned/production_cultures.csv')
df_fert = pd.read_csv('data/cleaned/fertilizers_nutrient.csv')
df_temp = pd.read_csv('data/cleaned/mean_temperature.csv')
df_precip = pd.read_csv('data/cleaned/precipitations.csv')

# Search for Estonia-related names in the raw files
print("Raw prod unique names containing eston/Eston:", [c for c in df_prod['Pays'].unique() if 'eston' in str(c).lower()])
print("Raw fert unique names containing eston/Eston:", [c for c in df_fert['Pays'].unique() if 'eston' in str(c).lower()])
print("Raw temp unique names containing eston/Eston:", [c for c in df_temp['Pays'].unique() if 'eston' in str(c).lower()])
print("Raw precip unique names containing eston/Eston:", [c for c in df_precip['Pays'].unique() if 'eston' in str(c).lower()])

# Check if Estonie is in french_to_english
from test_centroid_mapping import french_to_english
print("french_to_english mapping for 'Estonie':", french_to_english.get('Estonie'))
print("french_to_english mapping for 'Estonia':", french_to_english.get('Estonia'))

# Apply translation
df_prod['Pays_EN'] = df_prod['Pays'].map(french_to_english).fillna(df_prod['Pays'])
df_fert['Pays_EN'] = df_fert['Pays'].map(french_to_english).fillna(df_fert['Pays'])
df_temp['Pays_EN'] = df_temp['Pays'].map(french_to_english).fillna(df_temp['Pays'])
df_precip['Pays_EN'] = df_precip['Pays'].map(french_to_english).fillna(df_precip['Pays'])

print("\nAfter translation:")
print("prod unique mapped names containing eston/Eston:", [c for c in df_prod['Pays_EN'].unique() if 'eston' in str(c).lower()])
print("fert unique mapped names containing eston/Eston:", [c for c in df_fert['Pays_EN'].unique() if 'eston' in str(c).lower()])
print("temp unique mapped names containing eston/Eston:", [c for c in df_temp['Pays_EN'].unique() if 'eston' in str(c).lower()])
print("precip unique mapped names containing eston/Eston:", [c for c in df_precip['Pays_EN'].unique() if 'eston' in str(c).lower()])
