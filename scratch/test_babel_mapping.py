import pandas as pd
import babel
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Load unique country names from production
df_prod = pd.read_csv('data/cleaned/production_cultures.csv')
unique_pays = df_prod['Pays'].unique()

# Load country centroids
centroids = pd.read_csv('data/cleaned/country_centroids.csv')

# Build ISO to coordinates map
iso_to_coords = {}
for idx, row in centroids.iterrows():
    iso_to_coords[row['ISO']] = (row['latitude'], row['longitude'], row['COUNTRY'])

# Build French name to ISO code map using Babel
locale_fr = babel.Locale('fr')
fr_to_iso = {}
for code, name in locale_fr.territories.items():
    fr_to_iso[name.lower()] = code

# We also handle some variations and common naming styles
custom_mappings = {
    'pays-bas (royaume des)': 'NL',
    'royaume-uni de grande-bretagne et d\'irlande du nord': 'GB',
    'états-unis d\'amérique': 'US',
    'chine, continentale': 'CN',
    'chine - ras de hong-kong': 'HK',
    'chine - ras de macao': 'MO',
    'chine, taiwan province de': 'TW',
    'république tchèque': 'CZ',
    'viet nam': 'VN',
    'viêt nam': 'VN',
    'iran (république islamique d\')': 'IR',
    'république arabe syrienne': 'SY',
    'corée, république de': 'KR',
    'corée, république populaire démocratique de': 'KP',
    'tanzanie, république-unie de': 'TZ',
    'république démocratique du congo': 'CD',
    'congo, république du': 'CG',
    'congo (brazzaville)': 'CG',
    'congo (kinshasa)': 'CD',
    'bolivie (état plurinational de)': 'BO',
    'venezuela (république bolivarienne du)': 'VE',
    'micronésie (états fédérés de)': 'FM',
    'saint-christophe-et-niévès': 'KN',
    'saint-vincent-et-les grenadines': 'VC',
    'trinité-et-tobago': 'TT',
    'saint-martin': 'MF',
    'sint maarten': 'SX',
    'antigua-et-barbuda': 'AG',
    'brunéi darussalam': 'BN',
    'république de moldova': 'MD',
    'laos': 'LA',
    'république démocratique populaire lao': 'LA',
    'lacs': 'LA', # just in case
    'belgique-luxembourg': 'BE', # approximation
    'soudan (ex)': 'SD',
    'éthiopie rdp': 'ET',
    'tchécoslovaquie': 'CZ',
    'urss': 'RU',
}

# Resolve coordinates for all unique countries
matched = []
unmatched = []

for pays in unique_pays:
    pays_clean = str(pays).strip().lower()
    
    # Check custom mappings first
    code = custom_mappings.get(pays_clean)
    if not code:
        code = fr_to_iso.get(pays_clean)
        
    if code and code in iso_to_coords:
        lat, lon, eng_name = iso_to_coords[code]
        matched.append((pays, code, eng_name, lat, lon))
    else:
        unmatched.append((pays, code))

print(f"Matched {len(matched)} / {len(unique_pays)} entries.")
print("\nUnmatched entries:")
for pays, code in sorted(unmatched, key=lambda x: str(x[0])):
    print(f"  - {pays} (Code detected: {code})")
