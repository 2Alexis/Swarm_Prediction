import os
import pandas as pd
import babel
import sys

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "data/cleaned"

# English name to ISO map
locale_en = babel.Locale('en')
en_to_iso = {name.lower(): code for code, name in locale_en.territories.items()}

custom_english_mappings = {
    'iran, islamic rep.': 'IR',
    'korea, rep.': 'KR',
    'korea, dem. people\'s rep.': 'KP',
    'democratic people\'s republic of korea': 'KP',
    'republic of korea': 'KR',
    'russian federation': 'RU',
    'venezuela, rb': 'VE',
    'yemen, rep.': 'YE',
    'egypt, arab rep.': 'EG',
    'syrian arab republic': 'SY',
    'congo, dem. rep.': 'CD',
    'congo, rep.': 'CG',
    'bahamas, the': 'BS',
    'gambia, the': 'GM',
    'brunei darussalam': 'BN',
    'cote d\'ivoire': 'CI',
    'slovakia': 'SK',
    'czechia': 'CZ',
    'lao pdr': 'LA',
    'kyrgyz republic': 'KG',
    'micronesia, fed. sts.': 'FM',
    'st. lucia': 'LC',
    'st. vincent and the grenadines': 'VC',
    'st. kitts and nevis': 'KN',
    'trinidad and tobago': 'TT',
    'netherlands': 'NL',
    'belgium': 'BE',
    'united kingdom': 'GB',
    'united states': 'US',
    'hong kong sar, china': 'HK',
    'macao sar, china': 'MO',
    'taiwan, china': 'TW',
    'west bank and gaza': 'PS',
    'palestine': 'PS',
    'sao tome and principe': 'ST',
    'eswatini': 'SZ'
}

def get_english_iso(pays_name):
    clean = str(pays_name).strip().lower()
    code = custom_english_mappings.get(clean)
    if not code:
        code = en_to_iso.get(clean)
    return code

# Load one of the World Bank datasets
df_gdp = pd.read_csv(os.path.join(DATA_DIR, "wb_gdp_per_capita.csv"))
unique_gdp_countries = df_gdp['Pays'].unique()

resolved = 0
unresolved = []

for pays in unique_gdp_countries:
    code = get_english_iso(pays)
    if code:
        resolved += 1
    else:
        unresolved.append(pays)

print(f"Resolved {resolved} / {len(unique_gdp_countries)} countries in GDP per capita dataset.")
print("\nUnresolved (mostly regional groups):")
for u in sorted(unresolved):
    print("  -", u)
