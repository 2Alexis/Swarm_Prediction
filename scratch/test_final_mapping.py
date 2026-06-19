import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Complete custom French to ISO mapping
custom_mappings = {
    'pays-bas (royaume des)': 'NL',
    'pays-bas': 'NL',
    'belgique': 'BE',
    'allemagne': 'DE',
    'france': 'FR',
    'espagne': 'ES',
    'italie': 'IT',
    "royaume-uni de grande-bretagne et d'irlande du nord": 'GB',
    'royaume-uni': 'GB',
    "états-unis d'amérique": 'US',
    'états-unis': 'US',
    'chine': 'CN',
    'chine, continentale': 'CN',
    'japon': 'JP',
    'canada': 'CA',
    'australie': 'AU',
    'nouvelle-zélande': 'NZ',
    'brésil': 'BR',
    'inde': 'IN',
    'afrique du sud': 'ZA',
    'égypte': 'EG',
    'maroc': 'MA',
    'algérie': 'DZ',
    'tunisie': 'TN',
    'turquie': 'TR',
    'türkiye': 'TR',
    'grèce': 'GR',
    'pologne': 'PL',
    'suède': 'SE',
    'norvège': 'NO',
    'finlande': 'FI',
    'danemark': 'DK',
    'suisse': 'CH',
    'autriche': 'AT',
    'portugal': 'PT',
    'irlande': 'IE',
    'luxembourg': 'LU',
    'argentine': 'AR',
    'chili': 'CL',
    'colombie': 'CO',
    'pérou': 'PE',
    'mexique': 'MX',
    'fédération de russie': 'RU',
    'russie': 'RU',
    'bélarus': 'BY',
    'ukraine': 'UA',
    'roumanie': 'RO',
    'hongrie': 'HU',
    'tchécoslovaquie': 'CZ',
    'tchéquie': 'CZ',
    'république tchèque': 'CZ',
    'slovaquie': 'SK',
    'thaïlande': 'TH',
    'indonésie': 'ID',
    'malaisie': 'MY',
    'philippines': 'PH',
    'viet nam': 'VN',
    'viêt nam': 'VN',
    'vietnam': 'VN',
    'arabie saoudite': 'SA',
    "iran (république islamique d')": 'IR',
    'iran': 'IR',
    'irak': 'IQ',
    'iraq': 'IQ',
    'israël': 'IL',
    'jordanie': 'JO',
    'liban': 'LB',
    'syrie': 'SY',
    'république arabe syrienne': 'SY',
    'pakistan': 'PK',
    'bangladesh': 'BD',
    'corée, république de': 'KR',
    'corée du sud': 'KR',
    'république de corée': 'KR',
    'corée, république populaire démocratique de': 'KP',
    'corée du nord': 'KP',
    'république populaire démocratique de corée': 'KP',
    'taïwan': 'TW',
    'chine, taiwan province de': 'TW',
    'singapour': 'SG',
    'kenya': 'KE',
    'éthiopie': 'ET',
    'nigeria': 'NG',
    'nigéria': 'NG',
    'ghana': 'GH',
    'sénégal': 'SN',
    'cameroun': 'CM',
    "côte d'ivoire": 'CI',
    'madagascar': 'MG',
    'tanzanie, république-unie de': 'TZ',
    'tanzanie': 'TZ',
    'république-unie de tanzanie': 'TZ',
    'ouganda': 'UG',
    'soudan': 'SD',
    'soudan du sud': 'SS',
    'république démocratique du congo': 'CD',
    'congo drc': 'CD',
    'congo': 'CG',
    'congo, république du': 'CG',
    'congo (brazzaville)': 'CG',
    'congo (kinshasa)': 'CD',
    'angola': 'AO',
    'mozambique': 'MZ',
    'zimbabwe': 'ZW',
    'zambie': 'ZM',
    'botswana': 'BW',
    'namibie': 'NA',
    'maurice': 'MU',
    'seychelles': 'SC',
    'mali': 'ML',
    'niger': 'NE',
    'tchad': 'TD',
    'mauritanie': 'MR',
    'somalie': 'SO',
    'burkina faso': 'BF',
    'bénin': 'BJ',
    'togo': 'TG',
    'guinée': 'GN',
    'libéria': 'LR',
    'liberia': 'LR',
    'sierra leone': 'SL',
    'gambie': 'GM',
    'guinée-bissau': 'GW',
    'guinée équatoriale': 'GQ',
    'gabon': 'GA',
    'république centrafricaine': 'CF',
    'érythrée': 'ER',
    'djibouti': 'DJ',
    'rwanda': 'RW',
    'burundi': 'BI',
    'oman': 'OM',
    'yémen': 'YE',
    'qatar': 'QA',
    'koweït': 'KW',
    'bahreïn': 'BH',
    'émirats arabes unis': 'AE',
    'chypre': 'CY',
    'islande': 'IS',
    'slovénie': 'SI',
    'croatie': 'HR',
    'bosnie-herzégovine': 'BA',
    'serbie': 'RS',
    'monténégro': 'ME',
    'macédoine du nord': 'MK',
    'macédoine': 'MK',
    'albanie': 'AL',
    'bulgarie': 'BG',
    'moldavie': 'MD',
    'république de moldova': 'MD',
    'géorgie': 'GE',
    'arménie': 'AM',
    'azerbaïdjan': 'AZ',
    'kazakhstan': 'KZ',
    'ouzbékistan': 'UZ',
    'turkménistan': 'TM',
    'kirghizistan': 'KG',
    'tadjikistan': 'TJ',
    'népal': 'NP',
    'bhoutan': 'BT',
    'sri lanka': 'LK',
    'maldives': 'MV',
    'myanmar': 'MM',
    'birmanie': 'MM',
    'laos': 'LA',
    'république démocratique populaire lao': 'LA',
    'cambodge': 'KH',
    'mongolie': 'MN',
    'brunéi darussalam': 'BN',
    'brunei': 'BN',
    'papouasie-nouvelle-guinée': 'PG',
    'fidji': 'FJ',
    'salomon': 'SB',
    'îles salomon': 'SB',
    'vanuatu': 'VU',
    'samoa': 'WS',
    'tonga': 'TO',
    'tuvalu': 'TV',
    'nauru': 'NR',
    'kiribati': 'KI',
    'micronésie (états fédérés de)': 'FM',
    'micronésie': 'FM',
    'îles marshall': 'MH',
    'palaos': 'PW',
    'cuba': 'CU',
    'haïti': 'HT',
    'république dominicaine': 'DO',
    'jamaïque': 'JM',
    'trinité-et-tobago': 'TT',
    'bahamas': 'BS',
    'barbade': 'BB',
    'sainte-lucie': 'LC',
    'saint-vincent-et-les grenadines': 'VC',
    'grenade': 'GD',
    'antigua-et-barbuda': 'AG',
    'saint-christophe-et-niévès': 'KN',
    'belize': 'BZ',
    'honduras': 'HN',
    'panama': 'PA',
    'costa rica': 'CR',
    'nicaragua': 'NI',
    'guatemala': 'GT',
    'salvador': 'SV',
    'el salvador': 'SV',
    'guyana': 'GY',
    'suriname': 'SR',
    'équateur': 'EC',
    'venezuela (république bolivarienne du)': 'VE',
    'venezuela': 'VE',
    'bolivie (état plurinational de)': 'BO',
    'bolivie': 'BO',
    'paraguay': 'PY',
    'uruguay': 'UY',
    'luxembourg': 'LU',
    'malte': 'MT',
    'andorre': 'AD',
    'monaco': 'MC',
    'liechtenstein': 'LI',
    'saint-marin': 'SM',
    'saint-siège': 'VA',
    'vatican': 'VA',
    'cabo verde': 'CV',
    'timor-leste': 'TL',
    'réunion': 'RE',
    'guyane française': 'GF',
    'nouvelle-calédonie': 'NC',
    'polynésie française': 'PF',
    'porto rico': 'PR',
    'saint-kitts-et-nevis': 'KN',
    'sao tomé-et-principe': 'ST',
    'serbie-et-monténégro': 'CS',
    'sud-soudan': 'SS',
    'rfs de yougoslavie': 'RS',
    'tchécoslovaq': 'CZ',
    'palestine': 'PS',
    'dominique': 'DM',
    'nioué': 'NU',
    'îles cook': 'CK',
    'îles féroé': 'FO'
}

# 2. Load data
df_prod = pd.read_csv('data/cleaned/production_cultures.csv')
df_fert = pd.read_csv('data/cleaned/fertilizers_nutrient.csv')
df_temp = pd.read_csv('data/cleaned/mean_temperature.csv')
df_precip = pd.read_csv('data/cleaned/precipitations.csv')
centroids = pd.read_csv('data/cleaned/country_centroids.csv', keep_default_na=False)

# Add custom manual coordinates for Taiwan (TW) since it is missing from country_centroids.csv
taiwan_row = pd.DataFrame([{
    'longitude': 120.9605,
    'latitude': 23.6978,
    'COUNTRY': 'Taiwan',
    'ISO': 'TW',
    'COUNTRYAFF': 'Taiwan',
    'AFF_ISO': 'TW'
}])
centroids = pd.concat([centroids, taiwan_row], ignore_index=True)

# Build ISO to coordinates map
iso_to_coords = {}
for idx, row in centroids.iterrows():
    iso_to_coords[row['ISO']] = (row['latitude'], row['longitude'], row['COUNTRY'])

# Clean and apply mapping
df_prod['Pays_Clean'] = df_prod['Pays'].str.strip().str.lower()
df_prod['ISO'] = df_prod['Pays_Clean'].map(custom_mappings)

df_fert['Pays_Clean'] = df_fert['Pays'].str.strip().str.lower()
df_fert['ISO'] = df_fert['Pays_Clean'].map(custom_mappings)

df_temp['Pays_Clean'] = df_temp['Pays'].str.strip().str.lower()
df_temp['ISO'] = df_temp['Pays_Clean'].map(custom_mappings)

df_precip['Pays_Clean'] = df_precip['Pays'].str.strip().str.lower()
df_precip['ISO'] = df_precip['Pays_Clean'].map(custom_mappings)

# Filter out regional aggregates (anything not mapped to an ISO code)
df_prod_filtered = df_prod.dropna(subset=['ISO'])
df_fert_filtered = df_fert.dropna(subset=['ISO'])
df_temp_filtered = df_temp.dropna(subset=['ISO'])
df_precip_filtered = df_precip.dropna(subset=['ISO'])

df_rend_agg = df_prod_filtered[df_prod_filtered['Element'] == 'Rendement'].copy()
df_rend_agg = df_rend_agg[df_rend_agg['Valeur'] <= 100000]
df_rend_agg = df_rend_agg.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Rendement'})

df_temp_agg   = df_temp_filtered.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Temperature_C'})
df_precip_agg = df_precip_filtered.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Precipitations_mm'})
df_fert_agg = df_fert_filtered.groupby(['ISO', 'Annee'])['Valeur'].sum().reset_index().rename(columns={'Valeur': 'Engrais_kgha'})

df_full = df_rend_agg.copy()
df_full = pd.merge(df_full, df_temp_agg, on=['ISO', 'Annee'], how='left')
df_full = pd.merge(df_full, df_precip_agg, on=['ISO', 'Annee'], how='left')
df_full = pd.merge(df_full, df_fert_agg, on=['ISO', 'Annee'], how='left')

recents_full = df_full[df_full['Annee'] >= 2010].dropna(subset=['Rendement', 'Engrais_kgha'])

pays_stats = recents_full.groupby('ISO').agg(
    Rendement_med   = ('Rendement',        'median'),
    Engrais_moy     = ('Engrais_kgha',     'mean')
).dropna()

print(f"Total modeling countries (pays_stats by ISO): {len(pays_stats)}")

missing = set(pays_stats.index) - set(iso_to_coords.keys())
print(f"Missing from centroids file: {missing}")

# Print matched list
print("\nMatched Countries Sample:")
for iso in sorted(list(pays_stats.index))[:20]:
    lat, lon, eng_name = iso_to_coords[iso]
    print(f"  - ISO {iso}: {eng_name} (Lat: {lat:.4f}, Lon: {lon:.4f})")
