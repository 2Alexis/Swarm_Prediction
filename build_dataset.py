import os
import pandas as pd
import numpy as np
import pycountry
import babel
from scipy.spatial import cKDTree
from layer1_engine import Layer1Engine

DATA_DIR = "data/cleaned"
OUTPUT_FILE = os.path.join(DATA_DIR, "dataset_final_modelisation.csv")

# 1. Custom French to ISO map
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

# 2. Custom English to ISO map for World Bank/OWID datasets
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
    'eswatini': 'SZ',
    'antigua and barbuda': 'AG',
    'bosnia and herzegovina': 'BA',
    'cabo verde': 'CV',
    'curacao': 'CW',
    'myanmar': 'MM',
    'sint maartin (dutch part)': 'SX',
    'sint maarten (dutch part)': 'SX',
    'slovak republic': 'SK',
    'somalia, fed. rep.': 'SO',
    'st. martin (french part)': 'MF',
    'turkiye': 'TR',
    'turks and caicos islands': 'TC',
    'viet nam': 'VN',
    'virgin islands (u.s.)': 'VI'
}

locale_en = babel.Locale('en')
en_to_iso = {name.lower(): code for code, name in locale_en.territories.items()}

def get_english_iso(pays_name):
    clean = str(pays_name).strip().lower()
    code = custom_english_mappings.get(clean)
    if not code:
        code = en_to_iso.get(clean)
    return code

def get_iso_from_alpha3(code3):
    if len(str(code3)) == 3:
        country = pycountry.countries.get(alpha_3=str(code3).upper())
        if country:
            return country.alpha_2
    return None

# 3. Agricultural centroid overrides for large countries (Lat, Lon)
agricultural_centroids = {
    'CA': (50.50, -105.50), # Southern Plains (Saskatchewan)
    'DZ': (36.50, 3.00),   # Northern Coast
    'AU': (-34.50, 143.50), # Southeast (Murray-Darling)
    'EG': (30.10, 31.20),   # Nile Delta
    'RU': (52.00, 45.00),   # European Russia
    'CN': (34.50, 114.50),  # North China Plain
    'US': (41.50, -93.50),  # Midwest (Iowa)
    'BR': (-20.50, -48.50)  # Southeast/Center-West
}

def haversine(lat1, lon1, lat2, lon2):
    """Computes Haversine distance in kilometers between two points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def clean_columns(df):
    rename_dict = {}
    for col in df.columns:
        if 'l' in col.lower() and 'm' in col.lower() and 'nt' in col.lower():
            rename_dict[col] = 'Element'
        elif 'unit' in col.lower():
            rename_dict[col] = 'Unite'
    return df.rename(columns=rename_dict)

def build_dataset():
    print("[1/5] Chargement de tous les datasets historiques...")
    # Load all cleaned CSVs
    df_prod = clean_columns(pd.read_csv(os.path.join(DATA_DIR, "production_cultures.csv")))
    df_temp = pd.read_csv(os.path.join(DATA_DIR, "mean_temperature.csv"))
    df_precip = pd.read_csv(os.path.join(DATA_DIR, "precipitations.csv"))
    df_fert = pd.read_csv(os.path.join(DATA_DIR, "fertilizers_nutrient.csv"))
    df_pest = pd.read_csv(os.path.join(DATA_DIR, "pesticides.csv"))
    df_sols = pd.read_csv(os.path.join(DATA_DIR, "bilan_nutritif_sols.csv"))
    df_terres = pd.read_csv(os.path.join(DATA_DIR, "intrants_utilisation_terres.csv"))
    
    # Load socio-demographics and health datasets
    df_gdp = pd.read_csv(os.path.join(DATA_DIR, "wb_gdp_per_capita.csv"))
    df_life = pd.read_csv(os.path.join(DATA_DIR, "wb_life_expectancy.csv"))
    df_child = pd.read_csv(os.path.join(DATA_DIR, "wb_child_mortality.csv"))
    df_hdi = pd.read_csv(os.path.join(DATA_DIR, "owid_hdi.csv"))
    df_malaria = pd.read_csv(os.path.join(DATA_DIR, "wb_malaria_incidence.csv"))
    df_hydro = pd.read_csv(os.path.join(DATA_DIR, "wb_freshwater_withdrawal_pct.csv"))
    
    # Load centroids file with keep_default_na=False to prevent Namibia (NA) from becoming NaN
    df_centroids = pd.read_csv(os.path.join(DATA_DIR, "country_centroids.csv"), keep_default_na=False)
    
    # Add manual entry for Taiwan (TW) since it is missing in the centroids file
    taiwan_row = pd.DataFrame([{
        'longitude': 120.9605,
        'latitude': 23.6978,
        'COUNTRY': 'Taiwan',
        'ISO': 'TW',
        'COUNTRYAFF': 'Taiwan',
        'AFF_ISO': 'TW'
    }])
    df_centroids = pd.concat([df_centroids, taiwan_row], ignore_index=True)
    
    # Build ISO to coordinates map
    iso_to_coords = {}
    for idx, row in df_centroids.iterrows():
        iso_to_coords[row['ISO']] = (row['latitude'], row['longitude'], row['COUNTRY'])
        
    print("[2/5] Standardisation des noms de pays et codes ISO...")
    # 1. French names
    for df in [df_prod, df_fert, df_pest, df_sols, df_terres]:
        df['Pays_Clean'] = df['Pays'].str.strip().str.lower()
        df['ISO'] = df['Pays_Clean'].map(custom_mappings)
        
    # 2. English 3-letter codes
    for df in [df_temp, df_precip]:
        df['ISO'] = df['Code_Pays'].apply(get_iso_from_alpha3)
        
    # 3. English names
    for df in [df_gdp, df_life, df_child, df_hdi, df_malaria, df_hydro]:
        df['ISO'] = df['Pays'].apply(get_english_iso)
        
    # Filter out records that did not map to a valid country ISO (excluding regional aggregates)
    df_prod_f = df_prod.dropna(subset=['ISO']).copy()
    df_fert_f = df_fert.dropna(subset=['ISO']).copy()
    df_temp_f = df_temp.dropna(subset=['ISO']).copy()
    df_precip_f = df_precip.dropna(subset=['ISO']).copy()
    df_pest_f = df_pest.dropna(subset=['ISO']).copy()
    df_sols_f = df_sols.dropna(subset=['ISO']).copy()
    df_terres_f = df_terres.dropna(subset=['ISO']).copy()
    
    df_gdp_f = df_gdp.dropna(subset=['ISO']).copy()
    df_life_f = df_life.dropna(subset=['ISO']).copy()
    df_child_f = df_child.dropna(subset=['ISO']).copy()
    df_hdi_f = df_hdi.dropna(subset=['ISO']).copy()
    df_malaria_f = df_malaria.dropna(subset=['ISO']).copy()
    df_hydro_f = df_hydro.dropna(subset=['ISO']).copy()
    
    print("[3/5] Aggregation et fusion des tables agricoles et socio-economiques...")
    # Clean yield data
    df_yield = df_prod_f[df_prod_f['Element'] == 'Rendement'].copy()
    df_yield = df_yield.rename(columns={'Valeur': 'Rendement_kgha'})
    df_yield = df_yield[df_yield['Rendement_kgha'] <= 100000]
    df_yield = df_yield[df_yield['Rendement_kgha'] > 0]
    
    # Aggregate data
    df_yield_agg = df_yield.groupby(['ISO', 'Produit', 'Annee'])['Rendement_kgha'].mean().reset_index()
    
    # Bug Fix: Filter sols by actual budget component and group at the country-year level
    df_sols_filt = df_sols_f[df_sols_f['Produit'].str.lower().str.contains('bilan nutritif des sols', na=False)].copy()
    df_sols_agg = df_sols_filt.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Bilan_sols_kgha'})
    
    df_temp_agg = df_temp_f.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Temperature_C'})
    df_precip_agg = df_precip_f.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Precipitations_mm'})
    df_fert_agg = df_fert_f.groupby(['ISO', 'Annee'])['Valeur'].sum().reset_index().rename(columns={'Valeur': 'Engrais_kgha'})
    df_pest_agg = df_pest_f.groupby(['ISO', 'Annee'])['Valeur'].sum().reset_index().rename(columns={'Valeur': 'Pesticides_kgha'})
    
    # Aggregate socio-demographics & health
    df_gdp_agg = df_gdp_f.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'GDP_pc'})
    df_life_agg = df_life_f.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Life_Exp'})
    df_child_agg = df_child_f.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Child_Mort'})
    df_hdi_agg = df_hdi_f.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'HDI'})
    df_malaria_agg = df_malaria_f.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Malaria_Incidence'})
    df_hydro_agg = df_hydro_f.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Water_Withdrawal_pct'})
    
    # Pivot df_terres_f pour récupérer l'utilisation des terres
    df_terres_pivot = df_terres_f.pivot_table(
        index=['ISO', 'Annee'],
        columns='Produit',
        values='Valeur',
        aggfunc='mean'
    ).reset_index()
    
    rename_cols = {
        'Superficie du pays': 'Superficie_pays_ha',
        'Superficie des terres': 'Superficie_terres_ha',
        'Terres agricoles': 'Terres_agricoles_ha',
        'Terres en culture': 'Terres_culture_ha',
        'Terres arables': 'Terres_arables_ha',
        "Superficie équipée de systèmes d'irrigation": 'Irrigation_equipee_ha',
        'Superficie des terres réellement irriguée': 'Irrigation_reelle_ha',
        'Terr. agricoles sous agriculture biologique': 'Bio_ha'
    }
    existing_rename = {k: v for k, v in rename_cols.items() if k in df_terres_pivot.columns}
    df_terres_pivot = df_terres_pivot.rename(columns=existing_rename)
    
    # Assurer la présence de toutes les colonnes attendues
    for col in rename_cols.values():
        if col not in df_terres_pivot.columns:
            df_terres_pivot[col] = np.nan
            
    df_terres_agg = df_terres_pivot[['ISO', 'Annee'] + list(rename_cols.values())].copy()
    
    # Calcul des ratios pertinents pour la modélisation du rendement
    df_terres_agg['Part_terres_agricoles'] = df_terres_agg['Terres_agricoles_ha'] / df_terres_agg['Superficie_terres_ha']
    df_terres_agg['Part_terres_arables'] = df_terres_agg['Terres_arables_ha'] / df_terres_agg['Terres_agricoles_ha']
    df_terres_agg['Part_irriguee'] = df_terres_agg['Irrigation_equipee_ha'] / df_terres_agg['Terres_agricoles_ha']
    df_terres_agg['Part_bio'] = df_terres_agg['Bio_ha'] / df_terres_agg['Terres_agricoles_ha']
    
    # Merge tables
    df_master = df_yield_agg.copy()
    df_master = pd.merge(df_master, df_sols_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_temp_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_precip_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_fert_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_pest_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_terres_agg, on=['ISO', 'Annee'], how='left')
    
    # Merge demographics & health
    df_master = pd.merge(df_master, df_gdp_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_life_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_child_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_hdi_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_malaria_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_hydro_agg, on=['ISO', 'Annee'], how='left')
    
    # Keep only post-2010 records
    df_master = df_master[df_master['Annee'] >= 2010]
    # 'ISO' = clé de jointure entre couches + groupe du split par pays -> jamais NaN
    df_master = df_master.dropna(subset=['Rendement_kgha', 'ISO'])
    
    print(f"  -> Nombre total de lignes après fusion : {df_master.shape[0]:,}")
    
    print("[4/5] Resolution des coordonnes (avec surcharges des centroides agricoles)...")
    # Resolve coordinates for each country in df_master
    unique_countries = df_master['ISO'].unique()
    country_coords = {}
    
    for iso in unique_countries:
        # Check if there is an agricultural centroid override
        if iso in agricultural_centroids:
            lat, lon = agricultural_centroids[iso]
            name = iso_to_coords[iso][2] if iso in iso_to_coords else f"ISO_{iso}"
            print(f"  [OVERRIDE] {name} ({iso}) -> Agricultural Centroid (Lat: {lat}, Lon: {lon})")
        elif iso in iso_to_coords:
            lat, lon, name = iso_to_coords[iso]
        else:
            raise ValueError(f"CRITICAL: No coordinates found for ISO code: {iso}")
            
        country_coords[iso] = (lat, lon, name)
        
    # Add coordinates columns to master dataframe
    df_master['latitude'] = df_master['ISO'].apply(lambda x: country_coords[x][0])
    df_master['longitude'] = df_master['ISO'].apply(lambda x: country_coords[x][1])
    df_master['Pays_EN'] = df_master['ISO'].apply(lambda x: country_coords[x][2])
    
    print("[5/5] Extraction des features physiques...")
    
    # Charger explicitement les datasets de risques physiques et ressources
    df_v = pd.read_csv(os.path.join(DATA_DIR, "volcanoes_cleaned.csv"))
    df_eq = pd.read_csv(os.path.join(DATA_DIR, "earthquakes_cleaned.csv"))
    df_pp = pd.read_csv(os.path.join(DATA_DIR, "global_power_plants_cleaned.csv"))
    df_mrds = pd.read_csv(os.path.join(DATA_DIR, "mrds_cleaned.csv"), low_memory=False)
    df_coal = pd.read_csv(os.path.join(DATA_DIR, "coal_mines_cleaned.csv"))
    df_iron = pd.read_csv(os.path.join(DATA_DIR, "iron_mines_cleaned.csv"))
    df_oil = pd.read_csv(os.path.join(DATA_DIR, "oil_gas_cleaned.csv"))
    df_wd = pd.read_csv(os.path.join(DATA_DIR, "wood_density_cleaned.csv"))
    
    # Pré-construire les KDTrees pour des requêtes spatiales rapides
    # 1. Volcans
    df_v_clean = df_v.dropna(subset=['latitude', 'longitude'])
    v_tree = cKDTree(df_v_clean[['latitude', 'longitude']].values)
    
    # 2. Séismes
    df_eq_clean = df_eq.dropna(subset=['latitude', 'longitude'])
    eq_tree = cKDTree(df_eq_clean[['latitude', 'longitude']].values)
    
    # 3. Centrales électriques
    df_pp_clean = df_pp.dropna(subset=['latitude', 'longitude'])
    pp_tree = cKDTree(df_pp_clean[['latitude', 'longitude']].values)
    
    # 4. Ressources minérales (MRDS + Mines de Fer)
    df_mrds_clean = df_mrds.dropna(subset=['Latitude', 'Longitude'])
    df_mrds_sel = df_mrds_clean[['Nom', 'Latitude', 'Longitude', 'Commodite']].copy()
    df_iron_clean = df_iron.dropna(subset=['Latitude', 'Longitude'])
    df_iron_sel = df_iron_clean[['Nom', 'Latitude', 'Longitude']].copy()
    df_iron_sel['Commodite'] = 'Iron Ore'
    df_minerals = pd.concat([df_mrds_sel, df_iron_sel], ignore_index=True)
    minerals_tree = cKDTree(df_minerals[['Latitude', 'Longitude']].values)
    
    # 5. Énergies fossiles (Charbon + Pétrole/Gaz)
    df_coal_clean = df_coal.dropna(subset=['Latitude', 'Longitude'])
    coal_fuel = 'Type_Charbon' if 'Type_Charbon' in df_coal_clean.columns else ('Coal Type' if 'Coal Type' in df_coal_clean.columns else 'Coal')
    df_coal_sel = pd.DataFrame({
        'Nom': df_coal_clean['Nom'],
        'Latitude': df_coal_clean['Latitude'],
        'Longitude': df_coal_clean['Longitude'],
        'Type': df_coal_clean[coal_fuel] if coal_fuel in df_coal_clean.columns else 'Coal'
    })
    df_oil_clean = df_oil.dropna(subset=['Latitude', 'Longitude'])
    oil_fuel = 'Type_Combustible' if 'Type_Combustible' in df_oil_clean.columns else ('Fuel type' if 'Fuel type' in df_oil_clean.columns else 'Oil/Gas')
    df_oil_sel = pd.DataFrame({
        'Nom': df_oil_clean['Nom'],
        'Latitude': df_oil_clean['Latitude'],
        'Longitude': df_oil_clean['Longitude'],
        'Type': df_oil_clean[oil_fuel] if oil_fuel in df_oil_clean.columns else 'Oil/Gas'
    })
    df_fossils = pd.concat([df_coal_sel, df_oil_sel], ignore_index=True)
    fossils_tree = cKDTree(df_fossils[['Latitude', 'Longitude']].values)
    
    # 6. Densité du bois
    df_wd_clean = df_wd.dropna(subset=['Latitude', 'Longitude'])
    wd_tree = cKDTree(df_wd_clean[['Latitude', 'Longitude']].values)

    engine = Layer1Engine(raw_dir='data/raw')
    resolved_features = {}
    print(f"  Traitement de {len(unique_countries)} pays uniques...")
    for idx, iso in enumerate(unique_countries, 1):
        lat, lon, name = country_coords[iso]
        print(f"    ({idx}/{len(unique_countries)}) Calcul des caractéristiques physiques de {name} ({iso})...")
        
        # Récupération des features de base via Layer1Engine (topographie, climat de référence, hydrologie, sol)
        features = engine.get_physical_features(lat, lon)
        
        # Override spatial features avec nos requêtes KDTrees directes
        # 1. Volcans
        v_dist, v_idx = v_tree.query([lat, lon])
        closest_v = df_v_clean.iloc[v_idx]
        features['dist_to_volcano_km'] = round(haversine(lat, lon, closest_v['latitude'], closest_v['longitude']), 2)
        features['closest_volcano_type'] = str(closest_v['primary_volcano_type'])
        features['Volcanic_Hazard_Index'] = round(np.exp(-features['dist_to_volcano_km'] / 50.0), 4)
        
        # 2. Séismes (nombre dans un rayon de 300km)
        candidates_idx = eq_tree.query_ball_point([lat, lon], 5.0)
        eq_count = 0
        for eq_idx in candidates_idx:
            eq_row = df_eq_clean.iloc[eq_idx]
            if haversine(lat, lon, eq_row['latitude'], eq_row['longitude']) <= 300.0:
                eq_count += 1
        features['seismic_risk_index'] = round(min(10.0, eq_count / 15.0 * 10.0), 2)
        
        # Seismic Hazard Index
        dist_fault = features.get('dist_to_fault_km')
        features['Seismic_Hazard_Index'] = round(np.exp(-dist_fault / 100.0), 4) if dist_fault is not None else 0.0
        
        # 3. Centrales électriques (énergie)
        pp_dist, pp_idx = pp_tree.query([lat, lon])
        closest_pp = df_pp_clean.iloc[pp_idx]
        features['dist_to_energy_source_km'] = round(haversine(lat, lon, closest_pp['latitude'], closest_pp['longitude']), 2)
        features['closest_energy_type'] = str(closest_pp['primary_fuel'])
        
        # 4. Ressources minérales
        m_dist, m_idx = minerals_tree.query([lat, lon])
        closest_m = df_minerals.iloc[m_idx]
        features['dist_to_mineral_resource_km'] = round(haversine(lat, lon, closest_m['Latitude'], closest_m['Longitude']), 2)
        features['closest_mineral_type'] = str(closest_m['Commodite'])
        features['closest_mineral_site'] = str(closest_m['Nom'])
        
        # 5. Énergies fossiles
        f_dist, f_idx = fossils_tree.query([lat, lon])
        closest_f = df_fossils.iloc[f_idx]
        features['dist_to_fossil_resource_km'] = round(haversine(lat, lon, closest_f['Latitude'], closest_f['Longitude']), 2)
        features['closest_fossil_type'] = str(closest_f['Type'])
        features['closest_fossil_site'] = str(closest_f['Nom'])
        
        # 6. Densité du bois
        wd_dist, wd_idx = wd_tree.query([lat, lon])
        closest_wd = df_wd_clean.iloc[wd_idx]
        val_str = str(closest_wd['Wood Density']).replace(',', '.')
        try:
            features['estimated_wood_density_g_cm3'] = round(float(val_str), 3)
        except ValueError:
            features['estimated_wood_density_g_cm3'] = 0.55
            
        resolved_features[iso] = features
        
    # Mapper les features physiques dans le DataFrame principal
    first_iso = list(resolved_features.keys())[0]
    feature_keys = list(resolved_features[first_iso].keys())
    for key in feature_keys:
        df_master[key] = df_master['ISO'].map(lambda x: resolved_features[x][key])
        
    # Calculs et transformations de la synthèse (Cellule 74)
    print("[INFO] Application des calculs de synthese...")
    df_master['Temperature_C_sq'] = df_master['Temperature_C'] ** 2
    df_master['Precipitations_mm_sq'] = df_master['Precipitations_mm'] ** 2
    df_master['Engrais_Temp_interaction'] = df_master['Engrais_kgha'] * df_master['Temperature_C']
    df_master['log_Engrais_kgha'] = np.log1p(df_master['Engrais_kgha'].fillna(0.0))
    df_master['log_Pesticides_kgha'] = np.log1p(df_master['Pesticides_kgha'].fillna(0.0))
    df_master['log1p_Rendement'] = np.log1p(df_master['Rendement_kgha'])
    
    # ── CALCUL DES 9 CIBLES PHYSIQUES ET BIOMÉTRIQUES ────────────────────────
    print("[INFO] Calcul des 9 variables cibles pour le dataset final...")
    df_master['target_yield'] = df_master['log1p_Rendement']
    df_master['target_water_stress'] = df_master['Water_Withdrawal_pct']
    df_master['target_npp'] = df_master['npp_g_m2_yr']
    df_master['target_flood_risk'] = ((df_master['elevation'] <= 50) & (df_master['dist_to_freshwater_km'] <= 5.0)).astype(int)
    df_master['target_soil_degradation'] = df_master['Bilan_sols_kgha']
    df_master['target_fauna_density'] = df_master['fauna_herbivore_biomass_kg_km2']
    df_master['target_parasite_vsi'] = df_master['vector_suitability_index']
    df_master['target_thermal_anomaly'] = df_master['Temperature_C'] - df_master['temp_mean']
    
    cotiere = 100.0 * np.exp(-df_master['dist_to_coast_km'] / 5.0) * (df_master['tide_amplitude_m'] / 5.0)
    fluviale = 50.0 * np.exp(-df_master['dist_to_river_km'] / 1.0)
    df_master['target_erosion_prob'] = np.clip(cotiere + fluviale, 0.0, 100.0)
        
    # 1. Conserver ['Pays_EN', 'Produit', 'ISO'] (elles sont déjà dans le dataframe)
    
    # 2. Supprimer ['closest_mineral_site', 'closest_fossil_site']
    print("[INFO] Suppression des colonnes de sites uniques...")
    df_master = df_master.drop(columns=['closest_mineral_site', 'closest_fossil_site'], errors='ignore')
    
    # 3. One-hot encoding (pd.get_dummies) des variables catégorielles
    print("[INFO] Encodage (pd.get_dummies) des colonnes categorielles...")
    cat_cols_to_encode = ['closest_volcano_type', 'closest_energy_type', 'closest_mineral_type', 'closest_fossil_type']
    for col in cat_cols_to_encode:
        if col in df_master.columns:
            df_master[col] = df_master[col].fillna('Unknown').astype(str)
            
    df_master = pd.get_dummies(df_master, columns=cat_cols_to_encode, dtype=int)
        
    # Save the consolidated dataset
    df_master.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[SUCCESS] Dataset final sauvegarde dans : {OUTPUT_FILE}")
    print(f"  Shape : {df_master.shape[0]} lignes, {df_master.shape[1]} colonnes")
    
    # Check for missing coordinates or targets
    missing_coords = df_master['latitude'].isnull().sum()
    print(f"  Lignes sans coordonnees : {missing_coords}")
    assert missing_coords == 0, "Erreur critique : Certaines lignes n'ont pas de coordonnées."
    
if __name__ == "__main__":
    build_dataset()
