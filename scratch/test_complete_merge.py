import os
import pandas as pd
import pycountry
import babel
import sys

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "data/cleaned"

# 1. Custom French to ISO map
from test_final_mapping import custom_mappings

# 2. Custom English to ISO map
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

def test_merge():
    # Load all cleaned CSVs
    df_prod = pd.read_csv(os.path.join(DATA_DIR, "production_cultures.csv"))
    df_temp = pd.read_csv(os.path.join(DATA_DIR, "mean_temperature.csv"))
    df_precip = pd.read_csv(os.path.join(DATA_DIR, "precipitations.csv"))
    df_fert = pd.read_csv(os.path.join(DATA_DIR, "fertilizers_nutrient.csv"))
    df_pest = pd.read_csv(os.path.join(DATA_DIR, "pesticides.csv"))
    df_sols = pd.read_csv(os.path.join(DATA_DIR, "bilan_nutritif_sols.csv"))
    
    # Socio-demographics and health
    df_gdp = pd.read_csv(os.path.join(DATA_DIR, "wb_gdp_per_capita.csv"))
    df_life = pd.read_csv(os.path.join(DATA_DIR, "wb_life_expectancy.csv"))
    df_child = pd.read_csv(os.path.join(DATA_DIR, "wb_child_mortality.csv"))
    df_hdi = pd.read_csv(os.path.join(DATA_DIR, "owid_hdi.csv"))
    df_malaria = pd.read_csv(os.path.join(DATA_DIR, "wb_malaria_incidence.csv"))
    df_hydro = pd.read_csv(os.path.join(DATA_DIR, "wb_freshwater_withdrawal_pct.csv"))

    # Map country names / codes to ISO-2
    # 1. French names
    for df in [df_prod, df_fert, df_pest, df_sols]:
        df['Pays_Clean'] = df['Pays'].str.strip().str.lower()
        df['ISO'] = df['Pays_Clean'].map(custom_mappings)
        
    # 2. English 3-letter codes
    for df in [df_temp, df_precip]:
        df['ISO'] = df['Code_Pays'].apply(get_iso_from_alpha3)
        
    # 3. English names
    for df in [df_gdp, df_life, df_child, df_hdi, df_malaria, df_hydro]:
        df['ISO'] = df['Pays'].apply(get_english_iso)

    # Filter out rows without ISO
    df_prod = df_prod.dropna(subset=['ISO'])
    df_temp = df_temp.dropna(subset=['ISO'])
    df_precip = df_precip.dropna(subset=['ISO'])
    df_fert = df_fert.dropna(subset=['ISO'])
    df_pest = df_pest.dropna(subset=['ISO'])
    df_sols = df_sols.dropna(subset=['ISO'])
    df_gdp = df_gdp.dropna(subset=['ISO'])
    df_life = df_life.dropna(subset=['ISO'])
    df_child = df_child.dropna(subset=['ISO'])
    df_hdi = df_hdi.dropna(subset=['ISO'])
    df_malaria = df_malaria.dropna(subset=['ISO'])
    df_hydro = df_hydro.dropna(subset=['ISO'])

    # Aggregate yields
    df_yield = df_prod[df_prod['Element'] == 'Rendement'].copy().rename(columns={'Valeur': 'Rendement_kgha'})
    df_yield = df_yield[df_yield['Rendement_kgha'] <= 100000]
    df_yield = df_yield[df_yield['Rendement_kgha'] > 0]
    
    df_yield_agg = df_yield.groupby(['ISO', 'Produit', 'Annee'])['Rendement_kgha'].mean().reset_index()
    
    # Filter df_sols to get only the overall soil nutrient balance ("Bilan nutritif des sols")
    # and group it by ISO and Annee (not Produit) since it is country-level.
    df_sols_f = df_sols[df_sols['Produit'].str.lower().str.contains('bilan nutritif des sols', na=False)].copy()
    df_sols_agg = df_sols_f.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Bilan_sols_kgha'})
    
    df_temp_agg = df_temp.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Temperature_C'})
    df_precip_agg = df_precip.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Precipitations_mm'})
    df_fert_agg = df_fert.groupby(['ISO', 'Annee'])['Valeur'].sum().reset_index().rename(columns={'Valeur': 'Engrais_kgha'})
    df_pest_agg = df_pest.groupby(['ISO', 'Annee'])['Valeur'].sum().reset_index().rename(columns={'Valeur': 'Pesticides_kgha'})
    
    df_gdp_agg = df_gdp.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'GDP_pc'})
    df_life_agg = df_life.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Life_Exp'})
    df_child_agg = df_child.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Child_Mort'})
    df_hdi_agg = df_hdi.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'HDI'})
    df_malaria_agg = df_malaria.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Malaria_Incidence'})
    df_hydro_agg = df_hydro.groupby(['ISO', 'Annee'])['Valeur'].mean().reset_index().rename(columns={'Valeur': 'Water_Withdrawal_pct'})

    # Master left join starting with crop yield
    df_master = df_yield_agg.copy()
    df_master = pd.merge(df_master, df_sols_agg, on=['ISO', 'Annee'], how='left') # Joined by ISO and Annee
    df_master = pd.merge(df_master, df_temp_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_precip_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_fert_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_pest_agg, on=['ISO', 'Annee'], how='left')
    
    # Merge socio-demographics & health
    df_master = pd.merge(df_master, df_gdp_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_life_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_child_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_hdi_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_malaria_agg, on=['ISO', 'Annee'], how='left')
    df_master = pd.merge(df_master, df_hydro_agg, on=['ISO', 'Annee'], how='left')

    # Filter post-2010
    df_master = df_master[df_master['Annee'] >= 2010]
    
    print("Master shape:", df_master.shape)
    print("Columns:", df_master.columns.tolist())
    print("\nMissing values per column:")
    print(df_master.isnull().sum())
    print("\nUnique countries mapped:", df_master['ISO'].nunique())
    print("\nSample rows:")
    print(df_master.dropna(subset=['Temperature_C', 'GDP_pc', 'Bilan_sols_kgha']).head(5))

if __name__ == "__main__":
    test_merge()
