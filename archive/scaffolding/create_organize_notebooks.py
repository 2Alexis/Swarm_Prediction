"""Génère 2 notebooks : organization+processing + exploration+analyse."""
import json, os

def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }
def md(t): return {"cell_type":"markdown","metadata":{},"source":t}
def code(t): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":t}


# ════════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 — data_processing_organized.ipynb
# ════════════════════════════════════════════════════════════════════════

notebook_proc = nb([
md("""# 🗂️ Notebook 1 — Organisation & Traitement des datasets

## Objectifs
1. **Inventaire** complet de tous les datasets disponibles
2. **Réorganisation par catégorie** : 15 dossiers structurés
3. **Traitement** des nouveaux fichiers (IRENA, CMM, Ember, SE4ALL, GLW...)
4. **Production** de fichiers cleaned standardisés (ISO, Année)

## 15 catégories
| Catégorie | Description |
|---|---|
| `atmosphere` | Climat dynamique, émissions CO2/CH4/N2O, qualité air |
| `hydrologie` | Eau (stress, accès, retraits), niveau mer, glace |
| `sol_ecologie` | Sol, biodiversité (IUCN, EPI), forêt, feux |
| `agriculture` | Cultures végétales, intrants, suitability, SPAM |
| `elevage` | Animaux, viande, lait, œufs, maladies WAHIS |
| `peche` | Production halieutique, aquaculture (FAO) |
| `energie` | Production, conso, renouvelables, fossile, IRENA, Ember |
| `geologie` | Minéraux, séismes, volcans |
| `demographie` | Population, santé, vie, fécondité, mortalité |
| `economie` | PIB, commerce, finances, valeur production |
| `climat_indices` | ENSO, NAO, AMO, PDO, indices NOAA |
| `worldclim` | WorldClim BIO 1-19, T, P, élévation |
| `geographie` | Coords pays, centroides, features physiques |
| `shared` | datasets finaux, country_clusters, EcoCrop |
| `misc` | Utilitaires divers |"""),

md("## 1. Imports & configuration"),
code("""import os, sys, io, shutil, glob, zipfile
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

RAW = 'data/raw'
CLN = 'data/cleaned'
CATEGORIES = ['atmosphere','hydrologie','sol_ecologie','agriculture','elevage',
              'peche','energie','geologie','demographie','economie',
              'climat_indices','worldclim','geographie','shared','misc']"""),

md("## 2. Inventaire actuel"),
code("""for base in [RAW, CLN]:
    print(f'\\n=== {base} ===')
    for c in CATEGORIES:
        d = os.path.join(base, c)
        if os.path.isdir(d):
            n_csv = len([f for f in os.listdir(d) if f.endswith('.csv')])
            n_other = len(os.listdir(d)) - n_csv
            print(f'  {c:18s} : {n_csv:3d} csv + {n_other:2d} autres')"""),

md("""## 3. Fonction de catégorisation

Classifie chaque fichier dans une des 15 catégories selon des regex sur le nom de fichier."""),
code("""def categorize(filename):
    n = filename.lower()
    if any(k in n for k in ['edgar_','ch4','co2','n2o','f-gas','emissions','co2_','methane','ozone',
                              'climate_watch','temperature','pm25','mean_temperature','variation_temp',
                              'global_co2','berkeley_earth','owid_co','owid_methane','owid_n2o',
                              'owid_avg_monthly_temp','owid_temp_anomaly','worldclim_bio_extra',
                              'download_cmm','cmm_']):
        return 'atmosphere'
    if any(k in n for k in ['aqueduct','aquastat','hydro','water','sea_level','ocean','psmsl',
                              'cobe','grace','wri_','groundwater','tide','wb_freshwater',
                              'wb_safe_water','wb_sanitation','owid_water']):
        return 'hydrologie'
    if any(k in n for k in ['forest','tree_cover','ecologie','biodiv','iucn','epi_','firms',
                              'fire','ndvi','modis','viirs','gbif','biomass','pedologie','sols',
                              'soil','owid_forest','wb_forest','bilan_nutritif','wood_density',
                              'deforest']):
        return 'sol_ecologie'
    if any(k in n for k in ['spam','crop','yield','fertili','pesticid','land_use_input','landuse',
                              'intrants','arable','agricultur','cereal','vegetable','fruit','plant',
                              'sugar','cotton','tobacco','owid_agri','owid_cereal','owid_food',
                              'wb_agri','wb_cereal','wb_arable','wb_food_production','wb_fertili',
                              'wb_irrigat','production_cultures','ecocrop','gaez','crop_calendar']):
        return 'agriculture'
    if any(k in n for k in ['livestock','meat','milk','egg','cattle','poultry','sheep','pig',
                              'glw','production_animaux','owid_meat','owid_milk','owid_egg',
                              'wahis','donnees quantitatives']):
        return 'elevage'
    if any(k in n for k in ['fish','aqua_culture','aquaculture','fishery','peche','marine_protected',
                              'fishstat','globalproductionfish']):
        return 'peche'
    if any(k in n for k in ['irena','ember','electric','energy','power','grid','solar','wind',
                              'hydroelec','coal','oil','gas_prod','fossil','renewable','reshare',
                              'heatgen','r-elec','r_elec','monthly_capacity','yearly_full_release',
                              'se4all','sustainable_energy','targets_download','wb_electric',
                              'wb_energy','wb_renewable','wb_coal_rents','wb_oil_rents',
                              'wb_natgas_rents','wb_mineral_rents','wb_forest_rents','ember_yearly',
                              'global_power_plants']):
        return 'energie'
    if any(k in n for k in ['mrds','mineral','earthquake','seismic','volcano','volcan',
                              'tectonic','geolog','iron_mine','oil_gas','coal_mine']):
        return 'geologie'
    if any(k in n for k in ['population','mortality','fertility','birth','death','child_mort',
                              'life_exp','migrant','hunger','stunting','wasting','owid_birth',
                              'owid_death','owid_pop','wb_population','wb_birth','wb_death',
                              'wb_child_mortality','wb_life_exp','wb_dependency','wb_pop_',
                              'wb_stunting','wb_wasting','wb_overweight','wb_adult','wb_infant',
                              'wb_school','wb_employ_','wb_unemployment','wb_urban',
                              'wb_internet','wb_mobile','wb_broadband','wb_hospital','wb_physic',
                              'wb_health','wb_hiv','wb_malaria','wb_deaths_communic',
                              'global_hunger','who_pm25','who_ambient']):
        return 'demographie'
    if any(k in n for k in ['gdp','econom','trade','commerce','finance','debt','inflation',
                              'gini','poverty','value_of_production','value_production','wb_gdp',
                              'wb_trade','wb_inflation','wb_gini','wb_poverty','wb_public_debt',
                              'wb_rd_expenditure','wb_manuf','wb_services','wb_agri_value',
                              'fao_value_of','fao_value_production']):
        return 'economie'
    if any(k in n for k in ['climate_index','enso','nao','amo','pdo','soi','oni','ao_lag','ao_index']):
        return 'climat_indices'
    if 'worldclim' in n or n.startswith(('bio','prec','tmax','tmin','tvag','wind_','solrad','vapr','elev')):
        return 'worldclim'
    if n.startswith(('dataset_final','country_','ecocrop','datset_final')):
        return 'shared'
    if any(k in n for k in ['topographie','relief','elevation','slope','centroid','physical_features']):
        return 'geographie'
    return 'misc'

# Test
for f in ['edgar_2025_ch4.csv','wb_population_total.csv','spam2020_v2_yield_by_country.csv',
          'iucn_species_by_country.csv','irena_renewable_targets.csv']:
    print(f'{f:50s} → {categorize(f)}')"""),

md("## 4. Réorganisation (déplace fichiers)"),
code("""# Le script organize_and_process.py a déjà réorganisé.
# Pour ré-exécuter manuellement :
print('Réorganisation déjà appliquée. Pour relancer :')
print('  !python organize_and_process.py')"""),

md("""## 5. Traitement des nouveaux fichiers

### IRENA — Renewable Targets
Format Excel multi-sheets, parse les cibles renouvelables par pays."""),
code("""src = f'{RAW}/energie/targets_download.xlsx'
if os.path.exists(src):
    df = pd.read_excel(src, sheet_name='raw_data_long', engine='openpyxl')
    print(f'IRENA targets : {df.shape}')
    print(df.head(3))
else:
    print(f'⚠️ {src} introuvable')"""),

md("### CMM — Coal Mine Methane Emissions"),
code("""src = f'{RAW}/atmosphere/download_cmm_emissions.xlsx'
if os.path.exists(src):
    df = pd.read_excel(src, sheet_name='emissions', engine='openpyxl')
    print(f'CMM emissions : {df.shape}')
    print(df.head(3))
else:
    print('⚠️ pas dispo')"""),

md("### Ember Yearly Electricity"),
code("""src = f'{RAW}/energie/yearly_full_release_long_format.csv'
if os.path.exists(src):
    df = pd.read_csv(src, low_memory=False)
    print(f'Ember : {df.shape}')
    print(f'Catégories : {df["Category"].value_counts().head().to_dict()}')
else:
    print('⚠️ pas dispo')"""),

md("### SE4ALL — Sustainable Energy for All (WB)"),
code("""src = f'{RAW}/energie/P_Data_Extract_From_Sustainable_Energy_for_All.zip'
if os.path.exists(src):
    with zipfile.ZipFile(src) as z:
        print(f'Fichiers : {z.namelist()}')
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, low_memory=False)
            print(f'Shape : {df.shape}')
            indicators = [c for c in df.columns if '[' in c and ']' in c][:6]
            for ind in indicators:
                print(f'  - {ind[:80]}')"""),

md("### WAHIS — Animal Disease Outbreaks"),
code("""for src in [f'{RAW}/elevage/Données quantitatives 2026-06-24.csv',
             f'{RAW}/demographie/Données quantitatives 2026-06-24.csv',
             f'{RAW}/Données quantitatives 2026-06-24.csv']:
    if os.path.exists(src):
        try: df = pd.read_csv(src, low_memory=False)
        except: df = pd.read_csv(src, low_memory=False, encoding='latin-1')
        print(f'WAHIS : {df.shape}')
        print(f'Maladies top 5 : {df["Maladie"].value_counts().head().to_dict()}')
        break
else:
    print('⚠️ WAHIS non trouvé')"""),

md("## 6. Inventaire final des cleaned"),
code("""print('Fichiers cleaned par catégorie :\\n')
total = 0
for c in CATEGORIES:
    d = os.path.join(CLN, c)
    if os.path.isdir(d):
        files = [f for f in os.listdir(d) if f.endswith('.csv')]
        total += len(files)
        if files:
            print(f'\\n━━ {c} ({len(files)} csv) ━━')
            for f in sorted(files)[:8]:
                sz = os.path.getsize(os.path.join(d,f)) / 1024
                print(f'  {f:50s} ({sz:6.0f} KB)')
            if len(files) > 8:
                print(f'  ... et {len(files)-8} autres')
print(f'\\nTOTAL : {total} fichiers cleaned')"""),

md("""## 7. Récapitulatif du pipeline

```
DATA SOURCES (data/raw/)
   │
   ├── atmosphere/        (EDGAR, CMM, NOAA, WorldClim temp)
   ├── hydrologie/        (Aqueduct, AQUASTAT, PSMSL)
   ├── sol_ecologie/      (EPI, IUCN, GBIF, FIRMS)
   ├── agriculture/       (SPAM, FAO Fertilizers, Pesticides, FAOSTAT)
   ├── elevage/           (FAO Animal Prod, GLW4, WAHIS)
   ├── peche/             (FAO Fish Production)
   ├── energie/           (IRENA, Ember, SE4ALL)
   ├── geologie/          (USGS MRDS, séismes, volcans)
   ├── demographie/       (WB, OWID, WHO)
   ├── economie/          (FAO Value, WB GDP)
   ├── climat_indices/    (NOAA ENSO, NAO, AMO, PDO)
   ├── worldclim/         (WorldClim BIO 1-19)
   ├── geographie/        (Centroids, slopes)
   └── shared/            (datasets finaux, country_clusters, EcoCrop)

   ↓ TRAITEMENT (scripts Python)

CLEANED DATA (data/cleaned/) — même structure
   ↓
   datasets finaux multi-couches (dataset_final_v*_couche*.csv)
```

→ Voir Notebook 2 (exploration) pour visualisations + analyses""")
])


# ════════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — data_exploration_organized.ipynb
# ════════════════════════════════════════════════════════════════════════

notebook_explor = nb([
md("""# 📊 Notebook 2 — Exploration & Analyse Multi-Catégories

## Plan
1. Vue d'ensemble par catégorie (volumes, couvertures)
2. **Analyse Atmosphère** : émissions par secteur EDGAR
3. **Analyse Hydrologie** : water stress + niveau mer
4. **Analyse Sol/Écologie** : EPI Yale + biodiversité IUCN
5. **Analyse Agriculture** : SPAM 2020 par culture
6. **Analyse Élevage** : production animale FAO
7. **Analyse Énergie** : IRENA + Ember
8. **Analyse Démographie** : indicateurs WB + OWID
9. **Corrélations cross-catégories**"""),

md("## 1. Imports"),
code("""import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (12,6)

CLN = 'data/cleaned'
CATEGORIES = ['atmosphere','hydrologie','sol_ecologie','agriculture','elevage',
              'peche','energie','geologie','demographie','economie','shared']"""),

md("## 2. Vue d'ensemble par catégorie"),
code("""counts = {}
for c in CATEGORIES:
    d = f'{CLN}/{c}'
    if os.path.isdir(d):
        csvs = [f for f in os.listdir(d) if f.endswith('.csv')]
        counts[c] = len(csvs)

fig, ax = plt.subplots(figsize=(11, 6))
items = sorted(counts.items(), key=lambda x: -x[1])
cats = [x[0] for x in items]
vals = [x[1] for x in items]
colors = plt.cm.tab20(np.linspace(0,1,len(cats)))
ax.barh(cats, vals, color=colors, alpha=0.85)
for i, v in enumerate(vals):
    ax.text(v+0.5, i, str(v), va='center')
ax.set_xlabel('Nombre de fichiers CSV cleaned')
ax.set_title('Datasets disponibles par catégorie', weight='bold', fontsize=13)
ax.invert_yaxis()
plt.tight_layout(); plt.show()
print(f'TOTAL : {sum(vals)} fichiers cleaned dans {len(cats)} catégories')"""),

md("""## 3. 🌫️ ATMOSPHÈRE — Émissions EDGAR par secteur

EDGAR 2025 = JRC Joint Research Centre. Émissions GHG ventilées par 9-10 secteurs : Buildings, Energy, Industry, Transport, Agriculture, Waste..."""),
code("""edgar_co2 = pd.read_csv(f'{CLN}/atmosphere/edgar_2025_fossil_co2.csv')
print(f'EDGAR CO2: {edgar_co2.shape}, {edgar_co2["ISO"].nunique()} pays, années {edgar_co2["Annee"].min()}-{edgar_co2["Annee"].max()}')

# Évolution mondiale par secteur
edgar_cols = [c for c in edgar_co2.columns if c.startswith('edgar_fossil_co2_')]
world = edgar_co2.groupby('Annee')[edgar_cols].sum() / 1e3  # → Mt
fig, ax = plt.subplots(figsize=(13, 6))
world.plot(kind='area', stacked=True, ax=ax, alpha=0.8, colormap='Set2')
ax.set_title('Émissions CO2 fossiles mondiales par secteur (Mt CO2/an)',
              weight='bold', fontsize=13)
ax.set_xlabel('Année'); ax.set_ylabel('Mt CO2/an')
ax.legend([c.replace('edgar_fossil_co2_','') for c in edgar_cols],
           bbox_to_anchor=(1.02,1), loc='upper left', fontsize=9)
plt.tight_layout(); plt.show()"""),

code("""# Top 10 pays émetteurs CO2 en 2022
top2022 = edgar_co2[edgar_co2['Annee']==2022].assign(total=edgar_co2[edgar_cols].sum(axis=1)).nlargest(10,'total')
fig, ax = plt.subplots(figsize=(11,6))
ax.bar(top2022['ISO'], top2022['total']/1e3, color='firebrick', alpha=0.85)
ax.set_title('Top 10 émetteurs CO2 fossile 2022 (Mt)', weight='bold')
ax.set_ylabel('Émissions Mt CO2')
plt.tight_layout(); plt.show()"""),

md("## 4. 💧 HYDROLOGIE — WRI Aqueduct 4.0 + Sea Level"),
code("""aq = pd.read_csv(f'{CLN}/hydrologie/wri_aqueduct40_by_country.csv')
print(f'Aqueduct : {aq.shape}, {aq["ISO"].nunique()} pays')
print(f'Indicateurs : {list(aq.columns)[2:8]}')

# Top 15 pays par water stress (bws)
if 'aqueduct_bws_score' in aq.columns:
    top = aq.nlargest(15, 'aqueduct_bws_score')
    fig, ax = plt.subplots(figsize=(11,6))
    ax.barh(top['ISO'], top['aqueduct_bws_score'], color='royalblue', alpha=0.85)
    ax.set_title('Top 15 pays — Water Stress (BWS Aqueduct 2023)', weight='bold')
    ax.set_xlabel('Score 0-5')
    ax.invert_yaxis()
    plt.tight_layout(); plt.show()"""),

code("""# Sea Level PSMSL
psmsl = pd.read_csv(f'{CLN}/hydrologie/psmsl_sea_level_by_country.csv')
print(f'PSMSL : {psmsl.shape}, {psmsl["Pays"].nunique()} pays')

# Évolution mondiale sea level
world_sl = psmsl.groupby('Annee')['sea_level_mean'].mean().rolling(5, center=True).mean()
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(world_sl.index, world_sl.values, lw=2, color='navy')
ax.fill_between(world_sl.index, world_sl.values, alpha=0.3, color='navy')
ax.set_title('Niveau marin moyen mondial (PSMSL, moyenne mobile 5y)',
              weight='bold', fontsize=12)
ax.set_xlabel('Année'); ax.set_ylabel('mm relatif')
plt.tight_layout(); plt.show()"""),

md("## 5. 🌿 SOL & ÉCOLOGIE — EPI Yale + IUCN"),
code("""epi = pd.read_csv(f'{CLN}/sol_ecologie/epi_2024_indicators.csv')
print(f'EPI : {epi.shape}, indicateurs : {[c for c in epi.columns if c.startswith("epi_")][:10]}')

# Évolution EPI score global (si dispo)
epi_main = [c for c in epi.columns if c.startswith('epi_') and 'epi' in c.split('_')[1].lower()]
print(f'Indicateurs principaux EPI : {epi_main[:5]}')"""),

code("""# IUCN espèces menacées top 20 pays
iucn = pd.read_csv(f'{CLN}/sol_ecologie/iucn_species_by_country.csv')
print(f'IUCN : {len(iucn)} pays, total {iucn["iucn_observations"].sum():,} observations')

top = iucn.nlargest(20, 'iucn_species_count')
fig, ax = plt.subplots(figsize=(11,7))
ax.barh(top['ISO'], top['iucn_species_count'], color='darkgreen', alpha=0.85)
ax.set_title('Top 20 pays — Diversité espèces menacées IUCN (observations)', weight='bold')
ax.set_xlabel('Nombre d\\'espèces uniques observées')
ax.invert_yaxis()
plt.tight_layout(); plt.show()"""),

md("## 6. 🌾 AGRICULTURE — SPAM 2020 V2 (46 cultures)"),
code("""spam = pd.read_csv(f'{CLN}/agriculture/spam2020_v2_yield_by_country.csv')
print(f'SPAM : {spam.shape}, {spam["ISO"].nunique()} pays')

crop_cols = [c for c in spam.columns if c.startswith('spam_yield_')]
# Top 15 pays par yield moyen tous crops
spam['spam_avg_yield'] = spam[crop_cols].mean(axis=1)
top = spam.nlargest(15, 'spam_avg_yield')
fig, ax = plt.subplots(figsize=(11,6))
ax.barh(top['ISO'], top['spam_avg_yield'], color='forestgreen', alpha=0.85)
ax.set_title('Top 15 pays — Rendement moyen 46 cultures (SPAM 2020 V2)', weight='bold')
ax.set_xlabel('Yield moyen (kg/ha)')
ax.invert_yaxis()
plt.tight_layout(); plt.show()"""),

code("""# Heatmap 15 cultures clés × top 15 pays
key_crops = ['whea','rice','maiz','soyb','pota','toma','bana','citr','coff','grou','sunf','rape','cott','sugc','vege']
key_cols = [f'spam_yield_{c}' for c in key_crops if f'spam_yield_{c}' in spam.columns]
top15 = spam.nlargest(15, 'spam_avg_yield')
mat = top15.set_index('ISO')[key_cols]
# log
mat_log = np.log1p(mat)
fig, ax = plt.subplots(figsize=(13,7))
sns.heatmap(mat_log, cmap='YlGn', annot=False, cbar_kws={'label':'log(kg/ha)'}, ax=ax)
ax.set_title('Rendements SPAM 2020 — 15 cultures × Top 15 pays', weight='bold', fontsize=12)
ax.set_xticklabels([c.replace('spam_yield_','') for c in mat.columns], rotation=30)
plt.tight_layout(); plt.show()"""),

md("## 7. 🐄 ÉLEVAGE — Production FAO + WAHIS"),
code("""# Tous les fichiers élevage
fs = sorted([f for f in os.listdir(f'{CLN}/elevage') if f.endswith('.csv')])
print(f'Fichiers élevage : {fs}')
for f in fs[:3]:
    df = pd.read_csv(f'{CLN}/elevage/{f}')
    print(f'\\n{f} : {df.shape}, cols={list(df.columns)[:6]}')"""),

md("## 8. 🐟 PÊCHE — FAO Global Production Fish"),
code("""fish = pd.read_csv(f'{CLN}/peche/fish_production_total.csv')
print(f'Fish : {fish.shape}')

top10 = fish[fish['Annee']==2022].nlargest(10, 'fish_total_t')
fig, ax = plt.subplots(figsize=(11,6))
ax.barh(top10['ISO'], top10['fish_total_t']/1e6, color='steelblue', alpha=0.85)
ax.set_title('Top 10 producteurs halieutiques 2022 (M tonnes)', weight='bold')
ax.invert_yaxis()
plt.tight_layout(); plt.show()"""),

md("## 9. ⚡ ÉNERGIE — Ember + IRENA"),
code("""ember = pd.read_csv(f'{CLN}/energie/ember_electricity_by_source.csv')
print(f'Ember : {ember.shape}')
print(f'Sources dispo : {[c for c in ember.columns if c.startswith("ember_")][:10]}')

# Évolution mondiale renouvelable vs fossile
src_cols = [c for c in ember.columns if c.startswith('ember_')][:8]
world = ember.groupby('Annee')[src_cols].sum()
fig, ax = plt.subplots(figsize=(13,6))
world.plot(kind='area', stacked=True, ax=ax, alpha=0.75, colormap='Set2')
ax.set_title('Production électrique mondiale par source (Ember, TWh)',
              weight='bold', fontsize=13)
ax.set_xlabel('Année'); ax.set_ylabel('TWh')
ax.legend([c.replace('ember_','').replace('_twh','') for c in src_cols], fontsize=9)
plt.tight_layout(); plt.show()"""),

md("## 10. 👥 DÉMOGRAPHIE — Indicateurs WB principaux"),
code("""# Charger 4 indicateurs clés
def load_wb(name, value_col_name):
    f = f'{CLN}/demographie/wb_{name}.csv'
    if not os.path.exists(f): return None
    df = pd.read_csv(f)
    if 'Annee' in df.columns and len(df.columns) >= 3:
        val_col = [c for c in df.columns if c not in ('Pays','Annee','ISO3','ISO')][0]
        df = df.rename(columns={val_col: value_col_name})
        return df[['Pays','Annee',value_col_name]]
    return None

datasets = {
    'Population': load_wb('population_total','pop'),
    'Life Expectancy': load_wb('life_expectancy','life_exp'),
    'Child Mortality': load_wb('child_mortality','child_mort'),
    'Fertility': load_wb('birth_rate','birth_rate'),
}
for name, df in datasets.items():
    if df is not None:
        print(f'  {name} : {df.shape}')"""),

md("## 11. 🔗 Corrélations Cross-Catégories — émissions CO2 vs niveau marin vs température"),
code("""# Charger 4 séries mondiales agrégées
co2 = edgar_co2.assign(total=edgar_co2[edgar_cols].sum(axis=1))
co2_world = co2.groupby('Annee')['total'].sum() / 1e3  # Mt

sl_world = psmsl.groupby('Annee')['sea_level_mean'].mean()

# T globale via Berkeley si dispo
be_path = f'{CLN}/atmosphere/berkeley_earth_yearly.csv'
if os.path.exists(be_path):
    be = pd.read_csv(be_path)
    t_world = be.groupby('Annee')['be_t_anom_annual'].mean()
else:
    t_world = None

fig, axes = plt.subplots(1, 3, figsize=(18,5))

axes[0].plot(co2_world.index, co2_world.values, color='firebrick', lw=2)
axes[0].set_title('CO2 fossile mondial (Mt)', weight='bold')
axes[0].set_xlabel('Année')

axes[1].plot(sl_world.index, sl_world.values, color='navy', lw=2)
axes[1].set_title('Niveau marin moyen (mm)', weight='bold')
axes[1].set_xlabel('Année')

if t_world is not None:
    axes[2].plot(t_world.index, t_world.values, color='orange', lw=2)
    axes[2].set_title('Anomalie T mondiale (Berkeley)', weight='bold')
    axes[2].set_xlabel('Année')

plt.tight_layout(); plt.show()"""),

md("""## 12. Synthèse — Couverture par catégorie

```
ATMOSPHÈRE (22 csv)
  ⭐ EDGAR 2025 par secteur (CO2/CH4/N2O/F-gases)
  ⭐ Berkeley Earth (1743-2020)
  ✓ WHO PM2.5, OWID temp anomalies, CMM

HYDROLOGIE (13 csv)
  ⭐ WRI Aqueduct 4.0 (40 indicateurs water risk)
  ⭐ AQUASTAT bulk (40 variables)
  ⭐ PSMSL Sea Level (1750-2024)

SOL & ÉCOLOGIE (17 csv)
  ⭐ EPI 2024 Yale (31 indicateurs environnementaux)
  ⭐ IUCN Red List (4 780 espèces, 245 pays)
  ✓ FIRMS Fires recent, OWID forest

AGRICULTURE (22 csv)
  ⭐ SPAM 2020 V2 (46 cultures × 164 pays)
  ⭐ FAO N/P/K + Pesticides catégorisés
  ✓ EcoCrop suitability

ÉLEVAGE (8 csv) — manque WAHIS, GLW à intégrer

PÊCHE (6 csv) — FAO complet 1950-2022

ÉNERGIE (23 csv)
  ⭐ Ember 16 sources électriques
  ⭐ IRENA renewable targets
  ✓ SE4ALL, WB renewable %, fossile production

DÉMOGRAPHIE (46 csv) — WB + OWID complets
ÉCONOMIE (13 csv)
GÉOLOGIE (6 csv)
CLIMAT INDICES (6 csv) — NOAA ENSO/NAO/AMO/PDO
WORLDCLIM, SHARED, GEOGRAPHIE
```

### Total : ~200 fichiers cleaned organisés par catégorie

→ Prêt pour pipeline cascade V3 avec tous les nouveaux datasets""")
])


# ════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════
for name, nb_data in [
    ("data_processing_organized.ipynb", notebook_proc),
    ("data_exploration_organized.ipynb", notebook_explor),
]:
    with open(name, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, ensure_ascii=False, indent=1)
    print(f"[OK] {name}  ({len(nb_data['cells'])} cellules)")
print("\n[DONE]")
