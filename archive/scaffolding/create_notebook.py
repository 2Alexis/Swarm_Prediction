"""
create_notebook.py — Génère les 3 notebooks principaux du projet (V12) :
  1. data_processing.ipynb       — pipeline V1→V12 complet
  2. data_visualization.ipynb    — corrélations + plots V12 (cultures spé, suitability, religion)
  3. modelisation_complete.ipynb — entraînement multi-couches (legacy, voir couche*/notebook par couche)
"""
import json
import os

OUT_DIR = "."

def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}


# ════════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 — data_processing.ipynb (V12)
# ════════════════════════════════════════════════════════════════════════

notebook1 = nb([
md("""# 🌍 Notebook 1 — Pipeline de Building V12

> Construction complète des datasets depuis les sources brutes jusqu'à `dataset_final_v12_couche1.csv`.
> Final V12 : **8 400 lignes × 741 colonnes** (240 pays, 1990-2024, 8 clusters climatiques).

## Pipeline en 12 étapes

| Version | Contenu | Cibles ajoutées |
|---|---|---|
| V1-V2 | Base FAO + WB + WorldClim BIO 1-19 | 6 yields agrégés |
| V3 | NOAA climate + OWID disasters + UCDP + USGS | catastrophes |
| V4 | 39 WB + BIO 2-19 + features dérivées | — |
| V5 | NASA POWER + WHO PM2.5 + forêt OWID | soil_moisture, forest |
| V6 | 36 cultures spécifiques + CRU TS | cultures par espèce |
| V7 | Berkeley Earth 1743-2020 | anomalies multi-décennales |
| **V8 honnête** | **Imputation features + clustering KMeans** | — |
| **V9** | **Couche 1 : élevage + énergie + émissions** | 19 nouvelles cibles |
| **V10** | **Cultures spécifiques cibles + religion Pew + meat per type** | 36 cibles cultures |
| **V11** | **FAO EcoCrop suitability scores** | — (features) |
| **V12** | **Suitability adoucie + indices stress climatique** | — (features) |

> Note : pour seulement charger le dataset final, aller directement à la section 8."""),

md("## 1. Imports & configuration"),
code("""import os, sys, io, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

DATA_DIR = 'data/cleaned'
RAW_DIR  = 'data/raw'
print(f'Datasets cleaned présents : {len([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])}')"""),

md("""## 2. Mapping noms de pays → codes ISO

Standardisation : noms français FAO, noms anglais WB/OWID, codes alpha-3 FAO → ISO alpha-2 commun."""),
code("""from build_dataset import custom_mappings, get_english_iso, get_iso_from_alpha3

print(f'Mappings FR custom : {len(custom_mappings)} pays')
print(f'  "United States" → {get_english_iso("United States")}')
print(f'  "USA" → {get_iso_from_alpha3("USA")}')"""),

md("""## 3. Téléchargement initial (V1-V7)

Tous les datasets externes téléchargés via scripts dédiés."""),
code("""# Décommenter pour télécharger tous les datasets externes (~30 min)
# !python download_extra.py        # NOAA indices + OWID catastrophes
# !python download_more.py         # 39 WB indicators
# !python temporalize_geo.py       # Séismes/centrales par année
# !python fetch_nasa_power.py      # NASA POWER (~15 min)
# !python fetch_berkeley_earth.py  # Berkeley Earth (~10 min)
# !python fetch_more_datasets.py   # WHO + GFW + extras
# !python fetch_hansen.py          # Forêt via OWID
# !python add_worldclim_bio.py     # BIO 1-19 sur bbox pays
print('Scripts disponibles. Décommenter pour relancer.')"""),

md("""## 4. Build V1-V7 (base + enrichissements)

```bash
python build_dataset_v2.py    # V2 base
python enrich_v2.py           # V2 → V3 (NOAA + UCDP)
python enrich_v3.py           # V3 → V4 (39 WB + BIO 2-19)
python enrich_v4.py           # V4 → V5 (NASA POWER + WHO)
python enrich_v5.py           # V5 → V6 (cultures spé + CRU)
python enrich_v6.py           # V6 → V7 (Berkeley Earth)
```"""),
code("""# État des datasets V1-V7
for v in ['v2','v3','v4','v5','v6','v7']:
    f = f'{DATA_DIR}/dataset_final_{v}.csv'
    if os.path.exists(f):
        sz = os.path.getsize(f) / 1e6
        print(f'  ✓ {f}  ({sz:.1f} MB)')
    else:
        print(f'  ✗ {f} absent')"""),

md("""## 5. Imputation honnête V7 → V8

Étape clé : on impute **uniquement les features explicatives**, jamais les cibles ni leurs sources.

### Règles d'imputation
- **Cibles** (`target_*`) et **sources brutes** : intouchables (NaN conservés)
- **Features dynamiques** : interpolation linéaire par pays (limite=5 ans), puis ffill/bfill
- **Features statiques manquantes** : médiane par (cluster × année)

### Clustering
- KMeans 8 clusters sur 15 features environnementales statiques
- Servira en feature additionnelle (Global+Cluster)"""),
code("""# Imputation + clustering (~30s)
if not os.path.exists(f'{DATA_DIR}/dataset_final_v8_honest.csv'):
    print('Lancement: !python impute_honest.py')
else:
    print(f'✓ dataset_final_v8_honest.csv déjà présent')"""),

md("""## 6. Enrichissements Couche 1 (V8 → V9 → V10 → V11 → V12)

Spécifique au domaine **Planète** (environnement + agriculture + élevage).

### V9 — Élevage + Énergie + Émissions
- FAO `production_animaux` : lait, viandes (bovine/poulet/ovin/porc), œufs
- OWID : émissions CO2/CH4/N2O annuelles par pays
- OWID : énergies (solaire, éolien, hydro, charbon, pétrole, gaz)
- OWID : aquaculture, accès eau potable, marine protected

### V10 — Cultures spécifiques + Religion + Meat per type
- **36 cibles `target_yield_X`** (au lieu d'agrégats faibles) : soja, colza, tomate, pomme, banane...
- **Religion Pew 2010** : % muslim/christian/hindu/buddhist par pays (statique)
- **OWID Meat per type** : beef, pig, poultry, sheep/goat, milk, eggs consumption

### V11 — EcoCrop Suitability
- Téléchargement **FAO EcoCrop database** (2568 espèces)
- Pour 39 cultures principales : T_opt, T_abs, P_opt, P_abs, killing temp, latitude range
- Score 0-1 par (pays, année, culture)

### V12 — Suitability adoucie + Stress climatique
- Suitability EcoCrop **sans zéros stricts** (frost = pénalisation continue)
- **Heat stress index** : (T - 25°C) clipped
- **Frost risk index** : (-T_min) clipped
- **Aridity index** : P / PET (Thornthwaite)
- **Growing season index** v2
- **Continentalité** : amplitude T"""),
code("""# Build Couche 1 V9-V12 (~5 min)
print('Pour rebuild Couche 1 :')
print('  !python couche1_planete/enrich_couche1.py        # V8 → V9')
print('  !python couche1_planete/fetch_religion_meat.py   # téléchargement extras')
print('  !python couche1_planete/enrich_couche1_v2.py     # V9 → V10')
print('  !python couche1_planete/compute_suitability.py   # V10 → V11')
print('  !python couche1_planete/enrich_v12.py            # V11 → V12')

for v in ['v8_honest','v9_couche1','v10_couche1','v11_couche1','v12_couche1']:
    f = f'{DATA_DIR}/dataset_final_{v}.csv'
    if os.path.exists(f):
        sz = os.path.getsize(f) / 1e6
        print(f'  ✓ {f}  ({sz:.1f} MB)')
    else:
        print(f'  ✗ {f} absent')"""),

md("## 7. Enrichissement Couche 2 (le Sang)"),
code("""# Couche 2 utilise dataset_final_v8_honest.csv directement
# Pas d'enrichissement spécifique nécessaire — les cibles démographiques
# (Birth_Rate, Death_Rate, Fertility, Child_Mort) sont déjà dans V8.
print('Couche 2 charge directement dataset_final_v8_honest.csv')
print('  → 10 cibles démo/santé/catastrophes humaines')"""),

md("## 8. Chargement du dataset Couche 1 V12"),
code("""# Charge V12 si disponible, sinon fallback
for path, version in [(f'{DATA_DIR}/dataset_final_v12_couche1.csv', 'V12'),
                       (f'{DATA_DIR}/dataset_final_v11_couche1.csv', 'V11'),
                       (f'{DATA_DIR}/dataset_final_v10_couche1.csv', 'V10'),
                       (f'{DATA_DIR}/dataset_final_v9_couche1.csv',  'V9'),
                       (f'{DATA_DIR}/dataset_final_v8_honest.csv',   'V8')]:
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        print(f'✓ Chargé : {path} ({version})')
        break

df = df.dropna(subset=['ISO']).copy()
df['ISO'] = df['ISO'].astype(str)
df['Annee'] = df['Annee'].astype(int)
if 'cluster' in df.columns:
    df['cluster'] = df['cluster'].astype(int)

print(f'Shape: {df.shape}')
print(f'Pays: {df["ISO"].nunique()} | Années: {df["Annee"].min()}–{df["Annee"].max()}')
if 'cluster' in df.columns:
    print(f'Clusters: {df["cluster"].nunique()}')"""),

md("## 9. Vérification qualité — cibles disponibles"),
code("""targets = [c for c in df.columns if c.startswith('target_')]
print(f'Cibles ({len(targets)}) — taux de non-null :\\n')
# Grouper par préfixe
from collections import defaultdict
groups = defaultdict(list)
for t in targets:
    if 'yield_' in t: groups['Agriculture/Élevage'].append(t)
    elif any(k in t for k in ['stress','degradation','thermal','moisture','water_access']): groups['Environnement'].append(t)
    elif any(k in t for k in ['forest','tree','marine']): groups['Écologie'].append(t)
    elif any(k in t for k in ['co2','methane','n2o']): groups['Émissions'].append(t)
    elif any(k in t for k in ['solar','wind','hydro','coal','oil','gas']): groups['Énergie'].append(t)
    elif any(k in t for k in ['birth','death','mortality','fertility','migration','growth','life_exp']): groups['Démographie'].append(t)
    elif any(k in t for k in ['disaster']): groups['Catastrophes'].append(t)
    elif any(k in t for k in ['livestock','milk','carcass','eggs','aquaculture']): groups['Élevage'].append(t)
    else: groups['Autres'].append(t)

for g, ts in groups.items():
    print(f'\\n━━ {g} ({len(ts)} cibles) ━━')
    for t in sorted(ts)[:30]:
        n = df[t].notna().sum()
        pct = n/len(df)*100
        print(f'  {t:40s} {n:6,d} ({pct:4.0f}%)')"""),

md("## 10. NaN — avant vs après imputation V8"),
code("""nan_pct_total = df.isna().sum().sum() / (df.shape[0]*df.shape[1]) * 100
print(f'NaN total V12 : {nan_pct_total:.1f}%')

# Features imputables uniquement (hors cibles)
target_sources = ['yield_cereals_kgha','yield_oilcrops_kgha','yield_pulses_kgha','yield_roots_kgha',
                  'yield_fruits_kgha','yield_vegetables_kgha','Water_Withdrawal_pct','Bilan_sols_kgha',
                  'T_anomaly','Child_Mort','Life_Exp','Pop_Growth','Birth_Rate','Death_Rate',
                  'Net_Migration','disaster_deaths','disaster_affected','stunting_pct',
                  'Fertility_Rate','nasa_gwetroot','forest_share_pct','tree_cover_loss_ha']
untouchable = set(targets) | set(target_sources)
for src in target_sources:
    untouchable |= {c for c in df.columns if c.startswith(src + '_')}

imputable = [c for c in df.columns if c not in untouchable and c not in ('ISO','Annee')
             and df[c].dtype in ('float64','int64')]
nan_in_imputable = df[imputable].isna().sum().sum() / (df.shape[0]*len(imputable)) * 100
print(f'NaN dans features imputables : {nan_in_imputable:.2f}%  (≈0 attendu)')"""),

md("## 11. Profil des 8 clusters climatiques"),
code("""cluster_profile = pd.read_csv(f'{DATA_DIR}/country_clusters.csv')
profile_stats = cluster_profile.groupby('cluster').agg(
    n_pays=('ISO', 'nunique'),
    T_mean=('temp_mean', 'mean'),
    P_mean=('precip_mean', 'mean'),
    Lat=('latitude', 'mean'),
    Elev=('elevation', 'mean'),
).round(1)
print('Profil moyen par cluster :\\n')
print(profile_stats.to_string())
print('\\nExemples par cluster :')
for c in sorted(cluster_profile['cluster'].unique()):
    members = cluster_profile[cluster_profile['cluster']==c]['ISO'].head(8).tolist()
    n = len(cluster_profile[cluster_profile["cluster"]==c])
    print(f'  Cluster {c} ({n} pays) : {members}')"""),

md("## 12. Familles de variables V12"),
code("""GROUPS = {
    'Climat dynamique':      [c for c in df.columns if any(k in c for k in ['T_annual','P_annual','T_anomaly','P_anomaly','nasa_t2m','nasa_prec','be_t_anom'])],
    'Climat WorldClim':      [c for c in df.columns if c.startswith('bio') or c in ('temp_mean','precip_mean','wind_speed_mean')],
    'Stress climatique':     [c for c in df.columns if any(k in c for k in ['heat_stress','frost_risk','aridity','pet_annual','continentality','growing_season','climate_zone'])],
    'Suitability EcoCrop':   [c for c in df.columns if c.startswith('suit_')],
    'Sol & humidité':        [c for c in df.columns if any(k in c for k in ['clay','silt','sand','soil_pH','organic_carbon','gwet','Bilan_sols'])],
    'Agriculture':           [c for c in df.columns if any(k in c for k in ['Engrais','Pesticides','Terres','Bio_','Irrigation']) or (c.startswith('yield_') and 'kgha' in c)],
    'Élevage':               [c for c in df.columns if any(k in c for k in ['livestock','milk','meat_','eggs','aquaculture','dairy'])],
    'Religion (Pew)':        [c for c in df.columns if c.startswith('share_') and ('muslim' in c or 'christian' in c or 'hindu' in c or 'buddhist' in c or 'folk' in c or 'unaffiliated' in c or 'jew' in c)],
    'Géographie statique':   [c for c in df.columns if any(k in c for k in ['latitude','longitude','elevation','slope','dist_to_'])],
    'Démographie':           [c for c in df.columns if any(k in c for k in ['Population','GDP','HDI','Child_Mort','Life_Exp','Birth_Rate','Fertility','Pop_Growth','Urban_pct'])],
    'Catastrophes':          [c for c in df.columns if 'disaster' in c or 'earthquake' in c or 'volcan' in c],
    'Énergie':               [c for c in df.columns if any(k in c for k in ['elec_','Energy','Renew','Electricity','solar_','wind_','hydro_','coal_','oil_','gas_'])],
    'Indices climatiques':   [c for c in df.columns if any(k in c for k in ['enso','nao','amo','pdo','soi','ao_lag','co2_ppm'])],
    'Forêt':                 [c for c in df.columns if any(k in c for k in ['forest_','tree_cover','deforest'])],
    'Conflits':              [c for c in df.columns if 'conflict' in c],
    'Émissions':             [c for c in df.columns if any(k in c for k in ['co2_','methane','n2o'])],
    'Cluster':               ['cluster'],
}
print(f"{'Famille':<25s} {'#cols':>6s}")
print('─' * 35)
for g, cols in GROUPS.items():
    cols = [c for c in cols if c in df.columns]
    print(f'  {g:<25s} {len(cols):>5d}')"""),

md("## 13. Récapitulatif final V12"),
code("""print('═' * 60)
print('PIPELINE V12 — RÉCAPITULATIF')
print('═' * 60)
print(f'Shape         : {df.shape}')
print(f'Pays uniques  : {df["ISO"].nunique()}')
print(f'Année min/max : {df["Annee"].min()}–{df["Annee"].max()}')
if 'cluster' in df.columns:
    print(f'Clusters      : {df["cluster"].nunique()} (climat-éco)')
print(f'Cibles ML     : {len(targets)}')
print(f'NaN total     : {nan_pct_total:.1f}%')
print(f'Sources       : FAO + WB + OWID + WorldClim + NASA POWER + Berkeley Earth')
print(f'                + EcoCrop + Pew Research + WHO + USGS + NOAA + UCDP')
print()
print('→ Voir notebook 2 (visualization) et 3 (modelisation_complete).')
print('→ Ou par couche : couche1_planete/modelisation_couche1.ipynb, couche2_sang/modelisation_couche2.ipynb')""")
])


# ════════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — data_visualization.ipynb (V12)
# ════════════════════════════════════════════════════════════════════════

notebook2 = nb([
md("""# 📊 Notebook 2 — Visualisation & Analyses Exploratoires V12

Plots sur `dataset_final_v12_couche1.csv` (8 400 × 741, 62 cibles, 240 pays, 8 clusters).

## Sections
1. Couverture des 62 cibles
2. **Cultures spécifiques** — rendements par culture
3. **Profil des 8 clusters climatiques**
4. **EcoCrop suitability scores** — visualisation par culture/cluster
5. **Religion Pew 2010** — distributions par pays
6. **Meat consumption per type** — habitudes alimentaires
7. **Stress climatique** (heat/frost/aridity/PET)
8. Corrélations features → cibles (par groupe)
9. Évolution temporelle 4 cibles × 8 pays
10. Plafond de prédictibilité par cible"""),

md("## 1. Imports & chargement V12"),
code("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 90

# Charge V12 (Couche 1) avec fallback
for p in ['data/cleaned/dataset_final_v12_couche1.csv',
          'data/cleaned/dataset_final_v11_couche1.csv',
          'data/cleaned/dataset_final_v10_couche1.csv',
          'data/cleaned/dataset_final_v9_couche1.csv',
          'data/cleaned/dataset_final_v8_honest.csv']:
    if os.path.exists(p):
        df = pd.read_csv(p, low_memory=False)
        print(f'✓ Chargé : {p}')
        break

df = df.dropna(subset=['ISO']).copy()
df['ISO'] = df['ISO'].astype(str)
df['Annee'] = df['Annee'].astype(int)
if 'cluster' in df.columns: df['cluster'] = df['cluster'].astype(int)

targets = [c for c in df.columns if c.startswith('target_')]
print(f'Dataset: {df.shape}, {df["ISO"].nunique()} pays, {len(targets)} cibles')"""),

md("## 2. Couverture des 62 cibles"),
code("""target_coverage = pd.Series({t: df[t].notna().sum() for t in targets}).sort_values()
fig, ax = plt.subplots(figsize=(11, max(8, len(targets)*0.18)))
colors = ['#d62728' if v < 1000 else ('#ff7f0e' if v < 3000 else '#2ca02c') for v in target_coverage.values]
target_coverage.plot.barh(color=colors, ax=ax)
ax.set_title(f"Couverture des {len(targets)} cibles V12 — observations non-nulles", weight='bold', fontsize=13)
ax.set_xlabel('Nombre de lignes (sur 8 400)')
ax.axvline(1000, color='red', alpha=0.3, ls='--', label='<1000 : à éviter')
ax.axvline(3000, color='orange', alpha=0.3, ls='--', label='<3000 : limité')
ax.legend()
plt.tight_layout()
plt.show()"""),

md("""## 3. Cultures spécifiques — Distributions des rendements

**Justification** : on remplace les agrégats faibles (R²=0.09-0.14) par modèles dédiés par culture. Voici les distributions log(rendement) pour les 18 cultures les plus couvertes."""),
code("""specific_yields = [c for c in df.columns if c.startswith('yield_')
                   and not c.endswith('_kgha') and df[c].notna().sum() > 1500]
specific_yields = sorted(specific_yields, key=lambda c: -df[c].notna().sum())[:18]

fig, axes = plt.subplots(3, 6, figsize=(20, 11))
for ax, c in zip(axes.flatten(), specific_yields):
    s = df[c].dropna()
    ax.hist(np.log1p(s), bins=30, color='forestgreen', alpha=0.7, edgecolor='white')
    ax.axvline(np.log1p(s.median()), color='red', ls='--', lw=1.5, label=f'med={s.median():.0f}')
    ax.set_title(f'{c.replace("yield_","")}\\nn={len(s):,}', fontsize=10, weight='bold')
    ax.set_xlabel('log1p(kg/ha)', fontsize=8)
    ax.legend(fontsize=7)
plt.suptitle('Distributions log-rendement de 18 cultures spécifiques', weight='bold', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()"""),

md("## 4. Profil des 8 clusters climatiques"),
code("""cluster_profile = pd.read_csv('data/cleaned/country_clusters.csv')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1 : T° × P, coloré par cluster
ax = axes[0]
for c in sorted(cluster_profile['cluster'].unique()):
    sub = cluster_profile[cluster_profile['cluster']==c]
    ax.scatter(sub['temp_mean'], sub['precip_mean'], label=f'C{c} (n={len(sub)})',
               s=60, alpha=0.7, edgecolor='black', lw=0.5, color=plt.cm.tab10(c))
ax.set_xlabel('Température moyenne (°C)')
ax.set_ylabel('Précipitations annuelles (mm)')
ax.set_title('Pays dans l\\'espace climatique', weight='bold')
ax.legend(loc='best', fontsize=9)

# Plot 2 : Latitude × T°
ax = axes[1]
for c in sorted(cluster_profile['cluster'].unique()):
    sub = cluster_profile[cluster_profile['cluster']==c]
    ax.scatter(sub['latitude'], sub['temp_mean'], label=f'C{c}',
               s=60, alpha=0.7, edgecolor='black', lw=0.5, color=plt.cm.tab10(c))
ax.set_xlabel('Latitude (°)')
ax.set_ylabel('Température (°C)')
ax.set_title('Latitude vs Température par cluster', weight='bold')
ax.axvline(0, color='gray', ls=':', alpha=0.5)
ax.legend(fontsize=9)

# Plot 3 : Taille des clusters
ax = axes[2]
counts = cluster_profile['cluster'].value_counts().sort_index()
ax.bar(counts.index, counts.values, color=[plt.cm.tab10(c) for c in counts.index], alpha=0.85)
ax.set_xlabel('Cluster')
ax.set_ylabel('Nombre de pays')
ax.set_title('Taille de chaque cluster', weight='bold')
for i, v in enumerate(counts.values):
    ax.text(counts.index[i], v + 1, str(v), ha='center', fontsize=10, weight='bold')

plt.tight_layout()
plt.show()"""),

md("""## 5. 🆕 EcoCrop Suitability Scores — par culture

**Justification d'usage** : On utilise la base FAO EcoCrop (T_opt, P_opt, frost tolerance) pour calculer un score de viabilité climatique de chaque culture dans chaque pays. Score = 0 (impossible) à 1 (idéal)."""),
code("""# Heatmap suitability × cluster pour 16 cultures clés
suit_cols = [c for c in df.columns if c.startswith('suit_')]
if suit_cols:
    KEY_CROPS = ['suit_wheat','suit_rice','suit_maize','suit_soybeans','suit_potato',
                 'suit_tomato','suit_apple','suit_banana','suit_coconut','suit_olives',
                 'suit_grape','suit_dates','suit_strawberry','suit_orange','suit_mango','suit_avocado']
    KEY_CROPS = [c for c in KEY_CROPS if c in df.columns]

    # Moyenne par cluster
    pivot = df.groupby('cluster')[KEY_CROPS].mean().T
    pivot.index = [c.replace('suit_','') for c in pivot.index]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, cmap='RdYlGn', vmin=0, vmax=1, annot=True, fmt='.2f',
                cbar_kws={'label': 'Suitability score (0-1)'}, ax=ax)
    ax.set_title('Suitability EcoCrop par culture × cluster climatique', weight='bold', fontsize=13)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Culture')
    plt.tight_layout()
    plt.show()
    print('\\n→ Vert = idéal, Rouge = impossible. Note : datte uniquement viable en zone aride chaude (cluster 1).')"""),

code("""# Validation : suitability tomato vs rendement réel tomato (corrélation)
if 'suit_tomato' in df.columns and 'yield_tomato' in df.columns:
    sub = df.dropna(subset=['suit_tomato','yield_tomato'])
    r = sub[['suit_tomato','yield_tomato']].corr().iloc[0,1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sub['suit_tomato'], np.log1p(sub['yield_tomato']),
               c=sub['cluster'], cmap='tab10', alpha=0.4, s=15)
    ax.set_xlabel('Suitability EcoCrop tomate')
    ax.set_ylabel('log1p(yield_tomato_kgha)')
    ax.set_title(f'Validation suitability tomate vs rendement réel\\nr={r:+.3f}',
                  weight='bold', fontsize=12)
    plt.tight_layout()
    plt.show()"""),

md("""## 6. 🆕 Religion Pew 2010 — distribution par pays

**Justification** : religion influence régime alimentaire (porc, alcool, bovins sacrés) → utile pour prédire élevage."""),
code("""rel_cols = [c for c in df.columns if c.startswith('share_') and any(r in c for r in ['muslim','christian','hindu','buddhist','unaffiliated','folk','jew'])]
if rel_cols:
    # Distribution par cluster
    rel_by_cluster = df.groupby('cluster')[rel_cols].mean()
    rel_by_cluster.columns = [c.replace('share_','').replace('_pct','') for c in rel_by_cluster.columns]

    fig, ax = plt.subplots(figsize=(13, 7))
    rel_by_cluster.plot(kind='bar', stacked=True, ax=ax, colormap='tab10', alpha=0.85)
    ax.set_title('Composition religieuse moyenne par cluster climatique', weight='bold', fontsize=13)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('% population')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=10)
    ax.set_xticklabels([f'C{c}' for c in rel_by_cluster.index], rotation=0)
    plt.tight_layout()
    plt.show()"""),

code("""# Corrélation religion → élevage porcin (test logique)
if 'share_muslim_pct' in df.columns and 'target_pig_carcass' in df.columns:
    sub = df.dropna(subset=['share_muslim_pct','target_pig_carcass'])
    r = sub[['share_muslim_pct','target_pig_carcass']].corr().iloc[0,1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(sub['share_muslim_pct'], sub['target_pig_carcass'],
               alpha=0.4, s=15, c=sub['cluster'], cmap='tab10')
    ax.set_xlabel('% Population musulmane (Pew 2010)')
    ax.set_ylabel('target_pig_carcass (kg)')
    ax.set_title(f'Religion musulmane → Poids carcasse porc\\nr = {r:+.3f}',
                  weight='bold', fontsize=12)
    plt.tight_layout()
    plt.show()
    print(f'\\n→ Pays musulmans : moins de cochons (négatif logique).')"""),

md("""## 7. 🆕 Meat consumption per type (OWID)

**Justification** : 5 colonnes meat (beef, pig, poultry, sheep/goat, other) + lait + œufs → contexte de demande pour les modèles d'élevage."""),
code("""meat_cols = [c for c in df.columns if c.startswith('meat_') and c.endswith('_kg_pc')]
if meat_cols:
    # Comparaison consommation par cluster
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Boxplots par cluster
    ax = axes[0]
    melted = df[['cluster'] + meat_cols].melt(id_vars='cluster', var_name='meat_type', value_name='kg_pc')
    melted['meat_type'] = melted['meat_type'].str.replace('meat_','').str.replace('_kg_pc','')
    sns.boxplot(data=melted, x='meat_type', y='kg_pc', hue='cluster', ax=ax, palette='tab10')
    ax.set_title('Consommation viande (kg/hab/an) par type × cluster', weight='bold')
    ax.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.set_xlabel(''); ax.set_ylabel('kg / habitant / an')

    # Moyennes par cluster
    ax = axes[1]
    cluster_meat = df.groupby('cluster')[meat_cols].mean()
    cluster_meat.columns = [c.replace('meat_','').replace('_kg_pc','') for c in cluster_meat.columns]
    cluster_meat.plot(kind='bar', stacked=True, ax=ax, colormap='Paired', alpha=0.9)
    ax.set_title('Consommation moyenne par cluster (stacked)', weight='bold')
    ax.set_xlabel('Cluster'); ax.set_ylabel('Total kg/hab/an')
    ax.set_xticklabels([f'C{c}' for c in cluster_meat.index], rotation=0)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()"""),

md("""## 8. 🆕 Indices de stress climatique (V12)

**Justification** : Heat stress, frost risk, aridity, PET — features dérivées qui capturent les stress que les cultures subissent."""),
code("""stress_cols = ['heat_stress_index','frost_risk_index','aridity_index_calc','growing_season_index_v2','continentality']
stress_cols = [c for c in stress_cols if c in df.columns]

if stress_cols:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax, col in zip(axes.flatten(), stress_cols):
        # Moyenne par cluster
        cluster_means = df.groupby('cluster')[col].mean().sort_values()
        ax.bar(cluster_means.index, cluster_means.values,
               color=[plt.cm.tab10(c) for c in cluster_means.index], alpha=0.85)
        ax.set_title(col, weight='bold', fontsize=11)
        ax.set_xlabel('Cluster')

    # Carte mondiale des PET annuels (scatter lat/lon)
    if 'pet_annual' in df.columns:
        ax = axes.flatten()[5]
        recent = df[df['Annee']==2020].dropna(subset=['pet_annual','latitude','longitude'])
        sc = ax.scatter(recent['longitude'], recent['latitude'],
                        c=recent['pet_annual'], cmap='YlOrRd', s=30, alpha=0.8)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_title('PET annuel (2020) — Évapotranspiration', weight='bold')
        plt.colorbar(sc, ax=ax, label='PET (mm/an)')

    plt.suptitle('Indices de stress climatique V12 par cluster', weight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()"""),

md("## 9. Corrélations features → cibles (heatmap par groupe)"),
code("""# 5 cibles fortes + 5 cibles moyennes
KEY_TARGETS = ['target_co2_emissions','target_methane_emissions','target_forest_share',
               'target_thermal_anomaly','target_yield_cereals','target_yield_tomato',
               'target_milk_yield','target_cattle_carcass','target_stunting','target_birth_rate']
KEY_TARGETS = [t for t in KEY_TARGETS if t in df.columns]

GROUPS = {
    'Climat dynamique': [c for c in df.columns if any(k in c for k in ['T_annual','P_annual','T_anomaly','nasa_t2m','nasa_prec','be_t_anom_annual']) and '_lag' not in c and '_roll' not in c],
    'Suitability EcoCrop': [c for c in df.columns if c.startswith('suit_')][:12],
    'Stress climatique': [c for c in df.columns if c in ('heat_stress_index','frost_risk_index','aridity_index_calc','growing_season_index_v2','continentality','pet_annual')],
    'Religion (Pew)': [c for c in df.columns if c.startswith('share_') and ('muslim' in c or 'christian' in c or 'hindu' in c)],
    'Meat per type': [c for c in df.columns if c.startswith('meat_') and c.endswith('_kg_pc')],
    'Sol & humidité': [c for c in df.columns if any(k in c for k in ['clay','sand','soil_pH','organic_carbon','gwetroot','gwettop'])][:8],
}

fig, axes = plt.subplots(len(GROUPS), 1, figsize=(13, 3*len(GROUPS)))
for ax, (gname, gfeats) in zip(axes, GROUPS.items()):
    cols = [c for c in gfeats if c in df.columns]
    if not cols: continue
    cmat = df[cols + KEY_TARGETS].corr().loc[cols, KEY_TARGETS]
    sns.heatmap(cmat, ax=ax, cmap='RdBu_r', center=0, vmin=-0.8, vmax=0.8,
                annot=True, fmt='.2f', cbar=True, annot_kws={'size': 8})
    ax.set_title(f'{gname} → cibles', weight='bold')
    ax.set_xticklabels([t.replace('target_','') for t in KEY_TARGETS], rotation=30, ha='right', fontsize=8)
plt.tight_layout()
plt.show()"""),

md("## 10. Évolution temporelle (8 pays × 4 cibles)"),
code("""PAYS_SELECT = ['FR', 'US', 'CN', 'IN', 'BR', 'NG', 'RU', 'JP']
TIME_TARGETS = ['target_yield_tomato', 'target_thermal_anomaly',
                'target_co2_emissions', 'target_milk_yield']
TIME_TARGETS = [t for t in TIME_TARGETS if t in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, tgt in zip(axes.flatten(), TIME_TARGETS):
    for iso in PAYS_SELECT:
        sub = df[df['ISO']==iso].dropna(subset=[tgt]).sort_values('Annee')
        if len(sub) >= 5:
            cluster = sub['cluster'].iloc[0] if 'cluster' in sub.columns else 0
            ax.plot(sub['Annee'], sub[tgt], marker='o', ms=3, lw=1.5,
                    label=f'{iso} (C{cluster})', alpha=0.8)
    ax.set_title(tgt.replace('target_',''), weight='bold')
    ax.set_xlabel('Année'); ax.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.show()"""),

md("## 11. Plafond de prédictibilité — comparaison nouvelles features"),
code("""def stat_max_corr(tgt, only_groups=None):
    sub = df.dropna(subset=[tgt])
    num = sub.select_dtypes(include='number').columns.tolist()
    excl = set(targets) | {'ISO','Annee','T_ref','P_ref','cluster'}
    feats = [c for c in num if c not in excl]
    if only_groups:
        feats = [c for c in feats if any(p in c for p in only_groups)]
    if not feats: return np.nan
    cors = sub[feats].corrwith(sub[tgt]).abs().dropna()
    return cors.max() if len(cors) else np.nan

# Pour quelques cibles, calculer le max |r| pour chaque groupe de features
TEST_TARGETS = ['target_yield_tomato','target_yield_apple','target_yield_chicken_carcass'.replace('chicken_carcass','potato'),
                'target_milk_yield','target_co2_emissions','target_pig_carcass']
TEST_TARGETS = [t for t in TEST_TARGETS if t in df.columns]

groups_to_test = {
    'Climat de base': ['T_annual','P_annual','temp_mean','precip_mean'],
    'NASA POWER': ['nasa_t2m','nasa_prec','nasa_gwet'],
    'EcoCrop suit': ['suit_'],
    'Stress climat V12': ['heat_stress','frost_risk','aridity','growing_season'],
    'Religion': ['share_'],
    'Meat per type': ['meat_'],
    'Sol': ['clay','sand','soil_pH','organic_carbon'],
}

rows = []
for tgt in TEST_TARGETS:
    if tgt not in df.columns: continue
    row = {'cible': tgt.replace('target_','')}
    for gname, prefixes in groups_to_test.items():
        row[gname] = round(stat_max_corr(tgt, only_groups=prefixes), 3)
    rows.append(row)

if rows:
    summary = pd.DataFrame(rows).set_index('cible')
    print('Max |corrélation| par groupe de features (par cible) :\\n')
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(summary, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Max |r|'}, ax=ax)
    ax.set_title('Plafond de corrélation par groupe de features × cible',
                  weight='bold', fontsize=13)
    plt.tight_layout()
    plt.show()"""),

md("""## 12. Synthèse — justifications V12

```
NOUVEAUTÉS V12 INTÉGRÉES                          IMPACT
─────────────────────────────────────────────────────────────
🆕 Cultures spécifiques (36)    yield_X agg → spé      ★★★
🆕 EcoCrop suitability (39)     T_opt × P_opt × frost  ★★
🆕 Stress climat (5)            heat/frost/aridity/PET ★★
🆕 Religion Pew (7)             contexte alimentaire   ★
🆕 Meat per type (5)            beef/pig/poultry/...   ★
🆕 Élevage FAO (8)              lait/carcasses/œufs    ★★★
🆕 Émissions atmo (3)           CO2/CH4/N2O            ★★★
🆕 Énergie production (6)       solar/wind/hydro/...   ★★★
```

### Cibles débloquées par V12
- **Émissions CH4/N2O** → R²=0.96 ⭐⭐⭐
- **Production œufs** → R²=0.85 ⭐⭐
- **Rendement lait** → R²=0.83 ⭐⭐
- **Production énergie fossile** → R²=0.85-0.90 ⭐⭐
- **Cultures spécifiques top** : Concombre 0.59, Colza 0.58, Tomate 0.53, Pomme de terre 0.43

### Limites résiduelles
- Carcasses animales : R² 0.2-0.3 (race + intensification non publiques)
- Cultures niche : Mangue, Pois chiche, Cerise toujours R² < 0
- Migration nette : appartient à Couche 4

→ Voir `couche1_planete/modelisation_couche1.ipynb` pour le détail modèles""")
])


# ════════════════════════════════════════════════════════════════════════
# NOTEBOOK 3 — modelisation_complete.ipynb (V12 multi-couche)
# ════════════════════════════════════════════════════════════════════════

notebook3 = nb([
md("""# 🎯 Notebook 3 — Modélisation complète multi-couches (V12)

> Lance les entraînements des deux couches existantes et produit un rapport unifié.
>
> Pour le détail par couche, voir :
> - `couche1_planete/modelisation_couche1.ipynb` (62 cibles)
> - `couche2_sang/modelisation_couche2.ipynb` (10 cibles)

## Méthodologie
- Datasets : V12 Couche 1 (8 400 × 741) + V8 honnête Couche 2 (8 400 × 590)
- Anti-leak strict par cible
- 2 stratégies : Global / Global+Cluster
- Modèle : XGBoost 500 estim., depth=6, lr=0.05
- Split : GroupShuffleSplit par pays (test = pays jamais vus)"""),

md("## 1. Imports"),
code("""import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

# Charge les résultats des couches déjà entraînées
c1_results = 'couche1_planete/reports/results.csv'
c2_results = 'couche2_sang/reports/results.csv'

if os.path.exists(c1_results):
    res_c1 = pd.read_csv(c1_results)
    print(f'✓ Couche 1 : {len(res_c1)} cibles')
else:
    print('✗ Couche 1 — lancer : python couche1_planete/train.py')
    res_c1 = pd.DataFrame()

if os.path.exists(c2_results):
    res_c2 = pd.read_csv(c2_results)
    print(f'✓ Couche 2 : {len(res_c2)} cibles')
else:
    print('✗ Couche 2 — lancer : python couche2_sang/train.py')
    res_c2 = pd.DataFrame()"""),

md("## 2. Tableau combiné des résultats"),
code("""if not res_c1.empty:
    res_c1['Couche'] = '🌍 Planète'
if not res_c2.empty:
    res_c2['Couche'] = '🩸 Sang'

all_res = pd.concat([res_c1, res_c2], ignore_index=True)
all_res = all_res.sort_values('R² Global+Cluster', ascending=False)
print(f'Total cibles : {len(all_res)}\\n')

# Stats par catégorie
print('Distribution des R² :')
print(f'  ⭐⭐⭐ R² ≥ 0.85 : {(all_res["R² Global+Cluster"] >= 0.85).sum()}')
print(f'  ⭐⭐  R² ≥ 0.70 : {(all_res["R² Global+Cluster"] >= 0.70).sum()}')
print(f'  ⭐   R² ≥ 0.50 : {(all_res["R² Global+Cluster"] >= 0.50).sum()}')
print(f'  🟡 0.20-0.50  : {((all_res["R² Global+Cluster"] >= 0.20) & (all_res["R² Global+Cluster"] < 0.50)).sum()}')
print(f'  ❌ < 0.20     : {(all_res["R² Global+Cluster"] < 0.20).sum()}')"""),

md("## 3. Top 20 cibles toutes couches confondues"),
code("""top20 = all_res.head(20)[['Couche','Cible','Technique','R² Global+Cluster','MAE']]
print(top20.to_string(index=False))"""),

md("## 4. Visualisation comparative par couche"),
code("""fig, ax = plt.subplots(figsize=(12, max(8, len(all_res)*0.18)))
sorted_res = all_res.sort_values('R² Global+Cluster')
colors_map = {'🌍 Planète': 'steelblue', '🩸 Sang': 'crimson'}
colors = [colors_map.get(c, 'gray') for c in sorted_res['Couche']]
ax.barh(sorted_res['Cible'], sorted_res['R² Global+Cluster'], color=colors, alpha=0.85)
ax.axvline(0.5, color='gray', ls='--', alpha=0.5)
ax.axvline(0.7, color='green', ls='--', alpha=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlim(-0.5, 1.0)
ax.set_xlabel('R² (pays jamais vus)')
ax.set_title(f'Performance des {len(all_res)} cibles toutes couches', weight='bold', fontsize=14)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=v, label=k) for k,v in colors_map.items()], loc='lower right')
plt.tight_layout()
plt.show()"""),

md("## 5. Distribution R² par sous-couche"),
code("""sublayer_stats = all_res.groupby(['Couche','Sous-couche'])['R² Global+Cluster'].agg(
    ['mean','median','count','min','max']).round(3)
print(sublayer_stats.to_string())"""),

md("""## 6. Bilan & prochaines étapes

```
📊 BILAN V12

Couche 1 — La Planète (62 cibles)
  ⭐⭐⭐ Émissions CH4 / N2O      ≥ 0.95
  ⭐⭐  Énergie (charbon/pétrole)  ≥ 0.85
  ⭐⭐  Production œufs            0.85
  ⭐⭐  % forêt                    0.88
  ⭐⭐  Rendement lait             0.83
  ⭐   Cultures spé top           0.43-0.59 (concombre, tomate, colza)
  ❌   Cultures niche             Mangue, Pois chiche, Cerise

Couche 2 — Le Sang (10 cibles)  [strict env→démo]
  ⭐⭐⭐ Natalité, Fécondité       ≥ 0.95
  ⭐⭐  Mortalité infantile        0.82
  ⭐⭐  Espérance de vie           0.75
  ⭐   Croissance démo            0.69
  ❌   Migration nette            0.08 (→ Couche 4)
```

### À venir
- **Couche 3** — Le Moteur (économie, logistique, commerce)
- **Couche 4** — Le Chaos (politique, gouvernance, conflits)""")
])


# ════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════

for name, nb_data in [
    ("data_processing.ipynb", notebook1),
    ("data_visualization.ipynb", notebook2),
    ("modelisation_complete.ipynb", notebook3),
]:
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, ensure_ascii=False, indent=1)
    print(f"[OK] {name}  ({len(nb_data['cells'])} cellules)")

print("\n[DONE] 3 notebooks V12 générés.")
