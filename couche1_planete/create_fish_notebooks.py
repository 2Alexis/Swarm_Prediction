"""Génère les 2 notebooks FAO Fish : clean + explore."""
import json, os, re

OUT_DIR = "couche1_planete"

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
# NOTEBOOK A — clean_fish.ipynb (pipeline de nettoyage commenté)
# ════════════════════════════════════════════════════════════════════════

notebook_clean = nb([
md("""# 🐟 Pipeline de nettoyage — FAO Global Production Fish 2024.1.0

Notebook qui décompose le script `clean_fish.py` en étapes documentées.

## Dataset source
**FAO Global Production Quantity 2024.1.0** — `data/raw/GlobalProductionFish_2024.1.0/`

Contenu :
- `Global_production_quantity.csv` : 1.15M lignes (1950-2022)
- `CL_FI_COUNTRY_GROUPS.csv` : 247 pays avec UN/ISO2/ISO3 codes
- `CL_FI_SPECIES_GROUPS.csv` : 3591 espèces avec scientific name + groupes ISSCAAP
- `CL_FI_PRODUCTION_SOURCE_DET.csv` : 4 sources (Capture + 3 aquacultures)

## Pipeline en 9 étapes
1. Chargement (1.15M lignes brutes)
2. Filtrage MEASURE=Q_tlw (tonnes poids vivant)
3. Jointure pays UN → ISO2
4. Jointure espèces 3A → ISSCAAP + Scientific Name
5. Jointure source production
6. Détection outliers (winsorize optionnel)
7. Renommage final + sauvegarde
8. Agrégations multiples (par source, par ISSCAAP)
9. Stats finales

## Outputs
- `fish_production_raw_clean.csv` (190 MB, 1.1M lignes propres)
- `fish_production_by_source.csv` (capture/aquaculture par pays/année)
- `fish_production_total.csv` (totaux + diversité espèces)
- `fish_production_by_isscaap.csv` (par groupe biologique)"""),

md("## 1. Imports & chemins"),
code("""import os, sys, io
import numpy as np
import pandas as pd

RAW_DIR = '../data/raw/GlobalProductionFish_2024.1.0'  # depuis sous-dossier couche1_planete
OUT_DIR = '../data/cleaned'
os.makedirs(OUT_DIR, exist_ok=True)
print(f'Files raw : {os.listdir(RAW_DIR)[:10]}')"""),

md("## 2. Chargement données brutes"),
code("""df = pd.read_csv(f'{RAW_DIR}/Global_production_quantity.csv', low_memory=False)
print(f'{len(df):,} lignes brutes, années {df["PERIOD"].min()}–{df["PERIOD"].max()}')
print(f'Colonnes : {list(df.columns)}')
df.head()"""),

md("""## 3. Filtrage MEASURE=Q_tlw

Le dataset contient deux mesures :
- `Q_tlw` : Quantity in tonnes live weight (production en tonnes)
- `Q_no_1` : Number (individus pour certaines espèces)

On garde uniquement Q_tlw pour avoir une mesure homogène en tonnes."""),
code("""before = len(df)
df = df[df['MEASURE'] == 'Q_tlw'].copy()
df = df.drop(columns=['MEASURE'])
print(f'{len(df):,} lignes après filtre (était {before:,})')
print(f'STATUS counts: {df["STATUS"].value_counts().to_dict()}')
# A = official, I = imputed, N = national estimate, E = estimate"""),

md("## 4. Jointure pays UN_Code → ISO2 standardisé"),
code("""cn = pd.read_csv(f'{RAW_DIR}/CL_FI_COUNTRY_GROUPS.csv', low_memory=False)
cn = cn[['UN_Code', 'ISO2_Code', 'ISO3_Code', 'Name_En',
         'Continent_Group_En', 'EcoClass_Group_En']].rename(columns={
            'UN_Code': 'COUNTRY.UN_CODE',
            'ISO2_Code': 'ISO',
            'ISO3_Code': 'ISO3',
            'Name_En': 'Pays',
            'Continent_Group_En': 'Continent',
            'EcoClass_Group_En': 'EcoClass'
         })
df = df.merge(cn, on='COUNTRY.UN_CODE', how='left')
n_no_iso = df['ISO'].isna().sum()
print(f'{n_no_iso:,} lignes sans ISO ({n_no_iso/len(df)*100:.1f}%)')
print(f'→ ce sont les agrégats régionaux FAO (ex: World, Europe...) qui n\\'ont pas de code ISO')
df = df.dropna(subset=['ISO'])
print(f'\\n{len(df):,} lignes avec ISO valide, {df["ISO"].nunique()} pays uniques')"""),

md("""## 5. Jointure espèces → ISSCAAP

ISSCAAP = International Standard Statistical Classification of Aquatic Animals and Plants.
Ex de groupes : "Marine fishes nei", "Salmons, trouts, smelts", "Cyprinids", "Shrimps, prawns"..."""),
code("""sp = pd.read_csv(f'{RAW_DIR}/CL_FI_SPECIES_GROUPS.csv', low_memory=False)
sp = sp[['3A_Code', 'Name_En', 'Scientific_Name', 'Major_Group',
         'ISSCAAP_Group_En']].rename(columns={
            '3A_Code': 'SPECIES.ALPHA_3_CODE',
            'Name_En': 'Espece_En',
            'Major_Group': 'MajorGroup',
            'ISSCAAP_Group_En': 'ISSCAAP'
         })
df = df.merge(sp, on='SPECIES.ALPHA_3_CODE', how='left')
df['ISSCAAP'] = df['ISSCAAP'].fillna('Unknown')
df['MajorGroup'] = df['MajorGroup'].fillna('Unknown')

print(f'Top 10 groupes ISSCAAP :')
print(df['ISSCAAP'].value_counts().head(10).to_string())"""),

md("## 6. Jointure source production (Capture vs Aquaculture)"),
code("""ps = pd.read_csv(f'{RAW_DIR}/CL_FI_PRODUCTION_SOURCE_DET.csv', low_memory=False)
ps = ps[['Code', 'Name_En']].rename(columns={'Code': 'PRODUCTION_SOURCE_DET.CODE',
                                              'Name_En': 'Source'})
df = df.merge(ps, on='PRODUCTION_SOURCE_DET.CODE', how='left')

def simplify_source(s):
    if pd.isna(s): return 'Unknown'
    s = str(s).lower()
    if 'capture' in s: return 'Capture'
    if 'freshwater' in s: return 'Aquaculture_Freshwater'
    if 'brackish' in s: return 'Aquaculture_Brackish'
    if 'marine' in s: return 'Aquaculture_Marine'
    return 'Other'
df['SourceType'] = df['Source'].apply(simplify_source)
print(df['SourceType'].value_counts().to_string())"""),

md("""## 7. Détection outliers et NaN

Pas de NaN dans VALUE (vérifié plus haut). Pour les outliers :
- 45% des lignes sont à 0 (espèces non pêchées certaines années) → normal
- Max = 12.3M tonnes (anchois Pérou 1971) → valide historiquement
- Pas de winsorization : les valeurs extrêmes sont réelles"""),
code("""print(f'NaN dans VALUE : {df["VALUE"].isna().sum()}')
print(f'Zéros : {(df["VALUE"]==0).sum():,} ({(df["VALUE"]==0).mean()*100:.1f}%)')
print(f'Négatifs : {(df["VALUE"]<0).sum()}')
print(f'\\nQuantiles :')
print(df['VALUE'].describe(percentiles=[0.5, 0.9, 0.99, 0.999]).to_string())
print(f'\\nP99.9 = {df["VALUE"].quantile(0.999):,.0f} tonnes')
print(f'Max   = {df["VALUE"].max():,.0f} tonnes (production halieutique extrême historique)')"""),

md("## 8. Renommage final + sauvegarde"),
code("""df = df.rename(columns={
    'PERIOD': 'Annee',
    'VALUE': 'Tonnes',
    'AREA.CODE': 'Zone_FAO',
})
df = df[['ISO', 'ISO3', 'Pays', 'Continent', 'EcoClass', 'Annee',
         'SPECIES.ALPHA_3_CODE', 'Espece_En', 'Scientific_Name', 'MajorGroup', 'ISSCAAP',
         'Zone_FAO', 'SourceType', 'Source', 'Tonnes', 'STATUS']]
df.to_csv(f'{OUT_DIR}/fish_production_raw_clean.csv', index=False)
print(f'✓ {OUT_DIR}/fish_production_raw_clean.csv  ({len(df):,} lignes)')"""),

md("""## 9. Agrégations utiles

### A. Par source (Capture vs 3 aquacultures)"""),
code("""df_pos = df[df['Tonnes'] > 0]
by_src = (df_pos.groupby(['ISO', 'Annee', 'SourceType'])['Tonnes']
          .sum().unstack(fill_value=0).reset_index())
by_src.columns = ['ISO', 'Annee'] + [f'fish_{c}_t' for c in by_src.columns[2:]]
print(f'Shape : {by_src.shape}')
by_src.to_csv(f'{OUT_DIR}/fish_production_by_source.csv', index=False)
by_src.head()"""),

md("### B. Total + diversité d'espèces"),
code("""by_total = df_pos.groupby(['ISO', 'Annee']).agg(
    fish_total_t=('Tonnes', 'sum'),
    fish_species_count=('SPECIES.ALPHA_3_CODE', 'nunique'),
    fish_isscaap_count=('ISSCAAP', 'nunique'),
).reset_index()
print(f'Shape : {by_total.shape}')
by_total.to_csv(f'{OUT_DIR}/fish_production_total.csv', index=False)
by_total.head()"""),

md("### C. Top 10 groupes ISSCAAP par pays/année"),
code("""isscaap_top = (df_pos.groupby(['ISO', 'Annee', 'ISSCAAP'])['Tonnes']
                .sum().unstack(fill_value=0).reset_index())
# Garder seulement les 10 groupes ISSCAAP les plus produits mondialement
top_groups = (df_pos.groupby('ISSCAAP')['Tonnes'].sum()
              .sort_values(ascending=False).head(10).index)
keep_cols = ['ISO', 'Annee'] + [g for g in top_groups if g in isscaap_top.columns]
isscaap_top = isscaap_top[keep_cols].copy()

def safe_col(s):
    return 'fish_isscaap_' + ''.join(c if c.isalnum() else '_' for c in s.lower())[:40]
isscaap_top.columns = ['ISO', 'Annee'] + [safe_col(c) for c in isscaap_top.columns[2:]]
print(f'Shape : {isscaap_top.shape}')
isscaap_top.to_csv(f'{OUT_DIR}/fish_production_by_isscaap.csv', index=False)
isscaap_top.head()"""),

md("## 10. Stats finales"),
code("""print(f'Pays couverts : {by_total["ISO"].nunique()}')
print(f'Années : {by_total["Annee"].min()}–{by_total["Annee"].max()}')
print(f'Production totale 2022 : {by_total[by_total["Annee"]==2022]["fish_total_t"].sum()/1e6:.1f} M tonnes')

print(f'\\nTop 10 pays 2022 :')
top2022 = by_total[by_total['Annee']==2022].sort_values('fish_total_t', ascending=False).head(10)
for _, r in top2022.iterrows():
    print(f'  {r["ISO"]}: {r["fish_total_t"]/1e6:.2f} M t  ({int(r["fish_species_count"])} espèces)')

print(f'\\nOutputs créés :')
for f in ['fish_production_raw_clean.csv','fish_production_by_source.csv',
          'fish_production_total.csv','fish_production_by_isscaap.csv']:
    p = f'{OUT_DIR}/{f}'
    sz = os.path.getsize(p) / 1e6
    print(f'  {f}  ({sz:.1f} MB)')""")
])


# ════════════════════════════════════════════════════════════════════════
# NOTEBOOK B — explore_fish.ipynb (exploration + graphs + corr matrix)
# ════════════════════════════════════════════════════════════════════════

notebook_explore = nb([
md("""# 📊 Exploration FAO Global Production Fish — graphiques & corrélations

Notebook qui décompose `explore_fish.py` avec visualisations interactives.

## Sections
1. Imports & chargement
2. **Couverture** (pays × années × espèces)
3. **Distributions** production (totale + par source)
4. **Évolution mondiale 1950-2022** (capture vs aquaculture)
5. **Top 10 pays** producteurs — évolution temporelle
6. **Capture vs Aquaculture** top 15 pays
7. **Heatmap** groupes ISSCAAP × top 20 pays
8. **Top espèces mondiales** 2018-2022
9. **Corrélations** avec dataset principal V8
10. **Matrice corrélation** fish vs features clés
11. **Variation par cluster** climatique"""),

md("## 1. Imports & chargement des 3 fichiers cleaned"),
code("""import os, sys, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (12, 6)

OUT_DIR = '../data/cleaned'
df_total = pd.read_csv(f'{OUT_DIR}/fish_production_total.csv')
df_src   = pd.read_csv(f'{OUT_DIR}/fish_production_by_source.csv')
df_iss   = pd.read_csv(f'{OUT_DIR}/fish_production_by_isscaap.csv')
df_raw   = pd.read_csv(f'{OUT_DIR}/fish_production_raw_clean.csv', low_memory=False)
print(f'total: {df_total.shape}  by_source: {df_src.shape}  by_isscaap: {df_iss.shape}')
print(f'raw clean: {df_raw.shape}')"""),

md("## 2. Couverture du dataset"),
code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
years_per_iso = df_total.groupby('ISO')['Annee'].nunique().sort_values(ascending=False)
ax.hist(years_per_iso, bins=30, color='steelblue', edgecolor='white')
ax.axvline(years_per_iso.median(), color='red', ls='--', label=f'med={years_per_iso.median():.0f}')
ax.set_title("Nombre d'années couvertes par pays", weight='bold')
ax.set_xlabel('Années'); ax.legend()

ax = axes[1]
isos_per_year = df_total.groupby('Annee')['ISO'].nunique()
ax.plot(isos_per_year.index, isos_per_year.values, lw=2, color='forestgreen')
ax.fill_between(isos_per_year.index, isos_per_year.values, alpha=0.3, color='forestgreen')
ax.set_title('Nombre de pays reportant par année', weight='bold')
ax.set_xlabel('Année'); ax.set_ylabel('Nb pays')

ax = axes[2]
sp_per_iso = df_total.groupby('ISO')['fish_species_count'].mean().sort_values(ascending=False)
ax.hist(sp_per_iso, bins=30, color='coral', edgecolor='white')
ax.axvline(sp_per_iso.median(), color='blue', ls='--', label=f'med={sp_per_iso.median():.0f}')
ax.set_title("Diversité moyenne d'espèces par pays", weight='bold')
ax.set_xlabel('Nb espèces'); ax.legend()

plt.tight_layout()
plt.show()"""),

md("""## 3. Distributions — production totale (log) + répartition par source"""),
code("""fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
recent = df_total[df_total['Annee'] >= 2010]
ax.hist(np.log10(recent['fish_total_t'].clip(lower=1)), bins=40,
        color='steelblue', edgecolor='white')
ax.set_xlabel('log10(tonnes par pays/année, 2010-2022)')
ax.set_ylabel('Fréquence')
ax.set_title('Distribution production (log)', weight='bold')

ax = axes[1]
src_cols = [c for c in df_src.columns if c.startswith('fish_')]
src_2022 = df_src[df_src['Annee']==2022][src_cols].sum()
ax.pie(src_2022.values,
       labels=[c.replace('fish_','').replace('_t','') for c in src_2022.index],
       autopct='%1.1f%%', colors=plt.cm.Set3.colors)
ax.set_title('Production mondiale 2022 — répartition par source', weight='bold')

plt.tight_layout()
plt.show()"""),

md("## 4. Évolution mondiale 1950-2022 (stacked area)"),
code("""world = df_src.groupby('Annee')[src_cols].sum() / 1e6  # M tonnes

fig, ax = plt.subplots(figsize=(13, 6))
world.plot(kind='area', stacked=True, ax=ax, alpha=0.85,
            color=['#1f77b4','#ff7f0e','#2ca02c','#d62728'])
ax.set_title('Production halieutique mondiale 1950-2022 (M tonnes)',
              weight='bold', fontsize=14)
ax.set_xlabel('Année'); ax.set_ylabel('Millions de tonnes')
ax.legend([c.replace('fish_','').replace('_t','') for c in src_cols], loc='upper left')
plt.tight_layout()
plt.show()
print('→ Observation : aquaculture (vert+orange+rouge) dépasse capture (bleu) vers 2014.')"""),

md("## 5. Top 10 pays producteurs — évolution"),
code("""top10 = df_total[df_total['Annee']==2022].nlargest(10, 'fish_total_t')['ISO'].tolist()

fig, ax = plt.subplots(figsize=(13, 7))
for iso in top10:
    sub = df_total[df_total['ISO']==iso].sort_values('Annee')
    ax.plot(sub['Annee'], sub['fish_total_t']/1e6, lw=2, marker='o', ms=3,
            label=iso, alpha=0.9)
ax.set_title('Top 10 pays producteurs (2022) — évolution 1950-2022',
              weight='bold', fontsize=13)
ax.set_xlabel('Année'); ax.set_ylabel('Production (M tonnes)')
ax.legend(ncol=5, loc='upper left', fontsize=9)
ax.set_yscale('log')
plt.tight_layout()
plt.show()
print('→ La Chine domine massivement depuis ~1990 grâce à l\\'aquaculture intensive.')"""),

md("## 6. Capture vs Aquaculture — top 15 pays 2022"),
code("""top15 = df_total[df_total['Annee']==2022].nlargest(15, 'fish_total_t')['ISO'].tolist()
df15 = df_src[(df_src['ISO'].isin(top15)) & (df_src['Annee']==2022)].set_index('ISO').loc[top15]

capt = df15.get('fish_Capture_t', 0)
aqua = (df15.get('fish_Aquaculture_Freshwater_t', 0) +
        df15.get('fish_Aquaculture_Marine_t', 0) +
        df15.get('fish_Aquaculture_Brackish_t', 0))

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(top15))
ax.bar(x - 0.2, capt/1e6, width=0.4, label='Capture', color='steelblue', alpha=0.85)
ax.bar(x + 0.2, aqua/1e6, width=0.4, label='Aquaculture', color='forestgreen', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(top15)
ax.set_ylabel('Production 2022 (M tonnes)')
ax.set_title('Capture vs Aquaculture — Top 15 pays (2022)', weight='bold', fontsize=13)
ax.legend()
plt.tight_layout()
plt.show()
print('→ Asie domine aquaculture (CN, ID, VN, IN, BD).')
print('→ Russie, USA, Norvège, Chili = grande capture (poissons sauvages).')"""),

md("## 7. Heatmap groupes ISSCAAP × top 20 pays (2022, log)"),
code("""top20 = df_total[df_total['Annee']==2022].nlargest(20, 'fish_total_t')['ISO'].tolist()
iss_2022 = df_iss[(df_iss['ISO'].isin(top20)) & (df_iss['Annee']==2022)]
iss_2022 = iss_2022.set_index('ISO').loc[top20]
iss_cols = [c for c in iss_2022.columns if c.startswith('fish_isscaap_')]
mat = iss_2022[iss_cols] / 1e3  # k tonnes
mat.columns = [c.replace('fish_isscaap_','')[:25] for c in mat.columns]
log_mat = np.log10(mat.clip(lower=1))

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(log_mat, cmap='YlOrRd', annot=False,
            cbar_kws={'label': 'log10(k tonnes)'}, ax=ax)
ax.set_title('Production par groupe ISSCAAP × Top 20 pays (2022, log)',
              weight='bold', fontsize=13)
ax.set_xlabel(''); ax.set_ylabel('Pays')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()"""),

md("## 8. Top 20 espèces mondiales 2018-2022 cumulé"),
code("""recent_raw = df_raw[(df_raw['Annee'] >= 2018) & (df_raw['Tonnes'] > 0)]
top_sp = (recent_raw.groupby('Espece_En')['Tonnes'].sum()
           .sort_values(ascending=False).head(20) / 1e6)

fig, ax = plt.subplots(figsize=(12, 8))
top_sp.sort_values().plot(kind='barh', color='navy', alpha=0.85, ax=ax)
ax.set_title('Top 20 espèces mondiales 2018-2022 cumulé (M tonnes)',
              weight='bold', fontsize=13)
ax.set_xlabel('M tonnes (5 ans cumulés)')
plt.tight_layout()
plt.show()"""),

md("## 9. Corrélations avec dataset principal V8"),
code("""df_main = pd.read_csv(f'{OUT_DIR}/dataset_final_v8_honest.csv', low_memory=False)
df_main = df_main.dropna(subset=['ISO'])
merged = df_main.merge(df_total, on=['ISO','Annee'], how='left')
print(f'Merged: {len(merged):,} lignes, fish_total_t couvert : {merged["fish_total_t"].notna().sum():,}')

# Top 20 |corrélations|
merged['log_fish'] = np.log1p(merged['fish_total_t'].fillna(0))
num = merged.select_dtypes(include='number').columns.tolist()
excl = {'ISO','Annee','T_ref','P_ref','cluster','fish_total_t',
        'fish_species_count','fish_isscaap_count','log_fish'}
feats = [c for c in num if c not in excl]
cors = merged[feats].corrwith(merged['log_fish']).abs().sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(11, 8))
cors.sort_values().plot(kind='barh', color='teal', alpha=0.85, ax=ax)
ax.set_title('Top 20 |corrélations| features V8 → fish_total_t (log)',
              weight='bold', fontsize=12)
ax.set_xlabel('|corrélation|')
plt.tight_layout()
plt.show()"""),

md("""## 10. Matrice corrélation — fish vs features clés

Justifier l'usage du dataset fish : on attend des corrélations fortes avec :
- `dist_to_coast_km` (proximité côte → plus de pêche)
- `Population` (plus de gens → plus de consommation)
- `GDP_pc` (richesse → aquaculture)"""),
code("""key_feats = ['fish_total_t','fish_species_count','fish_isscaap_count']
for c in ['Population','GDP_pc','dist_to_coast_km','Urban_pct','nasa_t2m',
          'nasa_prectotcorr','forest_share_pct','Engrais_kgha']:
    if c in merged.columns:
        key_feats.append(c)
mat = merged[key_feats].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(mat, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, cbar_kws={'label': 'Pearson r'}, ax=ax)
ax.set_title('Matrice corrélation : fish × features clés', weight='bold', fontsize=12)
plt.tight_layout()
plt.show()"""),

md("## 11. Variation par cluster climatique"),
code("""if 'cluster' in merged.columns:
    recent_merged = merged[merged['Annee'] >= 2010]
    by_cluster = recent_merged.groupby('cluster').agg(
        median_total=('fish_total_t', 'median'),
        median_species=('fish_species_count', 'median'),
        countries=('ISO', 'nunique'),
    ).round(0)
    print('Stats par cluster :\\n')
    print(by_cluster.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    cluster_data = [recent_merged[recent_merged['cluster']==c]['fish_total_t'].dropna()
                     for c in sorted(recent_merged['cluster'].dropna().unique())]
    bp = ax.boxplot(cluster_data,
                     tick_labels=sorted(recent_merged['cluster'].dropna().unique().astype(int)),
                     patch_artist=True, showfliers=False)
    for patch, c in zip(bp['boxes'], sorted(recent_merged['cluster'].dropna().unique().astype(int))):
        patch.set_facecolor(plt.cm.tab10(c)); patch.set_alpha(0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Cluster'); ax.set_ylabel('Production (tonnes, log)')
    ax.set_title('Production par cluster climatique (2010-2022)', weight='bold')

    ax = axes[1]
    sp_data = [recent_merged[recent_merged['cluster']==c]['fish_species_count'].dropna()
                for c in sorted(recent_merged['cluster'].dropna().unique())]
    bp = ax.boxplot(sp_data,
                     tick_labels=sorted(recent_merged['cluster'].dropna().unique().astype(int)),
                     patch_artist=True, showfliers=False)
    for patch, c in zip(bp['boxes'], sorted(recent_merged['cluster'].dropna().unique().astype(int))):
        patch.set_facecolor(plt.cm.tab10(c)); patch.set_alpha(0.7)
    ax.set_xlabel('Cluster'); ax.set_ylabel("Diversité espèces")
    ax.set_title("Diversité d'espèces par cluster", weight='bold')

    plt.tight_layout()
    plt.show()"""),

md("""## 12. Récap & intégration future

### Nouvelles features potentielles pour la Couche 1

```
fish_total_t                       # Production totale par pays/année (M tonnes)
fish_Capture_t                     # Pêche capture sauvage
fish_Aquaculture_Freshwater_t      # Aquaculture eau douce
fish_Aquaculture_Marine_t          # Aquaculture marine
fish_Aquaculture_Brackish_t        # Aquaculture eau saumâtre
fish_species_count                 # Diversité d'espèces produites
fish_isscaap_count                 # Diversité de groupes biologiques
fish_isscaap_*                     # Top 10 groupes ISSCAAP en colonnes wide
```

### Nouvelles cibles potentielles
- `target_fish_total` = log1p(fish_total_t)
- `target_fish_capture` = log1p(fish_Capture_t)
- `target_fish_species_diversity` = fish_species_count

### Prochaine étape
Intégrer dans `enrich_couche1_v3.py` pour produire `dataset_final_v13_couche1.csv` puis re-entraîner."""),
])


# ════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════
for name, nb_data in [
    ("clean_fish.ipynb", notebook_clean),
    ("explore_fish.ipynb", notebook_explore),
]:
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, ensure_ascii=False, indent=1)
    print(f"[OK] {path}  ({len(nb_data['cells'])} cellules)")

print("\n[DONE] 2 notebooks Fish générés.")
