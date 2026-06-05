"""
Script de nettoyage des datasets agricoles avec filtrage des unités.
Corrigé pour gérer les encodings différents (latin-1 vs utf-8).
"""
import pandas as pd
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RAW_DIR = 'data/raw'
OUT_DIR = 'data/processed'
os.makedirs(OUT_DIR, exist_ok=True)


def load_csv(filepath):
    """Charge un CSV avec le bon encoding."""
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(filepath, encoding=enc, low_memory=False)
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue


def find_col(df, *names):
    """Trouve une colonne par nom, même avec des encodings différents."""
    for name in names:
        if name in df.columns:
            return name
    # Essayer une recherche partielle (sans accents)
    for name in names:
        base = name.lower().replace('é', '').replace('è', '').replace('ê', '').replace('à', '')
        for col in df.columns:
            col_clean = col.lower().replace('é', '').replace('è', '').replace('ê', '').replace('à', '')
            # Also handle garbled chars
            if base[:4] in col_clean or base[:4] in col.lower():
                return col
    return None


def clean_common(df, value_col='Valeur'):
    """Nettoyage commun."""
    df = df.dropna(subset=[value_col])
    df = df.drop_duplicates()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    if 'Annee' in df.columns:
        df['Annee'] = df['Annee'].astype(int)
    return df


def save(df, name):
    filepath = f'{OUT_DIR}/{name}'
    df.to_csv(filepath, index=False, encoding='utf-8')
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f'  ✅ {name} — {df.shape[0]} lignes, {df.shape[1]} col ({size_mb:.1f} Mo)')
    print(f'     Colonnes: {list(df.columns)}')


# ============================================================
print("=" * 60)
print("NETTOYAGE AVEC FILTRAGE DES UNITÉS")
print("=" * 60)

# ============================================================
# 1. BILAN NUTRITIF SOLS → kg/ha uniquement
# ============================================================
print('\n1. Bilan Nutritif Sols...')
df = load_csv(f'{RAW_DIR}/Bilan_Nutritif_Sols.csv')
print(f'   Colonnes brutes : {list(df.columns)}')

# Trouver les colonnes malgré l'encoding
col_unite = find_col(df, 'Unité', 'Unit\xe9', 'Unite')
col_element = find_col(df, 'Élément', '\xc9l\xe9ment', 'Element')
col_zone = find_col(df, 'Zone')
col_annee = find_col(df, 'Année', 'Ann\xe9e', 'Annee')
col_symbole = find_col(df, 'Symbole')
col_valeur = find_col(df, 'Valeur')
col_produit = find_col(df, 'Produit')

print(f'   Colonne unité trouvée : "{col_unite}" -> valeurs: {list(df[col_unite].unique())}')

# Filtrer kg/ha
df = df[df[col_unite] == 'kg/ha'].copy()
print(f'   Après filtre kg/ha : {len(df)} lignes')

# Supprimer colonnes inutiles - on garde SEULEMENT Pays, Produit, Annee, Valeur
cols_keep = {col_zone: 'Pays', col_produit: 'Produit', col_annee: 'Annee', col_valeur: 'Valeur'}
df = df[[c for c in cols_keep.keys()]].copy()
df = df.rename(columns=cols_keep)

df = clean_common(df)
save(df, 'bilan_nutritif_sols.csv')

# ============================================================
# 2. BILAN NUTRITIF TERRES CULTIVÉES → kg/ha et %
# ============================================================
print('\n2. Bilan Nutritif Terres Cultivées...')
df = load_csv(f'{RAW_DIR}/Bilan_Nutritif_Terres_Cultivés.csv')
print(f'   Colonnes brutes : {list(df.columns)}')

col_unite = find_col(df, 'Unité', 'Unit\xe9', 'Unite')
col_element = find_col(df, 'Élément', '\xc9l\xe9ment', 'Element')
col_zone = find_col(df, 'Zone')
col_annee = find_col(df, 'Année', 'Ann\xe9e', 'Annee')
col_valeur = find_col(df, 'Valeur')
col_produit = find_col(df, 'Produit')

print(f'   Colonne unité : "{col_unite}" -> valeurs: {list(df[col_unite].unique())}')

# Filtrer kg/ha et %
df = df[df[col_unite].isin(['kg/ha', '%'])].copy()
print(f'   Après filtre kg/ha + % : {len(df)} lignes')

cols_keep = {col_zone: 'Pays', col_produit: 'Produit', col_element: 'Element',
             col_annee: 'Annee', col_unite: 'Unite', col_valeur: 'Valeur'}
df = df[[c for c in cols_keep.keys()]].copy()
df = df.rename(columns=cols_keep)

df = clean_common(df)
save(df, 'bilan_nutritif_terres_cultivees.csv')

# ============================================================
# 3. VARIATION TEMPÉRATURE → °C (déjà ok)
# ============================================================
print('\n3. Variation Température...')
df = load_csv(f'{RAW_DIR}/Env_Variation_Temp.csv')

col_zone = find_col(df, 'Zone')
col_annee = find_col(df, 'Année', 'Ann\xe9e', 'Annee')
col_valeur = find_col(df, 'Valeur')
col_element = find_col(df, 'Élément', '\xc9l\xe9ment', 'Element')
col_mois = find_col(df, 'Mois')

cols_keep = {col_zone: 'Pays', col_mois: 'Mois', col_element: 'Element',
             col_annee: 'Annee', col_valeur: 'Valeur'}
df = df[[c for c in cols_keep.keys()]].copy()
df = df.rename(columns=cols_keep)

df = clean_common(df)
save(df, 'variation_temperature.csv')

# ============================================================
# 4. INTRANTS / UTILISATION TERRES → 1000 ha et %
# ============================================================
print('\n4. Intrants / Utilisation Terres...')
df = load_csv(f'{RAW_DIR}/Intrants_Terres_Utilisation.csv')

col_unite = find_col(df, 'Unité', 'Unit\xe9', 'Unite')
col_element = find_col(df, 'Élément', '\xc9l\xe9ment', 'Element')
col_zone = find_col(df, 'Zone')
col_annee = find_col(df, 'Année', 'Ann\xe9e', 'Annee')
col_valeur = find_col(df, 'Valeur')
col_produit = find_col(df, 'Produit')

print(f'   Colonne unité : "{col_unite}" -> valeurs: {list(df[col_unite].unique())}')

df = df[df[col_unite].isin(['1000 ha', '%'])].copy()
print(f'   Après filtre : {len(df)} lignes')

cols_keep = {col_zone: 'Pays', col_produit: 'Produit', col_element: 'Element',
             col_annee: 'Annee', col_unite: 'Unite', col_valeur: 'Valeur'}
df = df[[c for c in cols_keep.keys()]].copy()
df = df.rename(columns=cols_keep)

df = clean_common(df)
save(df, 'intrants_utilisation_terres.csv')

# ============================================================
# 5. PRODUCTION CULTURES
# ============================================================
print('\n5. Production Cultures (⏳ fichier volumineux)...')
df = load_csv(f'{RAW_DIR}/ProdCulture_ProduitsAnimaux.csv')

col_unite = find_col(df, 'Unité', 'Unit\xe9', 'Unite')
col_element = find_col(df, 'Élément', '\xc9l\xe9ment', 'Element')
col_zone = find_col(df, 'Zone')
col_annee = find_col(df, 'Année', 'Ann\xe9e', 'Annee')
col_valeur = find_col(df, 'Valeur')
col_produit = find_col(df, 'Produit')

print(f'   Elements: {list(df[col_element].unique())}')

# Cultures végétales : Production (tonnes), Rendement (kg/ha), Superficie (ha)
mask_cultures = (
    df[col_element].isin(['Production', 'Rendement', 'Superficie récoltée']) &
    df[col_unite].isin(['tonnes', 'kg/ha', 'ha'])
)
df_cultures = df[mask_cultures].copy()
df_animaux = df[~mask_cultures].copy()

# Garder les bonnes colonnes
cols_keep = {col_zone: 'Pays', col_produit: 'Produit', col_element: 'Element',
             col_annee: 'Annee', col_unite: 'Unite', col_valeur: 'Valeur'}

df_cultures = df_cultures[[c for c in cols_keep.keys()]].copy()
df_cultures = df_cultures.rename(columns=cols_keep)
df_cultures = clean_common(df_cultures)
print(f'   Cultures : {len(df_cultures)} lignes, {df_cultures["Produit"].nunique()} produits')
save(df_cultures, 'production_cultures.csv')

df_animaux = df_animaux[[c for c in cols_keep.keys()]].copy()
df_animaux = df_animaux.rename(columns=cols_keep)
df_animaux = clean_common(df_animaux)
save(df_animaux, 'production_animaux.csv')

# ============================================================
# 6. FERTILIZERS → kg/ha uniquement
# ============================================================
print('\n6. Fertilizers by Nutrient...')
df = pd.read_csv(f'{RAW_DIR}/Inputs_FertilizersNutrient.csv', encoding='utf-8', low_memory=False)
print(f'   Elements: {list(df["Element"].unique())}')

# FILTRE : Use per area of cropland = kg/ha
df = df[df['Element'] == 'Use per area of cropland'].copy()
print(f'   Après filtre : {len(df)} lignes')

# Garder seulement Pays, Produit, Annee, Valeur
df = df[['Area', 'Item', 'Year', 'Value']].copy()
df = df.rename(columns={'Area': 'Pays', 'Item': 'Produit', 'Year': 'Annee', 'Value': 'Valeur'})

df = clean_common(df)
save(df, 'fertilizers_nutrient.csv')

# ============================================================
# 7. MEAN TEMPERATURE
# ============================================================
print('\n7. Mean Temperature...')
df = pd.read_csv(f'{RAW_DIR}/Mean_Temperature.csv', encoding='utf-8')

df = df[['REF_AREA', 'REF_AREA_LABEL', 'TIME_PERIOD', 'OBS_VALUE']].copy()
df = df.rename(columns={'REF_AREA': 'Code_Pays', 'REF_AREA_LABEL': 'Pays',
                         'TIME_PERIOD': 'Annee', 'OBS_VALUE': 'Valeur'})
df = clean_common(df)
save(df, 'mean_temperature.csv')

# ============================================================
# 8. PRECIPITATIONS
# ============================================================
print('\n8. Précipitations...')
df = pd.read_csv(f'{RAW_DIR}/Precipitations_mm.csv', encoding='utf-8')

df = df[['REF_AREA', 'REF_AREA_LABEL', 'TIME_PERIOD', 'OBS_VALUE']].copy()
df = df.rename(columns={'REF_AREA': 'Code_Pays', 'REF_AREA_LABEL': 'Pays',
                         'TIME_PERIOD': 'Annee', 'OBS_VALUE': 'Valeur'})
df = clean_common(df)
save(df, 'precipitations.csv')

# ============================================================
# 9. PESTICIDES → kg/ha uniquement
# ============================================================
print('\n9. Pesticides...')
df = pd.read_csv(f'{RAW_DIR}/Pesticides_Inputs.csv', encoding='utf-8', low_memory=False)
print(f'   Elements: {list(df["Element"].unique())}')

df = df[df['Element'] == 'Use per area of cropland'].copy()
print(f'   Après filtre : {len(df)} lignes')

df = df[['Area', 'Item', 'Year', 'Value']].copy()
df = df.rename(columns={'Area': 'Pays', 'Item': 'Produit', 'Year': 'Annee', 'Value': 'Valeur'})

df = clean_common(df)
save(df, 'pesticides.csv')

# ============================================================
# VÉRIFICATION FINALE
# ============================================================
print("\n" + "=" * 60)
print("VÉRIFICATION FINALE")
print("=" * 60)

for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.csv'):
        df = pd.read_csv(f'{OUT_DIR}/{f}', nrows=5)
        print(f'\n  📄 {f}')
        print(f'     Colonnes : {list(df.columns)}')
        print(f'     Première ligne : {dict(df.iloc[0])}')

print("\n🎉 Terminé !")
