"""
============================================================================
COUCHE 2 — Nettoyage & Fusion des données démographiques / épidémiologiques
============================================================================

Prend les CSV bruts collectés par `download_data.py` (dans data/couche2/) et
tente de les FUSIONNER en un seul dataset propre, keyé sur (Code_Pays, Annee).

Étapes :
  1. Chargement des indicateurs (un fichier = un indicateur).
  2. Nettoyage (types, valeurs aberrantes, doublons).
  3. Fusion (outer join) sur (Code_Pays, Pays, Annee) + région.
  4. Rapport de complétude (taux de remplissage par colonne).
  5. Export → data/couche2/dataset_couche2.csv

NB : `epidemies_rougeole_tycho.csv` (Project Tycho) est par état US / semaine,
PAS par (Pays, Année) → il ne se fusionne pas ici (voir le rapport).

Usage :  python couche2/clean_merge.py
"""

import io
import os
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_RACINE, "data", "couche2")

# Fichier  ->  nom de colonne dans le dataset fusionné
INDICATEURS = {
    "natalite_brute":       "Natalite_pour1000",
    "fecondite_totale":     "Fecondite_enf_par_femme",
    "mortalite_infantile":  "MortInfantile_pour1000",
    "mortalite_moins5ans":  "MortMoins5ans_pour1000",
    "mortalite_brute":      "MortBrute_pour1000",
    "densite_population":   "Densite_hab_km2",
    "population_totale":    "Population",
    "malnutrition_carence": "Carence_pct",
    "migration_nette":      "MigrationNette",
}


def charger(nom):
    """Charge un indicateur et renomme Valeur -> nom de colonne cible."""
    chemin = os.path.join(DATA_DIR, f"{nom}.csv")
    df = pd.read_csv(chemin)
    # Nettoyage de base
    df["Annee"] = pd.to_numeric(df["Annee"], errors="coerce").astype("Int64")
    df["Valeur"] = pd.to_numeric(df["Valeur"], errors="coerce")
    df = df.dropna(subset=["Annee", "Valeur"])
    df = df.drop_duplicates(subset=["Code_Pays", "Annee"], keep="last")
    return df.rename(columns={"Valeur": INDICATEURS[nom]})[
        ["Code_Pays", "Pays", "Annee", INDICATEURS[nom]]]


def main():
    print("=" * 64)
    print("COUCHE 2 — NETTOYAGE & FUSION")
    print("=" * 64)

    # 1-2. Chargement + nettoyage
    print("\n1/4  Chargement & nettoyage des indicateurs :")
    frames = {}
    for nom in INDICATEURS:
        df = charger(nom)
        frames[nom] = df
        print(f"   • {INDICATEURS[nom]:26s} {len(df):>6,} lignes | "
              f"{df['Code_Pays'].nunique()} pays | {int(df['Annee'].min())}-{int(df['Annee'].max())}")

    # 3. Fusion progressive (outer join) sur (Code_Pays, Pays, Annee)
    print("\n2/4  Fusion (outer join) sur (Code_Pays, Pays, Annee)...")
    merged = None
    for nom, df in frames.items():
        merged = df if merged is None else merged.merge(
            df, on=["Code_Pays", "Pays", "Annee"], how="outer")
    merged = merged.sort_values(["Pays", "Annee"]).reset_index(drop=True)

    # Ajout de la région (depuis pays_reference.csv)
    ref_path = os.path.join(DATA_DIR, "pays_reference.csv")
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path)[["Code_Pays", "Region"]]
        merged = merged.merge(ref, on="Code_Pays", how="left")
        cols = ["Code_Pays", "Pays", "Region", "Annee"] + list(INDICATEURS.values())
        merged = merged[cols]

    print(f"   → dataset fusionné : {merged.shape[0]:,} lignes × {merged.shape[1]} colonnes")
    print(f"   → {merged['Code_Pays'].nunique()} pays | "
          f"{int(merged['Annee'].min())}-{int(merged['Annee'].max())}")

    # 4. Rapport de complétude
    print("\n3/4  Taux de remplissage par colonne :")
    completude = merged.notna().mean().sort_values(ascending=False)
    for col, taux in completude.items():
        barre = "█" * int(taux * 20)
        print(f"   {col:26s} {taux:6.1%} {barre}")

    # 5. Export
    print("\n4/4  Export...")
    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, "dataset_couche2.csv")
    merged.to_csv(out, index=False, encoding="utf-8")
    taille = os.path.getsize(out) / 1024
    print(f"   ✅ {out}  ({taille:.0f} Ko)")

    print("\n" + "=" * 64)
    print("VERDICT FUSION")
    print("=" * 64)
    print(f"✅ {len(INDICATEURS)} indicateurs (World Bank/FAO) fusionnés sur (Pays, Année).")
    print("⚠️  epidemies_rougeole_tycho.csv NON fusionné : granularité = état US / semaine")
    print("    (pas de clé Pays/Année) → reste un dataset séparé pour l'analyse SIR.")
    return merged


if __name__ == "__main__":
    main()
