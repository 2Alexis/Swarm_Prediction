"""
============================================================================
TÉLÉCHARGEMENT DES VRAIES DONNÉES — COUCHE 2 (Démographie & Épidémiologie)
============================================================================

Télécharge des données 100 % RÉELLES (aucune donnée générée) depuis l'API
World Bank Open Data, pour chaque mécanique de la Couche 2 :

  1. Natalité & Fécondité ........ SP.DYN.CBRT.IN, SP.DYN.TFRT.IN
  2. Mortalité par âge ........... SP.DYN.IMRT.IN, SH.DYN.MORT, SP.DYN.CDRT.IN
  3. Densité de population ....... EN.POP.DNST, SP.POP.TOTL
  4. Malnutrition / Carence ...... SN.ITK.DEFC.ZS  (FAO, relayé par World Bank)
  6. Pression migratoire ......... SM.POP.NETM

Les agrégats régionaux (World, Africa…) sont filtrés → on ne garde que les vrais
pays. Sortie : un CSV (Code_Pays, Pays, Annee, Valeur) par indicateur, sur la
clé (Pays, Année) — compatible avec la Couche 1 FAO.

NB : la mécanique 5 (épidémiologie SIR / Project Tycho) et les sources fichier
(UN WPP, UNHCR) ne sont pas dans l'API World Bank → voir SOURCES_COUCHE2.md.

Usage :  python download_couche2_data.py
"""

import csv
import io
import json
import os
import sys
import time
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Racine du projet = dossier parent de couche2/ → on écrit dans <racine>/data/couche2.
_RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_RACINE, "data", "couche2")
API = "https://api.worldbank.org/v2"

# Indicateur World Bank -> (nom de fichier, mécanique)
INDICATEURS = {
    "SP.DYN.CBRT.IN": ("natalite_brute", "1. Natalité"),
    "SP.DYN.TFRT.IN": ("fecondite_totale", "1. Fécondité"),
    "SP.DYN.IMRT.IN": ("mortalite_infantile", "2. Mortalité infantile"),
    "SH.DYN.MORT":    ("mortalite_moins5ans", "2. Mortalité <5 ans"),
    "SP.DYN.CDRT.IN": ("mortalite_brute", "2. Mortalité adulte/brute"),
    "EN.POP.DNST":    ("densite_population", "3. Densité (hab/km²)"),
    "SP.POP.TOTL":    ("population_totale", "3. Population totale"),
    "SN.ITK.DEFC.ZS": ("malnutrition_carence", "4. Carence (sous-alimentation)"),
    "SM.POP.NETM":    ("migration_nette", "6. Migration nette"),
}


def _get_json(url, essais=3):
    """GET JSON avec quelques tentatives (réseau)."""
    for i in range(essais):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            if i == essais - 1:
                raise
            time.sleep(2)


def charger_pays_reels():
    """Liste des VRAIS pays (exclut les agrégats régionaux). -> dict iso3 -> infos."""
    url = f"{API}/country?format=json&per_page=400"
    _, data = _get_json(url)
    pays = {}
    for c in data:
        if c["region"]["value"] == "Aggregates":
            continue                       # World, Europe & Central Asia, etc.
        pays[c["id"]] = {
            "Pays": c["name"],
            "Region": c["region"]["value"],
            "Revenu": c["incomeLevel"]["value"],
            "Capitale": c.get("capitalCity", ""),
            "Longitude": c.get("longitude", ""),
            "Latitude": c.get("latitude", ""),
        }
    return pays


def telecharger_indicateur(code, pays_reels):
    """Télécharge un indicateur (toutes pages) et filtre sur les vrais pays."""
    page, lignes = 1, []
    while True:
        url = f"{API}/country/all/indicator/{code}?format=json&per_page=20000&page={page}"
        meta, data = _get_json(url)
        if not data:
            break
        for d in data:
            iso3 = d.get("countryiso3code") or ""
            if iso3 not in pays_reels:
                continue                   # agrégat -> ignoré
            if d["value"] is None:
                continue                   # pas de mesure cette année-là
            lignes.append((iso3, d["country"]["value"], int(d["date"]), d["value"]))
        if page >= meta["pages"]:
            break
        page += 1
    return lignes


def sauver(lignes, nom):
    chemin = os.path.join(OUT_DIR, f"{nom}.csv")
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Code_Pays", "Pays", "Annee", "Valeur"])
        w.writerows(sorted(lignes, key=lambda x: (x[1], x[2])))
    return chemin


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 64)
    print("TÉLÉCHARGEMENT DES VRAIES DONNÉES — COUCHE 2 (World Bank)")
    print("=" * 64)

    print("\n• Liste des pays réels (hors agrégats)...")
    pays_reels = charger_pays_reels()
    print(f"  → {len(pays_reels)} pays réels identifiés.")

    # Référence pays (avec coordonnées → utile pour la géographie / le voisinage).
    ref_path = os.path.join(OUT_DIR, "pays_reference.csv")
    with open(ref_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Code_Pays", "Pays", "Region", "Revenu", "Capitale", "Longitude", "Latitude"])
        for iso3, info in sorted(pays_reels.items()):
            w.writerow([iso3, info["Pays"], info["Region"], info["Revenu"],
                        info["Capitale"], info["Longitude"], info["Latitude"]])
    print(f"  → référence enregistrée : {ref_path}")

    print("\n• Indicateurs :")
    total = 0
    for code, (nom, meca) in INDICATEURS.items():
        lignes = telecharger_indicateur(code, pays_reels)
        chemin = sauver(lignes, nom)
        total += len(lignes)
        if lignes:
            annees = [l[2] for l in lignes]
            n_pays = len({l[0] for l in lignes})
            print(f"  ✅ {meca:32s} {code:16s} {len(lignes):>7,} lignes "
                  f"| {n_pays} pays | {min(annees)}–{max(annees)}")
        else:
            print(f"  ⚠️ {meca:32s} {code:16s} AUCUNE donnée")

    print("\n" + "=" * 64)
    print(f"Terminé : {total:,} lignes réelles dans {OUT_DIR}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
