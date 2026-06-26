"""
fetch_couche1_extra.py — Téléchargement des données manquantes pour Couche 1.

Récupère :
  - Élevage (FAO production_animaux déjà téléchargé, on l'utilise)
  - Pêche (OWID + FAO)
  - Atmosphère / émissions
  - Biodiversité (EPI Yale, Living Planet via OWID)
  - Eau détaillée (OWID water quality, withdrawal sectoriel)
  - Énergie renouvelable production
  - Risques naturels par type
"""
import os, sys, io, urllib.request, urllib.parse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

D = "data/cleaned"


def fetch_text(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


def owid_csv(slug, name):
    """Télécharge un dataset OWID grapher."""
    try:
        url = f"https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=true"
        txt = fetch_text(url, timeout=120)
        df = pd.read_csv(io.StringIO(txt))
        df.columns = [c.strip() for c in df.columns]
        ent = "Entity" if "Entity" in df.columns else df.columns[0]
        yr  = "Year"   if "Year"   in df.columns else df.columns[2]
        val_cols = [c for c in df.columns if c not in (ent, "Code", yr)]
        num_cols = [c for c in val_cols if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols: return False
        out = df[[ent, yr, num_cols[0]]].rename(columns={ent: "Pays", yr: "Annee", num_cols[0]: "Valeur"})
        out = out.dropna(subset=["Pays", "Annee", "Valeur"])
        out.to_csv(f"{D}/{name}.csv", index=False)
        print(f"  + {name}: {len(out)} lignes")
        return True
    except Exception as e:
        print(f"  ✗ {slug}: {e}")
        return False


# ── 1. Élevage / production animaux (déjà téléchargé via FAO !) ────────────
print("[1] Production animaux FAO (déjà disponible)…")
if os.path.exists(f"{D}/production_animaux.csv"):
    df_anim = pd.read_csv(f"{D}/production_animaux.csv")
    print(f"  Found : {df_anim.shape}, produits = {df_anim['Produit'].nunique()}")
    print(f"  Top produits :")
    print(df_anim["Produit"].value_counts().head(10).to_string())
else:
    print("  ✗ production_animaux.csv manquant")


# ── 2. Pêche et environnement marin ────────────────────────────────────────
print("\n[2] Pêche & environnement marin…")
FISH_DATASETS = [
    ("fish-and-seafood-production",                   "owid_fish_production"),
    ("seafood-and-fish-production",                   "owid_seafood_production"),
    ("global-fishery-catch",                          "owid_fishery_catch"),
    ("aquaculture-farmed-fish-production",            "owid_aquaculture"),
    ("share-of-fish-stocks-overfished",               "owid_overfishing_share"),
    ("marine-protected-areas",                        "owid_marine_protected"),
]
for slug, name in FISH_DATASETS:
    owid_csv(slug, name)


# ── 3. Atmosphère / émissions ──────────────────────────────────────────────
print("\n[3] Atmosphère / émissions…")
ATMO_DATASETS = [
    ("ghg-emissions-per-capita-vs-gdp-per-capita",    None),  # complexe
    ("annual-co2-emissions-per-country",              "owid_co2_emissions_annual"),
    ("co-emissions-per-capita",                       "owid_co_per_capita"),
    ("methane-emissions",                             "owid_methane_total"),
    ("nitrous-oxide-emissions",                       "owid_n2o_total"),
    ("share-co2-cement",                              "owid_co2_cement"),
    ("co2-by-source",                                 "owid_co2_by_source"),
    ("share-of-co2-emissions-from-agriculture",       "owid_co2_agri_share"),
    ("greenhouse-gas-emissions-from-agriculture",     "owid_ghg_agri"),
    ("sulphur-dioxide-emissions-per-country",         "owid_so2_emissions"),
    ("nitrogen-oxide-emissions",                      "owid_nox_emissions_alt"),
    ("global-ozone-layer-thickness",                  None),  # global, pas par pays
]
for slug, name in ATMO_DATASETS:
    if name: owid_csv(slug, name)


# ── 4. Risques naturels par type (occurrence/fréquence) ────────────────────
print("\n[4] Risques naturels…")
RISK_DATASETS = [
    ("annual-rate-of-natural-disasters",              "owid_disaster_rate"),
    ("share-of-deaths-from-natural-disasters",        "owid_disaster_deaths_share"),
    ("expected-annual-losses-natural-disasters",      "owid_disaster_losses"),
    ("global-disaster-frequency",                     "owid_global_disaster_freq"),
    ("frequency-of-droughts",                         "owid_drought_freq"),
    ("frequency-of-floods",                           "owid_flood_freq"),
    ("frequency-of-storms",                           "owid_storm_freq"),
]
for slug, name in RISK_DATASETS:
    owid_csv(slug, name)


# ── 5. Biodiversité & écosystèmes ─────────────────────────────────────────
print("\n[5] Biodiversité & écosystèmes…")
BIO_DATASETS = [
    ("environmental-performance-index",               "owid_epi"),
    ("share-of-land-area-protected",                  "owid_land_protected"),
    ("number-of-threatened-species",                  "owid_threatened_species"),
    ("living-planet-index",                           "owid_living_planet"),
    ("share-of-mammals-threatened-with-extinction",   "owid_mammals_threatened"),
    ("forest-area-as-share-of-land-area",             "owid_forest_share_alt"),
    ("share-fish-stocks-overfished",                  "owid_fish_overfish"),
    ("biodiversity-habitat-index",                    "owid_bhi"),
]
for slug, name in BIO_DATASETS:
    owid_csv(slug, name)


# ── 6. Eau (détaillé) ──────────────────────────────────────────────────────
print("\n[6] Eau détaillée…")
WATER_DATASETS = [
    ("renewable-internal-freshwater-resources-per-capita", "owid_water_renewable_pc"),
    ("water-withdrawals-per-capita",                       "owid_water_withdraw_pc"),
    ("share-of-water-resources-withdrawn",                 "owid_water_share_withdraw"),
    ("annual-freshwater-withdrawals",                      "owid_water_withdraw_total"),
    ("water-stress",                                       "owid_water_stress_alt"),
    ("agricultural-water-withdrawal-as-share",             "owid_water_agri_share"),
    ("industrial-water-withdrawal-as-share",               "owid_water_indus_share"),
    ("share-of-water-withdrawal-from-agriculture",         "owid_water_agri_alt"),
    ("share-of-the-population-with-access-to-improved-drinking-water", "owid_water_access"),
]
for slug, name in WATER_DATASETS:
    owid_csv(slug, name)


# ── 7. Énergie renouvelable (production réelle) ────────────────────────────
print("\n[7] Énergie renouvelable production…")
ENERGY_DATASETS = [
    ("solar-energy-consumption",                      "owid_solar_consumption"),
    ("wind-generation",                               "owid_wind_generation"),
    ("hydropower-consumption",                        "owid_hydro_consumption"),
    ("geothermal-energy-consumption",                 "owid_geothermal_consumption"),
    ("biofuels-production",                           "owid_biofuels_production"),
    ("share-of-electricity-from-low-carbon-sources",  "owid_lowcarbon_share"),
    ("renewable-energy-supply-by-source",             "owid_renewable_supply"),
    ("primary-energy-consumption-per-capita",         "owid_primary_energy_pc"),
]
for slug, name in ENERGY_DATASETS:
    owid_csv(slug, name)


# ── 8. Sols détaillés (érosion, etc) ───────────────────────────────────────
print("\n[8] Sols détaillés…")
SOIL_DATASETS = [
    ("agricultural-land-area",                        "owid_agri_land_total"),
    ("area-of-cropland-vs-pasture",                   None),
    ("change-in-land-use",                            "owid_land_use_change"),
    ("global-soil-erosion",                           "owid_soil_erosion_global"),
    ("change-in-soil-organic-carbon",                 "owid_soc_change"),
    ("pesticide-use-tonnes",                          "owid_pesticide_tonnes"),
    ("fertilizer-use-tonnes",                         "owid_fertilizer_tonnes"),
]
for slug, name in SOIL_DATASETS:
    if name: owid_csv(slug, name)


# ── 9. Ressources naturelles ──────────────────────────────────────────────
print("\n[9] Ressources naturelles…")
RES_DATASETS = [
    ("mineral-rents",                                 "owid_mineral_rents_alt"),
    ("forest-area",                                   "owid_forest_area_total"),
    ("share-of-electricity-fossil",                   "owid_fossil_elec_share"),
    ("coal-production-by-country",                    "owid_coal_production"),
    ("oil-production-by-country",                     "owid_oil_production"),
    ("gas-production-by-country",                     "owid_gas_production"),
]
for slug, name in RES_DATASETS:
    owid_csv(slug, name)


print("\n[DONE] Téléchargements Couche 1 terminés.")
