"""
analyze_cibles_features.py — Pour chaque cible de la cascade V3 :
  - R² test
  - Top 10 features par importance XGBoost
  - Provenance dataset de chaque feature (mapping prefix → source)
"""
import os, sys, io
import joblib
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Mapping préfixe → source dataset
PREFIX_TO_SOURCE = {
    # Climat
    "bio": "WorldClim BIO 1-19",
    "temp_": "WorldClim temperatures",
    "precip_": "WorldClim precipitations",
    "nasa_t2m": "NASA POWER monthly→annual",
    "nasa_prec": "NASA POWER monthly→annual",
    "nasa_rh2m": "NASA POWER humidité",
    "nasa_ps": "NASA POWER pression",
    "nasa_ws10m": "NASA POWER vent",
    "nasa_allsky": "NASA POWER radiation",
    "nasa_gwet": "NASA POWER soil moisture",
    "be_t_anom": "Berkeley Earth (1743-2020)",
    "be_t_baseline": "Berkeley Earth",
    "enso_": "NOAA ENSO indices",
    "nao": "NOAA NAO index",
    "amo": "NOAA AMO index",
    "pdo": "NOAA PDO index",
    "soi": "NOAA SOI",
    "ao_": "NOAA AO",
    "co2_ppm_global": "NOAA Mauna Loa CO2",

    # Géo statique
    "latitude": "Country centroids",
    "longitude": "Country centroids",
    "elevation": "USGS/ETOPO elevation",
    "slope": "ETOPO derived",
    "roughness": "ETOPO derived",
    "dist_to_": "Geographic distances",
    "tide_amplitude": "NOAA tide",
    "area_km2": "Country basics",
    "cluster": "KMeans climat-éco (8 clusters)",

    # Sol
    "clay_pct": "SoilGrids/FAO",
    "silt_pct": "SoilGrids/FAO",
    "sand_pct": "SoilGrids/FAO",
    "soil_pH": "SoilGrids",
    "organic_carbon": "SoilGrids",

    # Stress climatique dérivé
    "heat_stress": "Calculé (T > 25)",
    "frost_risk": "Calculé (T_min < 0)",
    "aridity": "Calculé Thornthwaite",
    "continentality": "Calculé T_max - T_min",
    "growing_season": "Calculé",
    "pet_annual": "Calculé Thornthwaite PET",
    "feature_npp": "Calculé NPP physique",
    "feature_fauna": "Calculé fauna density",
    "feature_photoperiod": "Calculé photoperiod",

    # Socio
    "Population": "World Bank",
    "GDP_pc": "World Bank",
    "GDP_total": "World Bank",
    "Urban_pct": "World Bank",
    "Electricity_pct": "World Bank",

    # Émissions
    "edgar_fossil_co2_": "🆕 EDGAR 2025 JRC (par secteur)",
    "edgar_ch4_": "🆕 EDGAR 2025 JRC CH4",
    "edgar_n2o_": "🆕 EDGAR 2025 JRC N2O",
    "edgar_f-gases_": "🆕 EDGAR 2025 JRC F-gases",
    "cmm_ch4_kt": "🆕 Coal Mine Methane",
    "cmm_satellite_": "🆕 CMM satellite monitoring",

    # Hydrologie
    "aqueduct_": "🆕 WRI Aqueduct 4.0 (40 indicateurs)",
    "aquastat_": "🆕 FAO AQUASTAT",
    "aqua_bulk_": "🆕 FAO AQUASTAT bulk",
    "sea_level_": "🆕 PSMSL Sea Level",

    # Sol/Écologie
    "epi_": "🆕 EPI 2024 Yale (31 indicateurs)",
    "iucn_": "🆕 IUCN Red List (4780 espèces)",
    "tectonic_plates_count": "🆕 Tectonic Plates geojson",
    "volcanoes_count": "🆕 Smithsonian volcanoes",
    "earthquake": "Calculé séismes",

    # Agriculture
    "Engrais": "FAO Engrais",
    "Pesticides": "FAO Pesticides",
    "Terres_agricoles": "FAO Land",
    "Terres_arables": "FAO Land",
    "Part_irriguee": "FAO Irrigation",
    "Part_bio": "FAO Organic",
    "Irrigation": "FAO",
    "suit_": "FAO EcoCrop suitability scores",
    "spam_yield_": "🆕 SPAM 2020 V2 (46 cultures rendement)",
    "spam_harvest_": "🆕 SPAM 2020 V2 (surface récoltée)",
    "fertilizer_N_": "🆕 FAO N/P/K séparés",
    "fertilizer_P_": "🆕 FAO N/P/K séparés",
    "fertilizer_K_": "🆕 FAO N/P/K séparés",
    "pest_": "🆕 FAO Pesticides catégorisés (insecticides/herbicides/...)",
    "landuse_": "🆕 FAO Land Use Inputs",
    "machinery_": "🆕 FAO Machinery (tracteurs, moissonneuses)",

    # Religion (Pew)
    "share_muslim": "Pew Research 2010",
    "share_christian": "Pew Research 2010",
    "share_hindu": "Pew Research 2010",
    "share_buddhist": "Pew Research 2010",
    "share_folk": "Pew Research 2010",
    "share_unaffiliated": "Pew Research 2010",

    # Élevage
    "meat_": "OWID meat consumption per type",
    "milk_consumption": "OWID",
    "egg_consumption": "OWID",
    "wahis_": "🆕 WAHIS Animal Diseases (OIE)",
    "glw4_": "🆕 GLW4 Gridded Livestock (TIF raster)",

    # Pêche
    "fish_isscaap_": "🆕 FAO Global Production Fish (ISSCAAP)",

    # Énergie
    "ember_": "🆕 Ember Climate (16 sources)",
    "powerplant_": "🆕 WRI Global Power Plant Database (17 fuels)",
    "irena_target_": "🆕 IRENA renewable targets",
    "reshare_": "🆕 IRENA % renewable",
    "heatgen_": "🆕 IRENA heat generation",
    "wb_renewable_": "World Bank",
    "wb_elec_access_pct": "World Bank SE4ALL",
    "coal_rents_pct": "World Bank",
    "oil_rents_pct": "World Bank",
    "natgas_rents": "World Bank",

    # Géologie
    "mrds_": "🆕 USGS MRDS Minerals (304k sites)",
    "oil_gas_": "🆕 Global Oil & Gas Extraction Tracker",
    "coal_n_mines": "🆕 Global Coal Mine Tracker",
    "coal_capacity": "🆕 Global Coal Mine Tracker",
    "iron_n_mines": "🆕 Global Iron Ore Mines Tracker",
    "eq_": "🆕 USGS Earthquakes (geocoded)",
    "earthquake_count": "Calculé séismes",

    # Économie
    "agri_value_pct": "World Bank/FAO",
    "EcoClass": "FAO classification",

    # Cascade
    "oof_": "Prédiction OOF cascade",
}

def get_source(feature_name):
    """Mappe un nom de feature à sa source dataset."""
    for prefix, source in PREFIX_TO_SOURCE.items():
        if feature_name == prefix or feature_name.startswith(prefix):
            return source
    return "Autre / dérivé"


# Charger résultats V3
results = pd.read_csv("couche1_planete/reports/cascade_v3_results.csv")
print(f"📊 ANALYSE COMPLÈTE — {len(results)} CIBLES CASCADE V3\n")
print("=" * 100)

models_dir = "couche1_planete/models_cascade_v3"

for _, row in results.sort_values("R² cascade V3", ascending=False).iterrows():
    label = row["Cible"]
    tgt = row["Technique"]
    sublayer = row["Sous-couche"]
    r2 = row["R² cascade V3"]
    mae = row["MAE"]
    n_obs = int(row["N obs"])
    n_feats = int(row["N features"])
    n_cascade = int(row["Cascade OOF"])

    # Statut
    if r2 >= 0.85: status = "⭐⭐⭐"
    elif r2 >= 0.70: status = "⭐⭐"
    elif r2 >= 0.50: status = "⭐"
    elif r2 >= 0.20: status = "🟡"
    else: status = "❌"

    print(f"\n┌─{'─'*98}─┐")
    print(f"│ {status}  {label:30s} ({tgt})")
    print(f"│ R² = {r2:+.4f}  |  MAE = {mae:.3f}  |  N obs = {n_obs:,}  |  Sous-couche : {sublayer}")
    print(f"│ Features utilisées : {n_feats} (dont {n_cascade} OOF cascade)")

    # Charger modèle et top features
    model_path = f"{models_dir}/best_{tgt}.joblib"
    if os.path.exists(model_path):
        data = joblib.load(model_path)
        pipe = data["pipe"]
        features = data["features"]
        model = pipe.named_steps.get("model")
        if model is not None and hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            top10 = imp.head(10)
            print(f"│ TOP 10 FEATURES :")
            for i, (feat, val) in enumerate(top10.items(), 1):
                source = get_source(feat)
                print(f"│   {i:2d}. {feat:45s} imp={val:.3f}  ← {source}")
    print(f"└─{'─'*98}─┘")


# Récap sources
print("\n\n📚 RÉCAP — DATASETS UTILISÉS")
print("=" * 80)
all_sources = {}
for _, row in results.iterrows():
    tgt = row["Technique"]
    model_path = f"{models_dir}/best_{tgt}.joblib"
    if not os.path.exists(model_path): continue
    data = joblib.load(model_path)
    features = data["features"]
    model = data["pipe"].named_steps.get("model")
    if model is None or not hasattr(model, "feature_importances_"): continue
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    top10 = imp.head(10)
    for feat in top10.index:
        source = get_source(feat)
        all_sources[source] = all_sources.get(source, 0) + 1

print("\nFréquence d'apparition des sources dans le top-10 par cible :\n")
for source, count in sorted(all_sources.items(), key=lambda x: -x[1])[:30]:
    bar = "█" * min(count, 50)
    print(f"  {count:4d}  {source:50s} {bar}")
