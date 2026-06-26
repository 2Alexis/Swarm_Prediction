"""
build_dataset_v2.py
====================
Refonte du pipeline Couche 1 — corrige les 5 défauts majeurs du V1 :

  1. Multi-échantillonnage des features physiques (≠ centroïde unique)
  2. Variables climatiques DYNAMIQUES + anomalies + lags + rolling
  3. Cibles dérivées (NPP / fauna / VSI / flood / erosion) reclassées en `feature_*`
  4. Agrégation du rendement par GRANDE FAMILLE → 1 ligne par (ISO, Année)
  5. Intégration de TOUS les datasets cleaned ignorés (UCDP, démo, socio, énergie)

Sortie : data/cleaned/dataset_final_v2.csv  (clé = ISO × Annee)
"""
import os
import math
import numpy as np
import pandas as pd
import pycountry
import babel
from scipy.spatial import cKDTree
from PIL import Image
import xarray as xr

from build_dataset import (
    custom_mappings, custom_english_mappings, agricultural_centroids,
    get_english_iso, get_iso_from_alpha3, haversine, clean_columns,
)

DATA_DIR = "data/cleaned"
RAW_DIR = "data/raw"
WC_DIR = os.path.join(RAW_DIR, "WorldClim")
RELIEF_DIR = os.path.join(RAW_DIR, "topographie_relief")
OUTPUT_FILE = os.path.join(DATA_DIR, "dataset_final_v2.csv")


# ── 1. AGRÉGATIONS DE PRODUITS (FAO) ─────────────────────────────────────────
# FAO publie déjà des agrégats. On les utilise comme cibles propres.
CROP_FAMILIES = {
    "cereals":      "Céréales, primaires",
    "oilcrops":     "Cultures Oleágineuses, équivalent huile",
    "pulses":       "Légumineuses Sèches, tot.",
    "roots":        "Racines&Tubercules, total",
    "fruits":       "Fruits Prim et Melons",
    "vegetables":   "Légumes Prim Exc Melons",
    "fibres":       "Cultur Textiles, équivalent fibre",
    "citrus":       "Agrumes, total",
    "treenuts":     "Fruits à Coque, total",
}


# ── 2. UTILS WORLDCLIM (échantillonnage multi-pixels) ────────────────────────
def open_wc_tif(folder, pattern, month=None):
    folder_path = os.path.join(WC_DIR, folder)
    if not os.path.exists(folder_path):
        return None
    month_str = f"{month:02d}" if month else None
    for f in os.listdir(folder_path):
        if f.endswith(".tif") and pattern in f and (month_str is None or month_str in f):
            return os.path.join(folder_path, f)
    return None


def read_wc_grid(tif_path, lat_min, lat_max, lon_min, lon_max):
    """Lit une bbox d'un raster WorldClim, retourne tableau 2D des valeurs valides."""
    if tif_path is None or not os.path.exists(tif_path):
        return None
    with Image.open(tif_path) as img:
        w, h = img.size
        col_min = max(0, int((lon_min + 180.0) / 360.0 * w))
        col_max = min(w, int((lon_max + 180.0) / 360.0 * w) + 1)
        row_min = max(0, int((90.0 - lat_max) / 180.0 * h))
        row_max = min(h, int((90.0 - lat_min) / 180.0 * h) + 1)
        if col_max <= col_min or row_max <= row_min:
            return None
        crop = img.crop((col_min, row_min, col_max, row_max))
        arr = np.array(crop, dtype=np.float32)
    arr = np.where((arr < -10000) | (arr > 1e10) | np.isnan(arr), np.nan, arr)
    if np.isnan(arr).all():
        return None
    return arr


# ── 3. ETOPO chargé une fois ────────────────────────────────────────────────
_etopo = None
def get_etopo():
    global _etopo
    if _etopo is None:
        path = os.path.join(RELIEF_DIR, "earth-topography-10arcmin.nc")
        _etopo = xr.open_dataset(path)
    return _etopo


def country_bbox_from_area(lat, lon, area_km2):
    """Demi-côté de la bbox déduit de la superficie (carré équivalent)."""
    side_km = math.sqrt(max(area_km2, 1.0))
    half_deg_lat = (side_km / 2.0) / 111.0
    cos_lat = max(0.2, math.cos(math.radians(lat)))
    half_deg_lon = (side_km / 2.0) / (111.0 * cos_lat)
    half_deg_lat = min(half_deg_lat, 25.0)  # cap pour Russie/Canada
    half_deg_lon = min(half_deg_lon, 35.0)
    return (lat - half_deg_lat, lat + half_deg_lat,
            lon - half_deg_lon, lon + half_deg_lon)


def land_mean(arr, land_mask=None):
    if arr is None:
        return None
    if land_mask is not None and land_mask.shape == arr.shape:
        vals = arr[(~np.isnan(arr)) & land_mask]
    else:
        vals = arr[~np.isnan(arr)]
    if len(vals) == 0:
        return None
    return float(np.nanmean(vals))


def sample_country_climate(lat, lon, area_km2):
    """Échantillonne climat & élévation sur la bbox du pays, moyenne sur les pixels TERRESTRES."""
    lat_min, lat_max, lon_min, lon_max = country_bbox_from_area(lat, lon, area_km2)

    ds = get_etopo()
    lat_arr = ds["latitude"].values
    lon_arr = ds["longitude"].values
    li_min = max(0, np.searchsorted(lat_arr, lat_min) - 1)
    li_max = min(len(lat_arr), np.searchsorted(lat_arr, lat_max) + 1)
    lo_min = max(0, np.searchsorted(lon_arr, lon_min) - 1)
    lo_max = min(len(lon_arr), np.searchsorted(lon_arr, lon_max) + 1)
    elev_grid = ds["topography"].isel(
        latitude=slice(li_min, li_max), longitude=slice(lo_min, lo_max)
    ).values.astype(np.float32)
    if elev_grid.size == 0:
        elev_grid = np.array([[0.0]], dtype=np.float32)
    land_etopo = elev_grid > 0

    elevation_mean = float(np.nanmean(elev_grid[land_etopo])) if land_etopo.any() else float(np.nanmean(elev_grid))
    elevation_std = float(np.nanstd(elev_grid[land_etopo])) if land_etopo.any() else 0.0
    elevation_max = float(np.nanmax(elev_grid))
    roughness = elevation_max - float(np.nanmin(elev_grid))

    # WorldClim — bbox + moyenne
    def mean_var(folder, pattern, month=None):
        path = open_wc_tif(folder, pattern, month)
        grid = read_wc_grid(path, lat_min, lat_max, lon_min, lon_max)
        return land_mean(grid, None)

    out = {
        "elevation": elevation_mean,
        "elevation_std": elevation_std,
        "roughness_m": roughness,
        "temp_mean":           mean_var("biovar", "bio_1.tif")  or mean_var("biovar", "bio1.tif"),
        "temp_max":            mean_var("biovar", "bio_5.tif"),
        "temp_min":            mean_var("biovar", "bio_6.tif"),
        "precip_mean":         mean_var("biovar", "bio_12.tif"),
        "precip_seasonality":  mean_var("biovar", "bio_15.tif"),
    }

    # Variables mensuelles agrégées
    wind = [mean_var("wind", "wind", month=m) for m in range(1, 13)]
    srad = [mean_var("solrad", "srad", month=m) for m in range(1, 13)]
    vapr = [mean_var("vapr", "vapr", month=m) for m in range(1, 13)]
    out["wind_speed_mean"]        = float(np.nanmean([w for w in wind if w is not None])) if any(w is not None for w in wind) else None
    out["solar_radiation_mean"]   = float(np.nanmean([s for s in srad if s is not None])) if any(s is not None for s in srad) else None
    out["vapor_pressure_mean"]    = float(np.nanmean([v for v in vapr if v is not None])) if any(v is not None for v in vapr) else None

    return out


# ── 4. CARGE GEOJSON HYDROLOGIE/COTE → distances depuis centroïde ──────────
import json
def min_dist_geojson(lat, lon, geojson):
    best = float("inf")
    def rec(c):
        nonlocal best
        if not c: return
        if isinstance(c[0], (int, float)):
            d = haversine(lat, lon, c[1], c[0])
            if d < best: best = d
        else:
            for it in c: rec(it)
    if "features" in geojson:
        for ft in geojson["features"]:
            rec(ft.get("geometry", {}).get("coordinates", []))
    return None if best == float("inf") else best


# ── 5. CHARGEMENT D'UN CSV (Pays, Annee, Valeur) → (ISO, Annee, NomCol) ───
def load_wb(name, colname, lang="en"):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    if "Pays" in df.columns:
        if lang == "en":
            df["ISO"] = df["Pays"].apply(get_english_iso)
        else:
            df["ISO"] = df["Pays"].str.strip().str.lower().map(custom_mappings)
    elif "Code_Pays" in df.columns:
        df["ISO"] = df["Code_Pays"].apply(get_iso_from_alpha3)
    df = df.dropna(subset=["ISO", "Annee"])
    df["Annee"] = df["Annee"].astype(int)
    df["Valeur"] = pd.to_numeric(df["Valeur"], errors="coerce")
    df = df.dropna(subset=["Valeur"])
    return df.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": colname})


# ── 6. CALCUL DES LAGS / ROLLING / ANOMALIES ────────────────────────────────
def add_temporal_features(df, group_col, cols, lags=(1, 3, 5), roll=5):
    df = df.sort_values([group_col, "Annee"]).copy()
    for c in cols:
        g = df.groupby(group_col)[c]
        for k in lags:
            df[f"{c}_lag{k}"] = g.shift(k)
        df[f"{c}_roll{roll}"]   = g.transform(lambda s: s.rolling(roll, min_periods=2).mean())
        df[f"{c}_roll{roll}_std"] = g.transform(lambda s: s.rolling(roll, min_periods=2).std())
    return df


# ── 7. SCRIPT PRINCIPAL ────────────────────────────────────────────────────
def build():
    print("[V2 1/8] Chargement des séries temporelles (FAO + WB + OWID)…")

    # Yield FAO — on garde Production + Superficie + Rendement
    df_prod = clean_columns(pd.read_csv(os.path.join(DATA_DIR, "production_cultures.csv")))
    df_prod["Pays_Clean"] = df_prod["Pays"].str.strip().str.lower()
    df_prod["ISO"] = df_prod["Pays_Clean"].map(custom_mappings)
    df_prod = df_prod.dropna(subset=["ISO"])
    df_prod["Valeur"] = pd.to_numeric(df_prod["Valeur"], errors="coerce")

    # Agrégats par grande famille (FAO publie déjà ces totaux)
    yield_rows = []
    for fam, prod in CROP_FAMILIES.items():
        sub = df_prod[(df_prod["Produit"] == prod) & (df_prod["Element"] == "Rendement")].copy()
        sub = sub[(sub["Valeur"] > 0) & (sub["Valeur"] <= 100000)]
        sub = sub.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean()
        sub = sub.rename(columns={"Valeur": f"yield_{fam}_kgha"})
        yield_rows.append(sub)

    # Production en tonnes (pour pondération + signal absolu)
    cereals_prod = df_prod[(df_prod["Produit"] == CROP_FAMILIES["cereals"]) & (df_prod["Element"] == "Production")].copy()
    cereals_prod = cereals_prod.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": "cereals_prod_t"})

    cereals_area = df_prod[(df_prod["Produit"] == CROP_FAMILIES["cereals"]) & (df_prod["Element"] == "Superficie récoltée")].copy()
    cereals_area = cereals_area.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": "cereals_area_ha"})

    # Climat annuel FAO (déjà dynamique pays/année)
    df_temp = pd.read_csv(os.path.join(DATA_DIR, "mean_temperature.csv"))
    df_temp["ISO"] = df_temp["Code_Pays"].apply(get_iso_from_alpha3)
    df_temp = df_temp.dropna(subset=["ISO"])
    df_temp = df_temp.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": "T_annual_C"})

    df_precip = pd.read_csv(os.path.join(DATA_DIR, "precipitations.csv"))
    df_precip["ISO"] = df_precip["Code_Pays"].apply(get_iso_from_alpha3)
    df_precip = df_precip.dropna(subset=["ISO"])
    df_precip = df_precip.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": "P_annual_mm"})

    # FAO intrants
    def load_fao(name, col, element=None, agg="sum"):
        df = clean_columns(pd.read_csv(os.path.join(DATA_DIR, name)))
        df["Pays_Clean"] = df["Pays"].str.strip().str.lower()
        df["ISO"] = df["Pays_Clean"].map(custom_mappings)
        df = df.dropna(subset=["ISO"])
        if element and "Element" in df.columns:
            df = df[df["Element"].str.contains(element, case=False, na=False)]
        df["Valeur"] = pd.to_numeric(df["Valeur"], errors="coerce")
        return df.groupby(["ISO", "Annee"], as_index=False)["Valeur"].agg(agg).rename(columns={"Valeur": col})

    df_fert = load_fao("fertilizers_nutrient.csv", "Engrais_kgha", agg="sum")
    df_pest = load_fao("pesticides.csv", "Pesticides_kgha", agg="sum")

    df_sols = clean_columns(pd.read_csv(os.path.join(DATA_DIR, "bilan_nutritif_sols.csv")))
    df_sols["Pays_Clean"] = df_sols["Pays"].str.strip().str.lower()
    df_sols["ISO"] = df_sols["Pays_Clean"].map(custom_mappings)
    df_sols = df_sols.dropna(subset=["ISO"])
    df_sols = df_sols[df_sols["Produit"].str.lower().str.contains("bilan nutritif des sols", na=False)]
    df_sols["Valeur"] = pd.to_numeric(df_sols["Valeur"], errors="coerce")
    df_sols = df_sols.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": "Bilan_sols_kgha"})

    # Terres → ratios pertinents
    df_terres = clean_columns(pd.read_csv(os.path.join(DATA_DIR, "intrants_utilisation_terres.csv")))
    df_terres["Pays_Clean"] = df_terres["Pays"].str.strip().str.lower()
    df_terres["ISO"] = df_terres["Pays_Clean"].map(custom_mappings)
    df_terres = df_terres.dropna(subset=["ISO"])
    df_terres["Valeur"] = pd.to_numeric(df_terres["Valeur"], errors="coerce")
    df_terres_p = df_terres.pivot_table(index=["ISO", "Annee"], columns="Produit",
                                        values="Valeur", aggfunc="mean").reset_index()
    rename_terres = {
        "Superficie du pays": "Superficie_pays_ha",
        "Superficie des terres": "Superficie_terres_ha",
        "Terres agricoles": "Terres_agricoles_ha",
        "Terres en culture": "Terres_culture_ha",
        "Terres arables": "Terres_arables_ha",
        "Superficie équipée de systèmes d'irrigation": "Irrigation_equipee_ha",
        "Superficie des terres réellement irriguée": "Irrigation_reelle_ha",
        "Terr. agricoles sous agriculture biologique": "Bio_ha",
    }
    df_terres_p = df_terres_p.rename(columns={k: v for k, v in rename_terres.items() if k in df_terres_p.columns})
    keep_terres = ["ISO", "Annee"] + [v for v in rename_terres.values() if v in df_terres_p.columns]
    df_terres_p = df_terres_p[keep_terres]
    df_terres_p["Part_terres_agricoles"] = df_terres_p["Terres_agricoles_ha"] / df_terres_p["Superficie_terres_ha"]
    df_terres_p["Part_terres_arables"]   = df_terres_p["Terres_arables_ha"]   / df_terres_p["Terres_agricoles_ha"]
    df_terres_p["Part_irriguee"]         = df_terres_p["Irrigation_equipee_ha"] / df_terres_p["Terres_agricoles_ha"]
    df_terres_p["Part_bio"]              = df_terres_p["Bio_ha"]               / df_terres_p["Terres_agricoles_ha"]

    print("[V2 2/8] Construction du squelette (ISO, Annee) — 1960→2024…")
    isos = pd.Series(sorted(set(
        list(df_temp["ISO"].unique()) + list(df_precip["ISO"].unique()) +
        list(df_terres_p["ISO"].unique()) + list(cereals_prod["ISO"].unique())
    )))
    years = list(range(1961, 2025))
    df = pd.MultiIndex.from_product([isos, years], names=["ISO", "Annee"]).to_frame(index=False)

    # Merge yields
    for sub in yield_rows:
        df = df.merge(sub, on=["ISO", "Annee"], how="left")
    df = df.merge(cereals_prod, on=["ISO", "Annee"], how="left")
    df = df.merge(cereals_area, on=["ISO", "Annee"], how="left")

    # Climat + intrants + sols + terres
    df = df.merge(df_temp,    on=["ISO", "Annee"], how="left")
    df = df.merge(df_precip,  on=["ISO", "Annee"], how="left")
    df = df.merge(df_fert,    on=["ISO", "Annee"], how="left")
    df = df.merge(df_pest,    on=["ISO", "Annee"], how="left")
    df = df.merge(df_sols,    on=["ISO", "Annee"], how="left")
    df = df.merge(df_terres_p, on=["ISO", "Annee"], how="left")

    print("[V2 3/8] Intégration des datasets ignorés (WB/OWID/UCDP)…")
    wb_datasets = [
        ("wb_gdp_per_capita.csv",        "GDP_pc"),
        ("wb_gdp_current_usd.csv",       "GDP_total_usd"),
        ("wb_life_expectancy.csv",       "Life_Exp"),
        ("wb_child_mortality.csv",       "Child_Mort"),
        ("wb_birth_rate.csv",            "Birth_Rate"),
        ("wb_death_rate.csv",            "Death_Rate"),
        ("wb_population_total.csv",      "Population"),
        ("wb_population_growth.csv",     "Pop_Growth"),
        ("wb_net_migration.csv",         "Net_Migration"),
        ("wb_urban_population_pct.csv",  "Urban_pct"),
        ("wb_agricultural_land_pct.csv", "Agri_Land_pct"),
        ("wb_inflation_cpi.csv",         "Inflation_CPI"),
        ("wb_unemployment_rate.csv",     "Unemployment"),
        ("wb_gini_index.csv",            "Gini"),
        ("wb_poverty_rate_190.csv",      "Poverty_190"),
        ("wb_public_debt_gdp.csv",       "Debt_GDP"),
        ("wb_trade_gdp.csv",             "Trade_GDP"),
        ("wb_renewable_energy_pct.csv",  "Renew_Energy_pct"),
        ("wb_electricity_access_pct.csv","Electricity_pct"),
        ("wb_energy_use_per_capita.csv", "Energy_pc"),
        ("wb_internet_users_pct.csv",    "Internet_pct"),
        ("wb_mobile_subscriptions.csv",  "Mobile_subs"),
        ("wb_health_expenditure_gdp.csv","Health_GDP"),
        ("wb_hospital_beds_per_1000.csv","Hospital_Beds"),
        ("wb_rd_expenditure_gdp.csv",    "RD_GDP"),
        ("wb_malaria_incidence.csv",     "Malaria"),
        ("wb_hiv_prevalence.csv",        "HIV"),
        ("wb_deaths_communicable_disease.csv", "Deaths_Communicable"),
        ("wb_freshwater_withdrawal_pct.csv",   "Water_Withdrawal_pct"),
    ]
    for fname, colname in wb_datasets:
        sub = load_wb(fname, colname, lang="en")
        if sub is not None:
            df = df.merge(sub, on=["ISO", "Annee"], how="left")

    for fname, colname in [
        ("owid_hdi.csv", "HDI"),
        ("owid_co2_emissions.csv", "CO2_emissions"),
        ("owid_energy.csv", "Energy_total"),
        ("owid_poverty.csv", "Poverty_OWID"),
        ("global_hunger_index.csv", "Hunger_Index"),
    ]:
        sub = load_wb(fname, colname, lang="en")
        if sub is not None:
            df = df.merge(sub, on=["ISO", "Annee"], how="left")

    # UCDP — comptage de conflits par pays/année (le `Pays` peut contenir des dyades)
    ucdp_path = os.path.join(DATA_DIR, "UcdpPrioConflict_v24_1.csv")
    if os.path.exists(ucdp_path):
        ucdp = pd.read_csv(ucdp_path)
        ucdp["countries"] = ucdp["Pays"].str.split(",")
        ucdp = ucdp.explode("countries")
        ucdp["country_clean"] = ucdp["countries"].str.strip()
        ucdp["ISO"] = ucdp["country_clean"].apply(get_english_iso)
        ucdp = ucdp.dropna(subset=["ISO"])
        ucdp_agg = ucdp.groupby(["ISO", "Annee"]).agg(
            conflict_intensity_max=("Valeur", "max"),
            conflict_count=("Valeur", "count"),
        ).reset_index()
        df = df.merge(ucdp_agg, on=["ISO", "Annee"], how="left")
        df["conflict_intensity_max"] = df["conflict_intensity_max"].fillna(0)
        df["conflict_count"] = df["conflict_count"].fillna(0)
        # Conflit cumulé sur 5 ans (mémoire des guerres)
        df = df.sort_values(["ISO", "Annee"])
        df["conflict_cumul_5y"] = df.groupby("ISO")["conflict_count"].transform(
            lambda s: s.rolling(5, min_periods=1).sum())

    print("[V2 4/8] Features physiques (multi-points par pays)…")
    df_centroids = pd.read_csv(os.path.join(DATA_DIR, "country_centroids.csv"), keep_default_na=False)
    taiwan_row = pd.DataFrame([{
        "longitude": 120.96, "latitude": 23.70, "COUNTRY": "Taiwan",
        "ISO": "TW", "COUNTRYAFF": "Taiwan", "AFF_ISO": "TW"
    }])
    df_centroids = pd.concat([df_centroids, taiwan_row], ignore_index=True)
    iso_to_coords = {r["ISO"]: (r["latitude"], r["longitude"], r["COUNTRY"])
                     for _, r in df_centroids.iterrows()}

    # Pour la bbox on a besoin d'une superficie : on prend la dernière connue par pays
    last_area = df.dropna(subset=["Superficie_terres_ha"]).groupby("ISO")["Superficie_terres_ha"].last() / 100.0  # ha → km²
    last_area = last_area.to_dict()

    # GeoJSONs hydro/côte chargés une fois
    def load_geojson(p):
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    rivers = load_geojson(os.path.join(RAW_DIR, "hydrologie_eau", "rivers.geojson"))
    lakes = load_geojson(os.path.join(RAW_DIR, "hydrologie_eau", "lakes.geojson"))
    coast = load_geojson(os.path.join(RELIEF_DIR, "coastlines.geojson"))

    unique_isos = df["ISO"].unique()
    iso_features = {}
    for i, iso in enumerate(unique_isos, 1):
        if iso in agricultural_centroids:
            lat, lon = agricultural_centroids[iso]
        elif iso in iso_to_coords:
            lat, lon, _ = iso_to_coords[iso]
        else:
            continue
        area = last_area.get(iso, 100_000)  # 100k km² par défaut
        if not area or pd.isna(area) or area <= 0:
            area = 100_000

        phys = sample_country_climate(lat, lon, area)
        phys["latitude"] = lat
        phys["longitude"] = lon
        phys["area_km2"] = area
        # Distances (depuis centroïde — peu coûteux)
        if rivers: phys["dist_to_river_km"] = min_dist_geojson(lat, lon, rivers)
        if lakes:  phys["dist_to_lake_km"]  = min_dist_geojson(lat, lon, lakes)
        if coast:  phys["dist_to_coast_km"] = min_dist_geojson(lat, lon, coast)
        if phys.get("dist_to_river_km") is not None and phys.get("dist_to_lake_km") is not None:
            phys["dist_to_freshwater_km"] = min(phys["dist_to_river_km"], phys["dist_to_lake_km"])

        iso_features[iso] = phys
        if i % 20 == 0 or i == len(unique_isos):
            print(f"   {i}/{len(unique_isos)} pays traités")

    phys_df = pd.DataFrame.from_dict(iso_features, orient="index").reset_index().rename(columns={"index": "ISO"})
    df = df.merge(phys_df, on="ISO", how="left")

    print("[V2 5/8] Anomalies climatiques + lags + rolling…")
    # Référence climato par pays = moyenne 1961-1990 (référence WMO)
    ref = df[(df["Annee"] >= 1961) & (df["Annee"] <= 1990)].groupby("ISO").agg(
        T_ref=("T_annual_C", "mean"), P_ref=("P_annual_mm", "mean")).reset_index()
    df = df.merge(ref, on="ISO", how="left")
    df["T_anomaly"] = df["T_annual_C"] - df["T_ref"]
    df["P_anomaly_pct"] = (df["P_annual_mm"] - df["P_ref"]) / df["P_ref"] * 100.0

    # Lags + rolling sur variables-clés
    df = add_temporal_features(df,
        group_col="ISO",
        cols=["T_annual_C", "P_annual_mm", "T_anomaly", "P_anomaly_pct",
              "yield_cereals_kgha", "Engrais_kgha", "Pesticides_kgha", "Bilan_sols_kgha",
              "Population", "GDP_pc"],
        lags=(1, 3, 5), roll=5)

    print("[V2 6/8] Features dérivées (renommées feature_*)…")
    # Miami NPP — calculé avec climat ANNUEL (donc dynamique !)
    T = df["T_annual_C"].fillna(df["temp_mean"])
    P = df["P_annual_mm"].fillna(df["precip_mean"])
    npp_t = 3000.0 / (1.0 + np.exp(1.315 - 0.119 * T))
    npp_p = 3000.0 * (1.0 - np.exp(-0.000664 * P))
    df["feature_npp"] = np.minimum(npp_t, npp_p)
    df["feature_fauna_density"] = 0.05 * df["feature_npp"]
    # VSI
    s_t = np.exp(-((T - 26.0) ** 2) / (2 * (5.0 ** 2)))
    s_p = 1.0 - np.exp(-P / 800.0)
    df["feature_vsi"] = 10.0 * s_t * s_p
    df["feature_flood_risk"] = ((df["elevation"] <= 50) &
                                (df["dist_to_freshwater_km"].fillna(9999) <= 5.0)).astype(int)
    # Photopériode
    phi = np.radians(df["latitude"].fillna(0))
    def photo(doy):
        delta = 0.409 * np.sin(2 * np.pi * (doy - 80) / 365)
        val = -np.tan(phi) * np.tan(delta)
        val = np.clip(val, -1.0, 1.0)
        return (24.0 / np.pi) * np.arccos(val)
    df["feature_photoperiod_summer"] = photo(172)
    df["feature_photoperiod_winter"] = photo(355)
    df["feature_photoperiod_range"] = (df["feature_photoperiod_summer"] - df["feature_photoperiod_winter"]).abs()

    print("[V2 7/8] Cibles ML (mesurées, non-circulaires)…")
    df["target_yield_cereals"]      = np.log1p(df["yield_cereals_kgha"])
    df["target_yield_oilcrops"]     = np.log1p(df["yield_oilcrops_kgha"])
    df["target_yield_pulses"]       = np.log1p(df["yield_pulses_kgha"])
    df["target_yield_roots"]        = np.log1p(df["yield_roots_kgha"])
    df["target_yield_fruits"]       = np.log1p(df["yield_fruits_kgha"])
    df["target_yield_vegetables"]   = np.log1p(df["yield_vegetables_kgha"])

    # Stress hydrique — winsorize à 200% (au-delà → erreur de source)
    ws = df["Water_Withdrawal_pct"].clip(upper=200.0)
    df["target_water_stress"] = np.log1p(ws)

    # Soil degradation
    df["target_soil_degradation"] = df["Bilan_sols_kgha"]

    # Anomalie thermique → maintenant DYNAMIQUE (T annuelle - T ref pays)
    df["target_thermal_anomaly"] = df["T_anomaly"]

    # Cibles démo (vraies cibles socio)
    df["target_child_mortality"]   = df["Child_Mort"]
    df["target_life_expectancy"]   = df["Life_Exp"]
    df["target_pop_growth"]        = df["Pop_Growth"]
    df["target_hunger"]            = df["Hunger_Index"] if "Hunger_Index" in df.columns else np.nan

    print("[V2 8/8] Filtre et export…")
    # On garde 1990+ (les lags 5 ans sont remplis dès 1995)
    df = df[df["Annee"] >= 1990].copy()
    # On exige au moins ISO + une cible
    df = df.dropna(subset=["ISO"])

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] {OUTPUT_FILE}")
    print(f"     shape = {df.shape}")
    print(f"     pays  = {df['ISO'].nunique()}")
    print(f"     années = {df['Annee'].min()}–{df['Annee'].max()}")
    print(f"     non-null par cible:")
    for c in df.columns:
        if c.startswith("target_"):
            print(f"       {c:35s} {df[c].notna().sum():,}")


if __name__ == "__main__":
    build()
