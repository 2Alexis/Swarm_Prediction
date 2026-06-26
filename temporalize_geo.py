"""
temporalize_geo.py — Casse la staticité des datasets géologiques en les agrégeant par (ISO, Année).

  - Séismes USGS (time → année + nearest country via KDTree)
  - Power plants (génération annuelle 2013-2019 par pays)
  - Volcans : recency de la dernière éruption (statique mais utile)
"""
import os, sys, io
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
OUT = "data/cleaned"

# ── Country centroids pour assignment ───────────────────────────────────────
centroids = pd.read_csv(f"{OUT}/country_centroids.csv", keep_default_na=False)
centroids = centroids.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
tw = pd.DataFrame([{"longitude": 120.96, "latitude": 23.70, "COUNTRY": "Taiwan",
                    "ISO": "TW", "COUNTRYAFF": "Taiwan", "AFF_ISO": "TW"}])
centroids = pd.concat([centroids, tw], ignore_index=True)
cen_tree = cKDTree(centroids[["latitude", "longitude"]].values)
cen_iso = centroids["ISO"].values

def nearest_iso(lats, lons):
    pts = np.column_stack([lats, lons])
    dists, idx = cen_tree.query(pts)
    return cen_iso[idx]

# ── 1. SÉISMES TEMPORELS ──────────────────────────────────────────────────
print("[1] Séismes (USGS) → (ISO, Année)…")
eq = pd.read_csv(f"{OUT}/earthquakes_cleaned.csv", low_memory=False)
eq = eq.dropna(subset=["time", "latitude", "longitude", "mag"])
eq["Annee"] = pd.to_datetime(eq["time"], errors="coerce", utc=True).dt.year
eq = eq.dropna(subset=["Annee"])
eq["Annee"] = eq["Annee"].astype(int)
eq = eq[eq["mag"] >= 4.0]  # filtre bruit (M<4 = micro-séismes)
eq["ISO"] = nearest_iso(eq["latitude"].values, eq["longitude"].values)
eq_agg = eq.groupby(["ISO", "Annee"]).agg(
    earthquake_count=("mag", "count"),
    earthquake_max_mag=("mag", "max"),
    earthquake_mean_mag=("mag", "mean"),
    earthquake_mean_depth=("depth", "mean"),
).reset_index()
eq_agg.to_csv(f"{OUT}/earthquakes_yearly.csv", index=False)
print(f"   {len(eq_agg)} (ISO, Année), {eq_agg['ISO'].nunique()} pays, {eq_agg['Annee'].min()}-{eq_agg['Annee'].max()}")

# ── 2. POWER PLANT GENERATION (2013-2019) → annual aggregates per ISO ─────
print("[2] Génération énergétique par pays/année…")
pp = pd.read_csv("data/raw/geologie/global_power_plants.csv", low_memory=False)
pp = pp.dropna(subset=["latitude", "longitude"])
pp["ISO"] = nearest_iso(pp["latitude"].values, pp["longitude"].values)
records = []
for year in range(2013, 2020):
    col = f"generation_gwh_{year}"
    if col not in pp.columns:
        continue
    sub = pp[["ISO", "primary_fuel", col]].dropna(subset=[col])
    sub = sub.rename(columns={col: "gen_gwh"})
    sub["Annee"] = year
    records.append(sub)
gen = pd.concat(records, ignore_index=True)
# Total generation per ISO/Year
gen_total = gen.groupby(["ISO", "Annee"], as_index=False)["gen_gwh"].sum().rename(columns={"gen_gwh": "elec_generation_gwh"})
# Share renewable
ren_fuels = ["Hydro", "Solar", "Wind", "Geothermal", "Biomass", "Wave and Tidal"]
gen["is_renew"] = gen["primary_fuel"].isin(ren_fuels).astype(int)
gen_ren = gen.groupby(["ISO", "Annee"]).apply(
    lambda x: (x["gen_gwh"] * x["is_renew"]).sum() / max(x["gen_gwh"].sum(), 1e-9)
).reset_index().rename(columns={0: "elec_renew_share"})
elec = gen_total.merge(gen_ren, on=["ISO", "Annee"], how="left")
elec.to_csv(f"{OUT}/power_generation_yearly.csv", index=False)
print(f"   {len(elec)} (ISO, Année)")

# ── 3. VOLCANS : récency dernière éruption (peut servir de feature statique enrichie) ──
print("[3] Volcans : récency éruption…")
volc = pd.read_csv("data/raw/geologie/volcanoes.csv")
volc = volc.dropna(subset=["latitude", "longitude"])
volc["ISO"] = nearest_iso(volc["latitude"].values, volc["longitude"].values)
# convert last_eruption_year (peut être "Unknown")
volc["last_year"] = pd.to_numeric(volc["last_eruption_year"], errors="coerce")
volc_agg = volc.groupby("ISO").agg(
    volcanoes_count=("volcano_number", "count"),
    last_eruption_year=("last_year", "max"),
    population_within_100km_max=("population_within_100_km", "max"),
).reset_index()
volc_agg.to_csv(f"{OUT}/volcanoes_country.csv", index=False)
print(f"   {len(volc_agg)} pays")

# ── 4. SEA LEVEL ALT FETCH (NOAA) ─────────────────────────────────────────
print("[4] Niveau des mers global (NOAA tide gauge)…")
try:
    import urllib.request
    url = "https://www.ncei.noaa.gov/access/monitoring/products/sea-level/sea-level-trends.csv"
    # Cette URL peut échouer ; on essaie une alternative directe
    alt_url = "https://climate.nasa.gov/system/internal_resources/details/original/121_Global_Sea_Level_Data_File.txt"
    req = urllib.request.Request(alt_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        txt = r.read().decode("utf-8", errors="ignore")
    rows = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0].replace(".", "").isdigit():
            try:
                yr = int(float(parts[2]))
                val = float(parts[5]) if len(parts) > 5 else None
                if val is not None:
                    rows.append({"Annee": yr, "Valeur": val})
            except Exception:
                continue
    if rows:
        df = pd.DataFrame(rows).groupby("Annee", as_index=False).mean()
        df.to_csv(f"{OUT}/global_sea_level.csv", index=False)
        print(f"   {len(df)} années")
    else:
        print("   ✗ aucune donnée parsée")
except Exception as e:
    print(f"   ✗ {e}")

print("\n[OK] Datasets géo temporalisés.")
