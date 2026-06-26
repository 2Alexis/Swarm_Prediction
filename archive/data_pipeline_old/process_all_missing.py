"""
process_all_missing.py — Traite TOUS les 12 datasets manquants en une passe.

  1. WAHIS Animal Disease (Données quantitatives 2026-06-24.csv)
  2. GLW4 cattle TIF (raster → zonal stats par pays)
  3. mrds.csv (USGS Minerals — points → count par pays)
  4. global_power_plants.csv (points → capacité par pays)
  5. Global-Oil-and-Gas-Extraction-Tracker xlsx
  6. Global Coal Mine Tracker x3 xlsx
  7. Global-Iron-Ore-Mines-Tracker xlsx
  8. Production-Consumption-Met-Coal-Iron-Ore xlsx
  9. earthquakes_usgs.csv
  10. volcanoes.csv + tectonic_plates.geojson
  11. IRENA R-ELECCAP/GEN/RESHARE/HEATGEN (parser tabulaire)
  12. SPAM Harvested Area zip
  13. CMM Satellite xlsx
"""
import os, sys, io, glob, zipfile, json, urllib.request, re
import pandas as pd
import numpy as np
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")
from build_dataset import custom_mappings, get_english_iso

import pycountry
def iso3_to_iso2(c):
    try:
        co = pycountry.countries.get(alpha_3=c)
        return co.alpha_2 if co else None
    except: return None

def name_to_iso2(name):
    if pd.isna(name): return None
    s = str(name).strip().lower()
    code = custom_mappings.get(s)
    if code: return code
    return get_english_iso(name)

RAW = "data/raw"
CLN = "data/cleaned"


# ════════════════════════════════════════════════════════════════════════
# 1. WAHIS — Animal Disease Outbreaks (Données quantitatives)
# ════════════════════════════════════════════════════════════════════════
print("\n[1] 🐄 WAHIS Animal Disease (Données quantitatives 2026-06-24)…")
for src in [f"{RAW}/elevage/Données quantitatives 2026-06-24.csv",
             f"{RAW}/misc/Données quantitatives 2026-06-24.csv"]:
    if not os.path.exists(src): continue
    try:
        df = pd.read_csv(src, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(src, low_memory=False, encoding="latin-1")
    print(f"   {df.shape}, cols: {list(df.columns)[:8]}")
    df["ISO"] = df["Pays"].apply(name_to_iso2)
    df = df.dropna(subset=["ISO","Année"])
    df["Année"] = pd.to_numeric(df["Année"], errors="coerce")
    df = df.dropna(subset=["Année"])
    df["Année"] = df["Année"].astype(int)
    for c in ["Cas","Morts","Vaccinés","Mis à mort et éliminés"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    agg_cols = {"Outbreak_id":"nunique","Maladie":"nunique"}
    for c in ["Cas","Morts"]:
        if c in df.columns:
            agg_cols[c] = "sum"
    agg = df.groupby(["ISO","Année"]).agg(
        wahis_outbreaks_total=("Outbreak_id","nunique"),
        wahis_diseases_unique=("Maladie","nunique"),
        wahis_cases=("Cas","sum") if "Cas" in df.columns else ("Outbreak_id","count"),
        wahis_deaths=("Morts","sum") if "Morts" in df.columns else ("Outbreak_id","count"),
    ).reset_index().rename(columns={"Année":"Annee"})
    out = f"{CLN}/elevage/wahis_animal_disease.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    agg.to_csv(out, index=False)
    print(f"   ✓ → {out} ({len(agg)} lignes, {agg['ISO'].nunique()} pays)")
    break
else:
    print("   ✗ WAHIS introuvable")


# ════════════════════════════════════════════════════════════════════════
# 2. GLW4 — Cattle density (GeoTIFF) — zonal stats par pays
# ════════════════════════════════════════════════════════════════════════
print("\n[2] 🐄 GLW4 Cattle Density (raster → zonal stats)…")
glw4 = f"{RAW}/elevage/GLW4-2020.D-DA.GLEAM3-ALL-LU.tif"
if os.path.exists(glw4):
    # Need country shapefile : Natural Earth via geopandas
    try:
        import rasterio
        from rasterstats import zonal_stats
        import geopandas as gpd

        # Try via geopandas built-in datasets (deprecated in geopandas 1.x)
        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            print(f"   ✓ Shapefile pays via geopandas ({len(world)} pays)")
        except Exception:
            # Download Natural Earth shapefile
            print(f"   Téléchargement Natural Earth countries shapefile…")
            ne_url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
            ne_zip = f"{RAW}/geographie/ne_110m_countries.zip"
            os.makedirs(os.path.dirname(ne_zip), exist_ok=True)
            if not os.path.exists(ne_zip):
                req = urllib.request.Request(ne_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    with open(ne_zip, "wb") as f:
                        f.write(r.read())
            ne_dir = f"{RAW}/geographie/ne_110m_countries"
            os.makedirs(ne_dir, exist_ok=True)
            with zipfile.ZipFile(ne_zip) as z:
                z.extractall(ne_dir)
            world = gpd.read_file(f"{ne_dir}/ne_110m_admin_0_countries.shp")
            print(f"   ✓ Natural Earth shapefile ({len(world)} pays)")

        # Inspecter raster
        with rasterio.open(glw4) as r:
            print(f"   Raster: shape={r.shape}, crs={r.crs}, bounds={r.bounds}")
            print(f"   nodata={r.nodata}, dtype={r.dtypes[0]}")

        # Reprojeter shapefile au CRS raster si nécessaire
        world_r = world.to_crs("EPSG:4326")  # GLW4 est en WGS84

        # Zonal stats : compter densité totale par pays
        print(f"   Calcul zonal stats (peut prendre 1-2 min)…")
        stats = zonal_stats(world_r, glw4, stats=["sum","mean","max","count"],
                              nodata=-1, geojson_out=False)
        # Combine
        results = []
        iso_col = "ISO_A3" if "ISO_A3" in world_r.columns else "iso_a3"
        name_col = "NAME" if "NAME" in world_r.columns else "name"
        for i, s in enumerate(stats):
            iso3 = world_r.iloc[i][iso_col] if iso_col in world_r.columns else None
            country = world_r.iloc[i][name_col] if name_col in world_r.columns else None
            results.append({
                "ISO3": iso3,
                "Country": country,
                "glw4_cattle_total": s.get("sum"),
                "glw4_cattle_mean": s.get("mean"),
                "glw4_cattle_max": s.get("max"),
                "glw4_cattle_count_pixels": s.get("count"),
            })
        df_glw = pd.DataFrame(results)
        df_glw["ISO"] = df_glw["ISO3"].apply(iso3_to_iso2)
        df_glw = df_glw.dropna(subset=["ISO"])
        df_glw["Annee"] = 2020  # GLW4 est statique pour 2020
        cols_out = ["ISO","Annee","glw4_cattle_total","glw4_cattle_mean","glw4_cattle_max"]
        df_glw[cols_out].to_csv(f"{CLN}/elevage/glw4_cattle_density.csv", index=False)
        print(f"   ✓ → {CLN}/elevage/glw4_cattle_density.csv ({len(df_glw)} pays)")
    except Exception as e:
        import traceback
        print(f"   ⚠️ GLW4: {str(e)[:200]}")
        # print(traceback.format_exc()[:500])
else:
    print(f"   ✗ {glw4} absent")


# ════════════════════════════════════════════════════════════════════════
# 3. MRDS — USGS Mineral Resources Data System
# ════════════════════════════════════════════════════════════════════════
print("\n[3] ⛏️ MRDS USGS Mineral Resources (points → counts par pays)…")
mrds = f"{RAW}/geologie/mrds.csv"
if os.path.exists(mrds):
    df = pd.read_csv(mrds, low_memory=False)
    print(f"   {df.shape}, cols: {list(df.columns)[:10]}")
    if "country" in df.columns and "com_type" in df.columns:
        df["ISO"] = df["country"].apply(name_to_iso2)
        df = df.dropna(subset=["ISO"])
        # Compter par pays + par type commodité
        agg = df.groupby("ISO").agg(
            mrds_sites_total=("dep_id", "count"),
            mrds_commodities_unique=("com_type", "nunique"),
        ).reset_index()
        agg["Annee"] = 2020  # snapshot statique
        agg.to_csv(f"{CLN}/geologie/mrds_minerals_by_country.csv", index=False)
        print(f"   ✓ → {CLN}/geologie/mrds_minerals_by_country.csv ({len(agg)} pays)")


# ════════════════════════════════════════════════════════════════════════
# 4. Global Power Plants
# ════════════════════════════════════════════════════════════════════════
print("\n[4] ⚡ Global Power Plants (capacité par pays)…")
for src in [f"{RAW}/energie/global_power_plants.csv",
             f"{RAW}/geologie/global_power_plants.csv"]:
    if not os.path.exists(src): continue
    df = pd.read_csv(src, low_memory=False)
    print(f"   {df.shape}, cols: {list(df.columns)[:10]}")
    # Format World Resources Institute Global Power Plant Database
    # typique : country, country_long, name, gppd_idnr, capacity_mw, latitude, longitude, primary_fuel
    iso_col = next((c for c in df.columns if c.lower() in ("country","country_iso","iso","country_code")), None)
    cap_col = next((c for c in df.columns if "capacity" in c.lower()), None)
    fuel_col = next((c for c in df.columns if "fuel" in c.lower() or "primary_fuel" in c.lower()), None)
    if iso_col and cap_col:
        df[cap_col] = pd.to_numeric(df[cap_col], errors="coerce")
        df = df.dropna(subset=[cap_col])
        # ISO mapping
        if df[iso_col].astype(str).str.len().mean() == 3:
            df["ISO"] = df[iso_col].apply(iso3_to_iso2)
        else:
            df["ISO"] = df[iso_col].apply(name_to_iso2)
        df = df.dropna(subset=["ISO"])
        # Capacité totale + par fuel
        if fuel_col:
            piv = df.pivot_table(index="ISO", columns=fuel_col, values=cap_col, aggfunc="sum").reset_index()
            rename = {c: f"powerplant_{str(c).lower().replace(' ','_')[:25]}_mw"
                       for c in piv.columns if c != "ISO"}
            piv = piv.rename(columns=rename)
            piv["powerplant_total_mw"] = piv[[c for c in piv.columns if c.startswith("powerplant_")]].sum(axis=1)
            piv["powerplant_n_plants"] = df.groupby("ISO").size().reindex(piv["ISO"]).values
            piv["Annee"] = 2024
            piv.to_csv(f"{CLN}/energie/global_power_plants_by_country.csv", index=False)
            print(f"   ✓ → {CLN}/energie/global_power_plants_by_country.csv ({len(piv)} pays × {piv.shape[1]-2} fuels)")
        else:
            agg = df.groupby("ISO").agg(
                powerplant_total_mw=(cap_col, "sum"),
                powerplant_n_plants=(iso_col, "count"),
            ).reset_index()
            agg["Annee"] = 2024
            agg.to_csv(f"{CLN}/energie/global_power_plants_by_country.csv", index=False)
            print(f"   ✓ → {CLN}/energie/global_power_plants_by_country.csv ({len(agg)} pays)")
        break


# ════════════════════════════════════════════════════════════════════════
# 5. Global Oil & Gas Extraction Tracker
# ════════════════════════════════════════════════════════════════════════
print("\n[5] 🛢️ Global Oil & Gas Extraction Tracker (xlsx)…")
src = f"{RAW}/geologie/Global-Oil-and-Gas-Extraction-Tracker-March-2026.xlsx"
if os.path.exists(src):
    try:
        sheets = pd.read_excel(src, sheet_name=None, engine="openpyxl")
        print(f"   Sheets: {list(sheets.keys())[:5]}")
        # Le sheet principal a typiquement les champs : Country, Production, Reserves, Status
        for sheet_name, df in sheets.items():
            if df.shape[0] < 10: continue
            print(f"   Sheet '{sheet_name}': {df.shape}")
            country_col = next((c for c in df.columns if "country" in str(c).lower()), None)
            if not country_col: continue
            df["ISO"] = df[country_col].apply(name_to_iso2)
            df = df.dropna(subset=["ISO"])
            # Agréger : nombre de projets par pays
            agg_cols = {}
            for c in df.columns:
                lc = str(c).lower()
                if any(k in lc for k in ["production","reserves","capacity","mboed","mboe"]):
                    val = pd.to_numeric(df[c], errors="coerce")
                    df[c] = val
            num_cols = [c for c in df.select_dtypes(include="number").columns
                         if c not in ("ISO",)]
            agg_d = {"n_oilgas_projects": ("ISO", "count")}
            for c in num_cols[:5]:
                agg_d[f"oilgas_{str(c).lower().replace(' ','_')[:20]}_sum"] = (c, "sum")
            agg = df.groupby("ISO").agg(**agg_d).reset_index()
            agg["Annee"] = 2026
            agg.to_csv(f"{CLN}/geologie/oil_gas_extraction_by_country.csv", index=False)
            print(f"   ✓ → {CLN}/geologie/oil_gas_extraction_by_country.csv ({len(agg)} pays)")
            break
    except Exception as e:
        print(f"   ⚠️ {str(e)[:120]}")


# ════════════════════════════════════════════════════════════════════════
# 6. Global Coal Mine Tracker (x3 xlsx)
# ════════════════════════════════════════════════════════════════════════
print("\n[6] ⛏️ Global Coal Mine Tracker (3 xlsx)…")
coal_files = glob.glob(f"{RAW}/geologie/Global*Coal*Tracker*.xlsx")
print(f"   {len(coal_files)} fichiers Coal Tracker")
all_coal = []
for src in coal_files:
    try:
        sheets = pd.read_excel(src, sheet_name=None, engine="openpyxl")
        for sn, df in sheets.items():
            if df.shape[0] < 5: continue
            # Cherche pays
            country_col = next((c for c in df.columns if "country" in str(c).lower()), None)
            if country_col:
                df["src_file"] = os.path.basename(src)
                df["src_sheet"] = sn
                all_coal.append(df)
                break
    except Exception as e:
        print(f"   ⚠️ {os.path.basename(src)}: {str(e)[:80]}")

if all_coal:
    # Merge all
    main = all_coal[0]  # main tracker
    country_col = next((c for c in main.columns if "country" in str(c).lower()), None)
    capacity_col = next((c for c in main.columns if "capacity" in str(c).lower()), None)
    if not capacity_col:
        capacity_col = next((c for c in main.columns if "production" in str(c).lower()
                              or "output" in str(c).lower() or "mtpa" in str(c).lower()), None)
    main["ISO"] = main[country_col].apply(name_to_iso2)
    main = main.dropna(subset=["ISO"])
    print(f"   Main tracker: {main.shape}, capacity_col={capacity_col}")

    agg_d = {"coal_n_mines": ("ISO", "count")}
    if capacity_col:
        main[capacity_col] = pd.to_numeric(main[capacity_col], errors="coerce")
        agg_d["coal_capacity_mtpa_sum"] = (capacity_col, "sum")
    agg = main.groupby("ISO").agg(**agg_d).reset_index()
    agg["Annee"] = 2026
    agg.to_csv(f"{CLN}/geologie/global_coal_mines_by_country.csv", index=False)
    print(f"   ✓ → {CLN}/geologie/global_coal_mines_by_country.csv ({len(agg)} pays)")


# ════════════════════════════════════════════════════════════════════════
# 7. Global Iron Ore Mines Tracker
# ════════════════════════════════════════════════════════════════════════
print("\n[7] ⛏️ Global Iron Ore Mines Tracker…")
src = f"{RAW}/geologie/Global-Iron-Ore-Mines-Tracker-August-2025-V1.xlsx"
if os.path.exists(src):
    try:
        sheets = pd.read_excel(src, sheet_name=None, engine="openpyxl")
        for sn, df in sheets.items():
            if df.shape[0] < 5: continue
            country_col = next((c for c in df.columns if "country" in str(c).lower()), None)
            if not country_col: continue
            df["ISO"] = df[country_col].apply(name_to_iso2)
            df = df.dropna(subset=["ISO"])
            agg = df.groupby("ISO").agg(iron_n_mines=("ISO", "count")).reset_index()
            agg["Annee"] = 2025
            agg.to_csv(f"{CLN}/geologie/global_iron_mines_by_country.csv", index=False)
            print(f"   ✓ → {CLN}/geologie/global_iron_mines_by_country.csv ({len(agg)} pays)")
            break
    except Exception as e:
        print(f"   ⚠️ {str(e)[:120]}")


# ════════════════════════════════════════════════════════════════════════
# 8. Production-Consumption Met Coal Iron Ore (Steel Industry)
# ════════════════════════════════════════════════════════════════════════
print("\n[8] 🏭 Production-Consumption Met Coal Iron Ore (Steel)…")
src = f"{RAW}/geologie/Production-Consumption-of-Met-Coal-Iron-Ore-by-Steel-Industry-December-2025-Standard-Copy-V1.xlsx"
if os.path.exists(src):
    try:
        sheets = pd.read_excel(src, sheet_name=None, engine="openpyxl")
        print(f"   Sheets: {list(sheets.keys())[:5]}")
        for sn, df in sheets.items():
            if df.shape[0] < 5: continue
            country_col = next((c for c in df.columns if "country" in str(c).lower()), None)
            if country_col:
                df["ISO"] = df[country_col].apply(name_to_iso2)
                df_v = df.dropna(subset=["ISO"])
                if len(df_v) > 5:
                    df_v.to_csv(f"{CLN}/geologie/steel_industry_raw.csv", index=False)
                    print(f"   ✓ → {CLN}/geologie/steel_industry_raw.csv ({len(df_v)} lignes, sheet {sn})")
                    break
    except Exception as e:
        print(f"   ⚠️ {str(e)[:120]}")


# ════════════════════════════════════════════════════════════════════════
# 9. earthquakes_usgs.csv
# ════════════════════════════════════════════════════════════════════════
print("\n[9] 🌍 earthquakes_usgs.csv…")
src = f"{RAW}/geologie/earthquakes_usgs.csv"
if os.path.exists(src):
    df = pd.read_csv(src, low_memory=False)
    print(f"   {df.shape}, cols: {list(df.columns)[:8]}")
    # Format USGS : time, latitude, longitude, depth, mag, magType, ...
    if "latitude" in df.columns and "longitude" in df.columns:
        try:
            import reverse_geocoder as rg
            print(f"   Reverse-geocoding {len(df):,} séismes...")
            coords = list(zip(df["latitude"], df["longitude"]))
            results = rg.search(coords, mode=1, verbose=False)
            df["ISO"] = [r["cc"] for r in results]
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df["Annee"] = df["time"].dt.year
            df = df.dropna(subset=["Annee"])
            df["Annee"] = df["Annee"].astype(int)
            df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
            agg = df.groupby(["ISO","Annee"]).agg(
                eq_count=("mag","count"),
                eq_max_mag=("mag","max"),
                eq_mean_mag=("mag","mean"),
                eq_mag_ge6=("mag", lambda s: (s>=6).sum()),
            ).reset_index()
            agg.to_csv(f"{CLN}/geologie/earthquakes_by_country_year.csv", index=False)
            print(f"   ✓ → {CLN}/geologie/earthquakes_by_country_year.csv ({len(agg)} lignes)")
        except Exception as e:
            print(f"   ⚠️ {str(e)[:120]}")


# ════════════════════════════════════════════════════════════════════════
# 10. volcanoes.csv + tectonic_plates.geojson
# ════════════════════════════════════════════════════════════════════════
print("\n[10] 🌋 Volcanoes + Tectonic Plates…")
src = f"{RAW}/geologie/volcanoes.csv"
if os.path.exists(src):
    try:
        df = pd.read_csv(src, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(src, low_memory=False, encoding="latin-1")
    print(f"   volcanoes: {df.shape}, cols: {list(df.columns)[:8]}")
    country_col = next((c for c in df.columns if "country" in str(c).lower()), None)
    if country_col:
        df["ISO"] = df[country_col].apply(name_to_iso2)
        df = df.dropna(subset=["ISO"])
        agg = df.groupby("ISO").agg(
            volcanoes_count=("ISO","count"),
        ).reset_index()
        agg["Annee"] = 2020
        agg.to_csv(f"{CLN}/geologie/volcanoes_by_country.csv", index=False)
        print(f"   ✓ → {CLN}/geologie/volcanoes_by_country.csv ({len(agg)} pays)")

# Tectonic plates (geojson) — pas vraiment de "par pays", mais on peut compter pour info
src = f"{RAW}/geologie/tectonic_plates.geojson"
if os.path.exists(src):
    try:
        import geopandas as gpd
        plates = gpd.read_file(src)
        print(f"   Tectonic plates: {len(plates)} polygones")
        # Compter le nombre de plates intersectant chaque pays
        try:
            world = gpd.read_file(f"{RAW}/geographie/ne_110m_countries/ne_110m_admin_0_countries.shp")
            joined = gpd.sjoin(world, plates, how="left", predicate="intersects")
            iso_col = "ISO_A3" if "ISO_A3" in world.columns else "iso_a3"
            agg = joined.groupby(iso_col).size().reset_index(name="tectonic_plates_count")
            agg = agg.rename(columns={iso_col:"ISO3"})
            agg["ISO"] = agg["ISO3"].apply(iso3_to_iso2)
            agg = agg.dropna(subset=["ISO"])
            agg["Annee"] = 2020
            agg[["ISO","Annee","tectonic_plates_count"]].to_csv(
                f"{CLN}/geologie/tectonic_plates_by_country.csv", index=False)
            print(f"   ✓ → {CLN}/geologie/tectonic_plates_by_country.csv ({len(agg)} pays)")
        except Exception as e:
            print(f"   ⚠️ shapefile: {str(e)[:100]}")
    except Exception as e:
        print(f"   ⚠️ {str(e)[:120]}")


# ════════════════════════════════════════════════════════════════════════
# 11. IRENA hierarchical CSVs (R-ELECCAP, R-ELECGEN, RESHARE, HEATGEN)
# ════════════════════════════════════════════════════════════════════════
print("\n[11] ⚡ IRENA hierarchical CSVs (parser tabulaire)…")

def parse_irena_csv(path, label_col):
    """IRENA CSV format : ligne titre, puis lignes hiérarchiques."""
    if not os.path.exists(path): return None
    # IRENA exports CSV utilisent format pivot avec colonnes décalées
    # Approche : lire avec read_csv en mode "robuste" puis nettoyer
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    # ligne 0 = titre, ligne 1 = en-têtes
    # Tab-separated en réalité ? Test
    sep = "\t" if "\t" in lines[1] else ","
    try:
        df = pd.read_csv(path, skiprows=1, sep=sep, low_memory=False)
    except Exception as e:
        return None
    return df

irena_files = [
    ("R-ELECCAP_20260624-114520.csv", "irena_capacity_mw"),
    ("R-ELECGEN_20260624-114636.csv", "irena_generation_gwh"),
    ("RESHARE_20260624-114707.csv",   "irena_renewable_share_pct"),
    ("HEATGEN_20260624-114228.csv",   "irena_heat_generation_tj"),
]
for fname, col_name in irena_files:
    for src in [f"{RAW}/energie/{fname}"]:
        if not os.path.exists(src): continue
        try:
            df = parse_irena_csv(src, col_name)
            if df is None or df.empty:
                print(f"   ⚠️ {fname}: read failed")
                continue
            print(f"   {fname}: {df.shape}, cols: {list(df.columns)[:6]}")
            # Cherche cols année + valeur + pays
            # IRENA exports : Region | Country/area | Technology | ... | Year | Value
            country_col = next((c for c in df.columns if "country" in str(c).lower()
                                  or "region" in str(c).lower() or "area" in str(c).lower()), df.columns[0])
            year_col = next((c for c in df.columns if "year" in str(c).lower()), None)
            val_col = df.columns[-1]
            if year_col is None:
                # Probably wide format → year cols
                year_cols = [c for c in df.columns if isinstance(c, str) and c.isdigit()
                              and 1900 < int(c) < 2030]
                if year_cols:
                    id_cols = [c for c in df.columns if c not in year_cols]
                    long = df.melt(id_vars=id_cols, value_vars=year_cols,
                                    var_name="Annee", value_name="Value")
                    long["Annee"] = pd.to_numeric(long["Annee"], errors="coerce").astype("Int64")
                else:
                    continue
            else:
                long = df.rename(columns={year_col:"Annee", val_col:"Value"})
                long["Annee"] = pd.to_numeric(long["Annee"], errors="coerce")
                long["Value"] = pd.to_numeric(long["Value"], errors="coerce")

            long = long.dropna(subset=["Annee","Value"])
            long["Annee"] = long["Annee"].astype(int)
            # Filtrer pays valides
            long["ISO"] = long[country_col].apply(name_to_iso2)
            long = long.dropna(subset=["ISO"])
            agg = long.groupby(["ISO","Annee"])["Value"].sum().reset_index()
            agg = agg.rename(columns={"Value": col_name})
            out = f"{CLN}/energie/{fname.split('_')[0].lower()}_by_country.csv"
            agg.to_csv(out, index=False)
            print(f"   ✓ → {out} ({len(agg)} lignes)")
        except Exception as e:
            print(f"   ⚠️ {fname}: {str(e)[:120]}")
        break


# ════════════════════════════════════════════════════════════════════════
# 12. SPAM Harvested Area
# ════════════════════════════════════════════════════════════════════════
print("\n[12] 🌾 SPAM Harvested Area (par culture)…")
spam_h = f"{RAW}/agriculture/spam2020_v2r2_harvested/spam2020V2r2_global_harvested_area/spam2020V2r2_global_H_TA.csv"
if os.path.exists(spam_h):
    head = pd.read_csv(spam_h, nrows=1)
    crop_cols = [c for c in head.columns if c not in
                  ("grid_code","x","y","FIPS0","FIPS1","FIPS2","ADM0_NAME","ADM1_NAME","ADM2_NAME")]
    print(f"   {len(crop_cols)} cultures")

    sum_d = {}
    chunk_idx = 0
    for chunk in pd.read_csv(spam_h, usecols=["ADM0_NAME"]+crop_cols, chunksize=100000, low_memory=False):
        chunk_idx += 1
        for country, sub in chunk.groupby("ADM0_NAME"):
            if country not in sum_d:
                sum_d[country] = {c: 0.0 for c in crop_cols}
            for c in crop_cols:
                vals = sub[c]
                vals_nz = vals[vals > 0]
                sum_d[country][c] += vals_nz.sum()
        if chunk_idx % 3 == 0: print(f"     chunk {chunk_idx}…")
    print(f"   {chunk_idx} chunks, {len(sum_d)} pays")

    rows = []
    for country in sum_d:
        row = {"ADM0_NAME": country}
        for c in crop_cols:
            row[f"spam_harvest_{c}"] = sum_d[country][c]
        rows.append(row)
    df_h = pd.DataFrame(rows)
    df_h["ISO"] = df_h["ADM0_NAME"].apply(name_to_iso2)
    df_h = df_h.dropna(subset=["ISO"])
    df_h["Annee"] = 2020
    cols_out = ["ISO","Annee"] + [c for c in df_h.columns if c.startswith("spam_harvest_")]
    df_h[cols_out].to_csv(f"{CLN}/agriculture/spam2020_v2_harvested_area_by_country.csv", index=False)
    print(f"   ✓ → {CLN}/agriculture/spam2020_v2_harvested_area_by_country.csv ({len(df_h)} pays × {len(crop_cols)} cultures)")


# ════════════════════════════════════════════════════════════════════════
# 13. CMM Satellite
# ════════════════════════════════════════════════════════════════════════
print("\n[13] 🛰️ CMM Satellite Monitoring…")
src = f"{RAW}/atmosphere/download_cmm_satellite.xlsx"
if os.path.exists(src):
    try:
        sheets = pd.read_excel(src, sheet_name=None, engine="openpyxl")
        print(f"   Sheets: {list(sheets.keys())}")
        for sn, df in sheets.items():
            iso_col = next((c for c in df.columns if "COUNTRY_CODE" in str(c) or "ISO" in str(c).upper()), None)
            if iso_col:
                df["ISO"] = df[iso_col].apply(lambda c: iso3_to_iso2(c) if isinstance(c,str) and len(c)==3 else None)
                df = df.dropna(subset=["ISO"])
                if "PCT_COAL_PRODUCTION_WITH_FAVOURABLE_CONDITIONS_FOR_SATELLITE_METHANE_MONITORING" in df.columns:
                    df_v = df[["ISO","PCT_COAL_PRODUCTION_WITH_FAVOURABLE_CONDITIONS_FOR_SATELLITE_METHANE_MONITORING"]].copy()
                    df_v.columns = ["ISO","cmm_satellite_pct_favorable"]
                    df_v["Annee"] = 2024
                    df_v.to_csv(f"{CLN}/atmosphere/cmm_satellite_monitoring.csv", index=False)
                    print(f"   ✓ → {CLN}/atmosphere/cmm_satellite_monitoring.csv ({len(df_v)} pays)")
                    break
    except Exception as e:
        print(f"   ⚠️ {str(e)[:120]}")


# ════════════════════════════════════════════════════════════════════════
# BILAN FINAL
# ════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════════════════════════")
print("📊 BILAN — NOUVEAUX FICHIERS CLEANED CRÉÉS")
print("══════════════════════════════════════════════════════════════")
new_keywords = ["wahis","glw4","mrds_minerals","power_plants","oil_gas_extraction",
                 "coal_mines","iron_mines","steel_industry","earthquakes_by_country",
                 "volcanoes_by_country","tectonic_plates","irena_","r-elec","r_elec",
                 "spam2020_v2_harvested","cmm_satellite"]
created = []
for cat in ["atmosphere","hydrologie","sol_ecologie","agriculture","elevage",
             "peche","energie","geologie"]:
    d = f"{CLN}/{cat}"
    if os.path.isdir(d):
        for f in os.listdir(d):
            if any(k in f.lower() for k in new_keywords):
                sz = os.path.getsize(f"{d}/{f}") / 1024
                created.append((cat, f, sz))
                print(f"  ✓ [{cat}] {f}  ({sz:.0f} KB)")
print(f"\n{len(created)} nouveaux fichiers cleaned")
