"""
fetch_user_datasets.py — Téléchargement auto des 6 catégories demandées.

Pour chaque source :
  - Tente le téléchargement direct (URL stable trouvée)
  - Si échec → marque "MANUAL" avec instructions
  - Sauvegarde dans data/raw/<category>/ ou data/cleaned/

Catégories :
  1. Atmosphère
  2. Hydrologie
  3. Sol & Écologie
  4. Agriculture
  5. Élevage
  6. Énergie
"""
import os, sys, io, urllib.request, urllib.parse, json, zipfile, time
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
RAW = "data/raw"
CLN = "data/cleaned"


def fetch(url, out_path, timeout=120, max_mb=500):
    """Téléchargement direct. Retourne True si OK."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "*/*",
        })
        with urllib.request.urlopen(req, timeout=timeout) as r:
            cl = int(r.headers.get("Content-Length", 0))
            if cl > max_mb * 1e6:
                return False, f"Trop gros ({cl/1e6:.0f}MB > {max_mb}MB)"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk: break
                    f.write(chunk)
            sz = os.path.getsize(out_path) / 1e6
            return True, f"{sz:.1f} MB"
    except Exception as e:
        return False, str(e)[:80]


def fetch_text(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


results = []
def report(category, name, status, detail, url):
    sym = "✓" if status == "OK" else ("⚠️" if status == "MANUAL" else "✗")
    print(f"  {sym} {name:40s} {status:8s} {detail}")
    results.append({"Categorie": category, "Source": name, "Status": status,
                     "Detail": detail, "URL": url})


# ════════════════════════════════════════════════════════════════════════
print("\n══════ 1. ATMOSPHÈRE ══════")
# ════════════════════════════════════════════════════════════════════════
cat_dir = f"{RAW}/atmosphere_extras"
os.makedirs(cat_dir, exist_ok=True)

# EDGAR GHG 2024 — direct download
url = "https://edgar.jrc.ec.europa.eu/dataset_ghg2024"  # page HTML
# EDGAR vraies données : files-edgar...
edgar_urls = [
    ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/v90_GHG_2024/v90_GHG_TOTALS_1970-2023.zip",
     "EDGAR_GHG_totals_1970-2023.zip"),
    ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/datasets/v90_GHG_2024/v90_GHG_BY_SECTOR_1970-2023.zip",
     "EDGAR_GHG_by_sector.zip"),
]
for u, fname in edgar_urls:
    ok, det = fetch(u, f"{cat_dir}/{fname}", max_mb=300)
    report("Atmosphere", fname, "OK" if ok else "FAIL", det, u)

# Climate Watch CAIT historical emissions — direct CSV
cw_url = "https://www.climatewatchdata.org/data-explorer/historical-emissions"
# Vraie URL CSV : (testée)
cw_csv = "https://www.climatewatchdata.org/api/v1/data/historical_emissions/download.csv?source_ids[]=83&gas_ids[]=224"
ok, det = fetch(cw_csv, f"{cat_dir}/climate_watch_emissions.csv")
report("Atmosphere", "Climate Watch CAIT", "OK" if ok else "MANUAL",
       det if ok else "→ Aller manuellement sur le site", cw_url)

# NOAA Ozone (ozonesondes mondial)
# Le site /gmd/ozwv/ a des fichiers FTP NetCDF par station
noaa_url = "https://gml.noaa.gov/aftp/data/ozwv/Ozonesonde/Ascension Island/100 Meter Average Files/ascen_201001_100m.l100"
# Trop spécifique — on documente comme manuel
report("Atmosphere", "NOAA Ozone", "MANUAL",
       "Données ozonesondes par station (NetCDF FTP)", "https://www.esrl.noaa.gov/gmd/ozwv/")

# NASA MCD19A2 AOD — Earthdata login obligatoire
report("Atmosphere", "NASA MCD19A2 AOD", "MANUAL",
       "Earthdata login requis (gratuit)", "https://lpdaac.usgs.gov/products/mcd19a2v061/")


# ════════════════════════════════════════════════════════════════════════
print("\n══════ 2. HYDROLOGIE ══════")
# ════════════════════════════════════════════════════════════════════════
cat_dir = f"{RAW}/hydrologie_extras"
os.makedirs(cat_dir, exist_ok=True)

# GRDC — Login obligatoire pour download via portail
report("Hydrologie", "GRDC River Discharge", "MANUAL",
       "Inscription obligatoire (gratuit) sur le portail GRDC",
       "https://www.bafg.de/GRDC/EN/02_srvcs/21_tmsrs/210_prtl/prtl_node.html")

# NASA GRACE-FO Groundwater — direct sur nasagrace
grace_url = "https://nasagrace.unl.edu/data/groundwater/gws_anomaly_global.tif"
ok, det = fetch(grace_url, f"{cat_dir}/grace_gws_anomaly.tif")
report("Hydrologie", "NASA GRACE Groundwater", "OK" if ok else "MANUAL",
       det, "https://nasagrace.unl.edu/")

# WRI Aqueduct 4.0 — Excel direct
aqueduct_urls = [
    ("https://files.wri.org/d8/s3fs-public/2023-07/Aqueduct40_baseline_annual_y2023m07d05.zip",
     "WRI_Aqueduct40_baseline.zip"),
    ("https://files.wri.org/d8/s3fs-public/2023-07/Aqueduct40_future_annual_y2023m07d05.zip",
     "WRI_Aqueduct40_future.zip"),
]
for u, fname in aqueduct_urls:
    ok, det = fetch(u, f"{cat_dir}/{fname}", max_mb=300)
    report("Hydrologie", fname, "OK" if ok else "FAIL", det, u)

# FAO AQUASTAT — interface de query, pas de bulk download direct
# Mais ils ont des CSV pré-faits :
aquastat_urls = [
    "https://www.fao.org/aquastat/statistics/query/results.html?regionQuery=true&yearGrouping=BYTH",
]
# Try main metadata file
aqu_csv = "https://www.fao.org/aquastat/databases/maindatabase/AQUASTAT_main_database.zip"
ok, det = fetch(aqu_csv, f"{cat_dir}/aquastat_main.zip", max_mb=100)
report("Hydrologie", "FAO AQUASTAT", "OK" if ok else "MANUAL",
       det, "https://www.fao.org/aquastat/statistics/query/index.html?lang=en")

# NOAA COBE2 SST (NetCDF)
cobe_url = "https://psl.noaa.gov/thredds/fileServer/Datasets/COBE2/sst.mon.mean.nc"
ok, det = fetch(cobe_url, f"{cat_dir}/cobe2_sst.nc", max_mb=200)
report("Hydrologie", "NOAA COBE2 SST", "OK" if ok else "MANUAL",
       det, "https://psl.noaa.gov/data/gridded/data.cobe2.html")


# ════════════════════════════════════════════════════════════════════════
print("\n══════ 3. SOL & ÉCOLOGIE ══════")
# ════════════════════════════════════════════════════════════════════════
cat_dir = f"{RAW}/sol_ecologie_extras"
os.makedirs(cat_dir, exist_ok=True)

# NASA FIRMS — API requiert MAP_KEY (inscription gratuite)
firms_url = "https://firms.modaps.eosdis.nasa.gov/api/country/csv/YOUR_MAP_KEY/MODIS_NRT/USA/10"
report("Sol_Ecologie", "NASA FIRMS Fires", "MANUAL",
       "MAP_KEY API gratuit (inscription rapide)",
       "https://firms.modaps.eosdis.nasa.gov/active_fire/")

# MODIS NDVI — Earthdata login
report("Sol_Ecologie", "MODIS NDVI/EVI", "MANUAL",
       "Earthdata login + download par tile",
       "https://modis.gsfc.nasa.gov/data/dataprod/mod13.php")

# IUCN Red List
report("Sol_Ecologie", "IUCN Red List", "MANUAL",
       "Inscription + export par pays",
       "https://www.iucnredlist.org/search/stats")

# Yale EPI 2024 — Excel direct
epi_urls = [
    ("https://epi.yale.edu/downloads/epi2024results06112024.xlsx", "EPI_2024_results.xlsx"),
    ("https://epi.yale.edu/downloads/epi2024expandedresults06112024.xlsx", "EPI_2024_expanded.xlsx"),
]
for u, fname in epi_urls:
    ok, det = fetch(u, f"{cat_dir}/{fname}", max_mb=50)
    report("Sol_Ecologie", fname, "OK" if ok else "FAIL", det, u)

# Global Forest Watch tree cover loss (CSV direct)
gfw_urls = [
    "https://data.globalforestwatch.org/api/download/v1/dataset/umd_tree_cover_loss/version/v1.10/format/csv",
]
for u in gfw_urls:
    ok, det = fetch(u, f"{cat_dir}/gfw_tree_cover_loss.csv", max_mb=100)
    report("Sol_Ecologie", "GFW Tree Cover Loss", "OK" if ok else "MANUAL",
           det, "https://data.globalforestwatch.org/")


# ════════════════════════════════════════════════════════════════════════
print("\n══════ 4. AGRICULTURE ══════")
# ════════════════════════════════════════════════════════════════════════
cat_dir = f"{RAW}/agriculture_extras"
os.makedirs(cat_dir, exist_ok=True)

# SPAM 2020 — direct download zip (Harvard Dataverse)
spam_urls = [
    ("https://dataverse.harvard.edu/api/access/datafile/10056008", "SPAM2020_V1r0_yield.zip"),  # might not work
    ("https://s3.amazonaws.com/mapspam/2020/v1r0/spam2020_v1r0_global_yield.zip", "spam2020_yield.zip"),
    ("https://s3.amazonaws.com/mapspam/2020/v1r0/spam2020_v1r0_global_production.zip", "spam2020_production.zip"),
]
spam_ok = False
for u, fname in spam_urls:
    ok, det = fetch(u, f"{cat_dir}/{fname}", max_mb=500)
    if ok:
        spam_ok = True
    report("Agriculture", fname, "OK" if ok else "FAIL", det, u)
if not spam_ok:
    report("Agriculture", "SPAM 2020", "MANUAL",
           "Téléchargement direct depuis mapspam.info (zip ~1GB)",
           "https://www.mapspam.info/data/")

# FAO Crop Calendar
report("Agriculture", "FAO Crop Calendar", "MANUAL",
       "Interface interactive, pas de bulk", "https://cropcalendar.apps.fao.org/")

# FAOSTAT Fertilizers by Nutrient (RFN)
faostat_bulk = [
    ("https://bulks-faostat.fao.org/production/Inputs_FertilizersNutrient_E_All_Data_(Normalized).zip",
     "FAO_fertilizers_nutrient.zip"),
    ("https://bulks-faostat.fao.org/production/Inputs_Pesticides_Use_E_All_Data_(Normalized).zip",
     "FAO_pesticides_use.zip"),
    ("https://bulks-faostat.fao.org/production/Inputs_LandUse_E_All_Data_(Normalized).zip",
     "FAO_land_use_inputs.zip"),
    ("https://bulks-faostat.fao.org/production/Inputs_Machinery_E_All_Data_(Normalized).zip",
     "FAO_machinery.zip"),
]
for u, fname in faostat_bulk:
    ok, det = fetch(u, f"{cat_dir}/{fname}", max_mb=300)
    report("Agriculture", fname, "OK" if ok else "FAIL", det, u)

# GAEZ 4.0 — manuel (interface complexe)
report("Agriculture", "GAEZ 4.0", "MANUAL",
       "Interface géo, download par culture",
       "https://gaez.fao.org/pages/data-access-download")

# CGIAR varieties
cgiar_urls = [
    ("https://data.cgiar.org/dataset/varieties-1980-2014/resource/varieties.csv", "cgiar_varieties.csv"),
]
for u, fname in cgiar_urls:
    ok, det = fetch(u, f"{cat_dir}/{fname}", max_mb=20)
    report("Agriculture", "CGIAR Varieties", "OK" if ok else "MANUAL",
           det, "https://data.cgiar.org/dataset/varieties-1980-2014")

# WorldPop cropland
report("Agriculture", "WorldPop Cropland", "MANUAL",
       "Téléchargement GeoTIFF lourd par pays",
       "https://www.worldpop.org/methods/croplands")

# ESA WorldCover 2021 — manuel (S2 grid huge)
report("Agriculture", "ESA WorldCover 2021", "MANUAL",
       "10m global landcover, très lourd",
       "https://esa-worldcover.org/en")


# ════════════════════════════════════════════════════════════════════════
print("\n══════ 5. ÉLEVAGE ══════")
# ════════════════════════════════════════════════════════════════════════
cat_dir = f"{RAW}/elevage_extras"
os.makedirs(cat_dir, exist_ok=True)

# FAO Gridded Livestock (GLW4) — direct zips
glw_urls = [
    ("https://storage.googleapis.com/fao-maps-catalog-data/uuid/9d1e149b-d63f-4213-978b-317a8eb42d02/resources/GLW4-2020.D-DA.CTL.zip",
     "GLW4_cattle_2020.zip"),
]
for u, fname in glw_urls:
    ok, det = fetch(u, f"{cat_dir}/{fname}", max_mb=200)
    report("Elevage", fname, "OK" if ok else "MANUAL", det,
           "https://www.fao.org/livestock-systems/global-distributions/")

if all(r["Status"] != "OK" for r in results if r["Source"].startswith("GLW4")):
    report("Elevage", "FAO GLW4 Livestock", "MANUAL",
           "Téléchargement TIF par espèce",
           "https://www.fao.org/livestock-systems/global-distributions/cattle/en/")

# OECD-FAO Agricultural Outlook
oecd_urls = [
    ("https://stats.oecd.org/SDMX-JSON/data/HIGH_AGLINK_2024/A.WLD.ME.AN+PR+QP+QC+IM+EX+CR+ST+IM+EX.AGRO.A/all?",
     "oecd_fao_outlook.json"),
]
for u, fname in oecd_urls:
    ok, det = fetch(u, f"{cat_dir}/{fname}", max_mb=50)
    report("Elevage", "OECD-FAO Outlook", "OK" if ok else "MANUAL", det,
           "https://www.agri-outlook.org/data/")

# ILRI Research Data — manuel
report("Elevage", "ILRI Research Data", "MANUAL",
       "Multiples datasets, pas de bulk",
       "https://www.ilri.org/research-data")

# WAHIS OIE Animal Health — API authenticated
report("Elevage", "WAHIS Animal Health", "MANUAL",
       "API OIE, inscription requise",
       "https://wahis.woah.org/")

# FAO Pastoralism Hub
report("Elevage", "FAO Pastoralism", "MANUAL",
       "Documents et études, pas de dataset structuré",
       "https://www.fao.org/pastoralist-knowledge-hub/")


# ════════════════════════════════════════════════════════════════════════
print("\n══════ 6. ÉNERGIE ══════")
# ════════════════════════════════════════════════════════════════════════
cat_dir = f"{RAW}/energie_extras"
os.makedirs(cat_dir, exist_ok=True)

# IRENA Capacity Statistics
irena_urls = [
    ("https://www.irena.org/-/media/Files/IRENA/Agency/Statistics/Download-statistics/Capacity-and-Generation/CapacityStatistics2024.xlsx",
     "IRENA_capacity_2024.xlsx"),
    ("https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2024/Mar/IRENA_RE_Capacity_Statistics_2024.pdf",
     "IRENA_capacity_2024.pdf"),
]
for u, fname in irena_urls:
    ok, det = fetch(u, f"{cat_dir}/{fname}", max_mb=50)
    report("Energie", fname, "OK" if ok else "FAIL", det, u)

# Ember Climate — direct CSV
ember_urls = [
    ("https://ember-climate.org/app/uploads/2023/07/global-electricity-review-2023-monthly-data.csv",
     "ember_monthly_electricity.csv"),
    ("https://storage.googleapis.com/emb-prod-bkt-publicdata/public-downloads/yearly_full_release_long_format.csv",
     "ember_yearly_electricity.csv"),
]
for u, fname in ember_urls:
    ok, det = fetch(u, f"{cat_dir}/{fname}", max_mb=50)
    report("Energie", fname, "OK" if ok else "FAIL", det, u)

# WB SE4ALL via WB API (already partial)
se4all_urls = [
    ("https://api.worldbank.org/v2/country/all/indicator/EG.ELC.RNEW.ZS?format=json&per_page=20000",
     "wb_renewable_pct.json"),
    ("https://api.worldbank.org/v2/country/all/indicator/EG.FEC.RNEW.ZS?format=json&per_page=20000",
     "wb_renewable_final_pct.json"),
    ("https://api.worldbank.org/v2/country/all/indicator/EG.ELC.ACCS.ZS?format=json&per_page=20000",
     "wb_elec_access.json"),
]
for u, fname in se4all_urls:
    ok, det = fetch(u, f"{cat_dir}/{fname}", max_mb=20)
    report("Energie", fname, "OK" if ok else "FAIL", det, u)


# ════════════════════════════════════════════════════════════════════════
# RAPPORT FINAL
# ════════════════════════════════════════════════════════════════════════
print("\n\n══════════════════════════════════════════════════════════════")
print("📊 BILAN TÉLÉCHARGEMENTS")
print("══════════════════════════════════════════════════════════════")

df_report = pd.DataFrame(results)
counts = df_report.groupby(["Categorie","Status"]).size().unstack(fill_value=0)
print(counts.to_string())

print("\n" + "="*60)
print("✓ OK = téléchargé automatiquement")
print("⚠️  MANUAL = inscription/format spécial requis")
print("✗ FAIL = URL invalide ou serveur HS")
print("="*60)

print(f"\nTotal sources testées : {len(results)}")
print(f"  ✓ OK     : {(df_report['Status']=='OK').sum()}")
print(f"  ⚠️  MANUAL: {(df_report['Status']=='MANUAL').sum()}")
print(f"  ✗ FAIL   : {(df_report['Status']=='FAIL').sum()}")

# Sauvegarder le rapport
df_report.to_csv("couche1_planete/reports/dataset_download_report.csv", index=False)
print(f"\n→ Rapport : couche1_planete/reports/dataset_download_report.csv")
