"""
fetch_missing_datasets.py — Tentative de récupération automatique des datasets manquants.

Pour chaque cible <0.5 du tableau, essai download direct via URL stable.
Note : beaucoup nécessitent inscription/API key. On marque MANUAL quand ça échoue.
"""
import os, sys, io, urllib.request, urllib.parse, json, time, zipfile
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pycountry
def iso3_to_iso2(c):
    try:
        co = pycountry.countries.get(alpha_3=c)
        return co.alpha_2 if co else None
    except: return None


def fetch_file(url, out, max_mb=300, timeout=120):
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64) AppleWebKit/537.36",
            "Accept": "*/*",
        })
        with urllib.request.urlopen(req, timeout=timeout) as r:
            cl = int(r.headers.get("Content-Length", 0))
            if cl > max_mb * 1e6: return False, f"too big {cl/1e6:.0f}MB"
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as f:
                while True:
                    chunk = r.read(1<<20)
                    if not chunk: break
                    f.write(chunk)
            sz = os.path.getsize(out)/1e6
            return True, f"{sz:.1f} MB"
    except Exception as e:
        return False, str(e)[:80]


def fetch_text(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


results = []
def report(category, name, status, detail, url=""):
    sym = {"OK":"✓","FAIL":"✗","MANUAL":"⚠️"}.get(status,"?")
    print(f"  {sym} {name:50s} {status:8s} {detail}")
    results.append({"Catégorie":category, "Nom":name, "Status":status, "Détail":detail, "URL":url})


print("══════════════════════════════════════════════════════════════")
print("📡 RÉCUPÉRATION DATASETS MANQUANTS — Cibles <0.5 R²")
print("══════════════════════════════════════════════════════════════\n")


# ════════════════════════════════════════════════════════════════════════
print("🥇 1. FAOSTAT — bulk downloads pour cibles ratées")
# ════════════════════════════════════════════════════════════════════════
faostat_targets = [
    # Élevage détaillé
    ("Production_LivestockPrimary",      "FAO_livestock_primary.zip", "élevage carcasses"),
    ("Production_LivestockProcessed",    "FAO_livestock_processed.zip", "élevage transformé"),
    ("Population_LiveStock",             "FAO_livestock_population.zip", "cheptel par pays"),
    # Cultures détaillées
    ("Production_Crops_Livestock",       "FAO_crops_livestock.zip", "cultures + élevage"),
    ("Indicators_AgriEnv",               "FAO_agri_env_indicators.zip", "agri-env indicateurs"),
    # Inputs détaillés
    ("Inputs_FertilizersProduct",        "FAO_fert_product.zip", "engrais par produit"),
    # Pêche
    ("Fisheries_GlobalProduction",       "FAO_fisheries_global.zip", "pêche détaillée"),
    # Forêt
    ("Forestry_Production_Trade",        "FAO_forestry_prod_trade.zip", "forêt prod+trade"),
    ("Forestry_E_All_Data",              "FAO_forestry_full.zip", "forêt bulk"),
    # Commerce
    ("Trade_DetailedTradeMatrix",        "FAO_trade_matrix.zip", "trade matrix"),
    # Emissions
    ("Environment_Emissions_intensities","FAO_emissions_intensities.zip", "intensités émissions"),
    ("Environment_Emissions_byCategory", "FAO_emissions_by_category.zip", "émissions catégories"),
]
for code, fname, desc in faostat_targets:
    url = f"https://bulks-faostat.fao.org/production/{code}_E_All_Data_(Normalized).zip"
    ok, det = fetch_file(url, f"data/raw/agriculture/{fname}", max_mb=200)
    report("Agriculture/FAO", desc, "OK" if ok else "FAIL", det, url)
    time.sleep(0.5)


# ════════════════════════════════════════════════════════════════════════
print("\n🥈 2. USGS NEIC Earthquakes — historique 1900-2025")
# ════════════════════════════════════════════════════════════════════════
# Format : https://earthquake.usgs.gov/fdsnws/event/1/query?...
# Limite 20000 events par requête. On découpe par tranches d'années.
print("   Téléchargement séismes magnitude >= 5 par décennies…")
all_eq = []
for start_y, end_y in [(1900,1929),(1930,1959),(1960,1979),(1980,1999),(2000,2009),(2010,2019),(2020,2025)]:
    url = (f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
           f"&starttime={start_y}-01-01&endtime={end_y}-12-31"
           f"&minmagnitude=5.0&orderby=time&limit=20000")
    try:
        txt = fetch_text(url, timeout=180)
        from io import StringIO
        df = pd.read_csv(StringIO(txt))
        print(f"   {start_y}-{end_y}: {len(df)} séismes M≥5")
        all_eq.append(df)
    except Exception as e:
        print(f"   ✗ {start_y}-{end_y}: {str(e)[:60]}")
    time.sleep(0.5)

if all_eq:
    eq_full = pd.concat(all_eq, ignore_index=True)
    out = "data/raw/geologie/usgs_neic_earthquakes_M5plus_1900-2025.csv"
    eq_full.to_csv(out, index=False)
    report("Géologie", "USGS NEIC M5+ historical", "OK",
           f"{len(eq_full):,} séismes → {out}", "")


# ════════════════════════════════════════════════════════════════════════
print("\n🥉 3. WDPA Protected Planet — Aires protégées")
# ════════════════════════════════════════════════════════════════════════
# WDPA = World Database on Protected Areas
# CSV téléchargeable directement par pays via API : https://api.protectedplanet.net/
# Mais besoin de token. Sans token, on peut tenter les bulk downloads :
wdpa_urls = [
    # CSV bulk par catégorie
    "https://wcmc.io/wdpa_country_summary",  # CSV summary
    "https://d1gam3xoknrgr2.cloudfront.net/current/WDPA_Country_Summary.csv",
]
for u in wdpa_urls:
    ok, det = fetch_file(u, "data/raw/sol_ecologie/wdpa_country_summary.csv", max_mb=100)
    if ok:
        report("Sol/Écologie", "WDPA Country Summary", "OK", det, u)
        break
else:
    report("Sol/Écologie", "WDPA Protected Planet", "MANUAL",
           "API requiert token — inscription gratuite",
           "https://www.protectedplanet.net/en/thematic-areas/wdpa")


# ════════════════════════════════════════════════════════════════════════
print("\n🌳 4. FAO FRA 2025 — Forest Resources Assessment")
# ════════════════════════════════════════════════════════════════════════
fra_urls = [
    "https://fra-data.fao.org/api/v2/country/_all/_all/FRA2025/_excel",
    "https://fra-data.fao.org/api/data/FRA2025/country/csv",
]
for u in fra_urls:
    ok, det = fetch_file(u, "data/raw/sol_ecologie/fao_fra2025_data.zip", max_mb=100)
    if ok:
        report("Sol/Écologie", "FAO FRA 2025", "OK", det, u)
        break
else:
    report("Sol/Écologie", "FAO FRA 2025", "MANUAL",
           "Interface complexe (par pays)",
           "https://fra-data.fao.org/")


# ════════════════════════════════════════════════════════════════════════
print("\n🐄 5. OECD-FAO Agricultural Outlook 2024-2033")
# ════════════════════════════════════════════════════════════════════════
oecd_urls = [
    # SDMX endpoint OECD
    "https://stats.oecd.org/SDMX-JSON/data/HIGH_AGLINK_2024/all/all/?contentType=csv",
    "https://sdmx.oecd.org/public/rest/data/OECD.TAD.ATM,DSD_AGR@DF_OUTLOOK_2024_2033,1.0/...A/?dimensionAtObservation=AllDimensions&format=csv",
    # Direct file (might not exist)
    "https://www.agri-outlook.org/data/Aglink_2024-2033_full_dataset.xlsx",
]
for u in oecd_urls:
    ok, det = fetch_file(u, "data/raw/elevage/oecd_fao_outlook_2024.csv", max_mb=200)
    if ok:
        report("Élevage", "OECD-FAO Outlook 2024-2033", "OK", det, u)
        break
else:
    report("Élevage", "OECD-FAO Outlook", "MANUAL",
           "Bulk download nécessite navigation interactive",
           "https://www.agri-outlook.org/data/")


# ════════════════════════════════════════════════════════════════════════
print("\n🐕 6. WOAH (OIE) WAHIS Veterinary Drugs / Animal Vaccinations")
# ════════════════════════════════════════════════════════════════════════
# JECFA toolbox = pour évaluer résidus médicamenteux, pas dataset par pays
# WOAH/OIE veterinary stats : nécessite compte
# Le seul bulk public : "World animal health information system" via API
report("Élevage", "JECFA Veterinary Drugs Toolbox", "MANUAL",
       "Toolkit d'évaluation, pas dataset de couverture vaccinale animale",
       "https://www.fao.org/jefca-toolbox-veterinary-drugs-assessment/")
report("Élevage", "WOAH WAHIS Vaccination data", "MANUAL",
       "Nécessite compte OIE/WAHIS",
       "https://wahis.woah.org/")


# ════════════════════════════════════════════════════════════════════════
print("\n🦋 7. GBIF Occurrences — biodiversité par pays")
# ════════════════════════════════════════════════════════════════════════
# GBIF Statistics API
gbif_url = "https://api.gbif.org/v1/occurrence/search?facet=country&limit=0&facetLimit=300"
try:
    txt = fetch_text(gbif_url, timeout=60)
    data = json.loads(txt)
    facets = data.get("facets", [])
    for f in facets:
        if f.get("field") == "COUNTRY":
            rows = []
            for c in f.get("counts", []):
                iso2 = c["name"]  # ISO2
                count = c["count"]
                rows.append({"ISO": iso2, "gbif_occurrences": count})
            df = pd.DataFrame(rows)
            df["Annee"] = 2024
            os.makedirs("data/cleaned/sol_ecologie", exist_ok=True)
            df.to_csv("data/cleaned/sol_ecologie/gbif_occurrences_by_country.csv", index=False)
            report("Sol/Écologie", "GBIF Occurrences by country", "OK",
                   f"{len(df)} pays → cleaned", gbif_url)
            break
except Exception as e:
    report("Sol/Écologie", "GBIF API", "FAIL", str(e)[:80], gbif_url)


# ════════════════════════════════════════════════════════════════════════
print("\n🍎 8. USDA NASS — Apples / Strawberries / Specialty crops")
# ════════════════════════════════════════════════════════════════════════
# USDA NASS API : https://quickstats.nass.usda.gov/api/
# Sans API key, accès limité. Try public stats
report("Agriculture", "USDA NASS Quick Stats", "MANUAL",
       "API key requise (gratuit)",
       "https://quickstats.nass.usda.gov/api/")


# ════════════════════════════════════════════════════════════════════════
print("\n🌍 9. EuroStat Agriculture")
# ════════════════════════════════════════════════════════════════════════
# EuroStat fournit données publiques via SDMX
euro_endpoints = [
    # Crop production (apro_acs_a)
    ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/apro_cpsh1/?format=TSV",
      "eurostat_apro_cpsh1.tsv", "Crop production"),
    # Livestock (apro_mt_lscatl + lspig + lsgoat + lsovin)
    ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/apro_mt_lscatl/?format=TSV",
      "eurostat_cattle_livestock.tsv", "Cattle livestock"),
    ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/apro_mt_lspig/?format=TSV",
      "eurostat_pig_livestock.tsv", "Pig livestock"),
    ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/apro_mt_lsgoat/?format=TSV",
      "eurostat_goat_livestock.tsv", "Goat livestock"),
    ("https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/apro_mt_lsovin/?format=TSV",
      "eurostat_sheep_livestock.tsv", "Sheep livestock"),
]
os.makedirs("data/raw/elevage/eurostat", exist_ok=True)
for url, fname, desc in euro_endpoints:
    out = f"data/raw/elevage/eurostat/{fname}"
    ok, det = fetch_file(url, out, max_mb=50)
    report("EuroStat", desc, "OK" if ok else "FAIL", det, url)
    time.sleep(0.3)


# ════════════════════════════════════════════════════════════════════════
# BILAN
# ════════════════════════════════════════════════════════════════════════
print("\n\n══════════════════════════════════════════════════════════════")
print("📊 BILAN — RÉCUPÉRATION DATASETS MANQUANTS")
print("══════════════════════════════════════════════════════════════")

df_r = pd.DataFrame(results)
counts = df_r.groupby("Status").size()
print(f"\n  Total testés : {len(df_r)}")
print(f"  ✓ OK      : {counts.get('OK', 0)}")
print(f"  ⚠️ MANUAL : {counts.get('MANUAL', 0)}")
print(f"  ✗ FAIL    : {counts.get('FAIL', 0)}")

print("\n  ✓ Téléchargés avec succès :")
for r in results:
    if r["Status"] == "OK":
        print(f"     • {r['Nom']}  {r['Détail']}")

df_r.to_csv("couche1_planete/reports/missing_datasets_fetch_report.csv", index=False)
print(f"\n→ Rapport : couche1_planete/reports/missing_datasets_fetch_report.csv")
