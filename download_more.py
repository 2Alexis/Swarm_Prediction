"""
download_more.py — Téléchargement massif de nouveaux datasets gratuits :

  * 35 indicateurs World Bank supplémentaires (API directe, pas d'auth)
  * 12 datasets OWID supplémentaires (food, schooling, air pollution, meat...)
  * FAO Food Balance Sheets bulk (gratuit, URL directe)
"""
import os, sys, io, urllib.request, urllib.parse, json
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
OUT = "data/cleaned"
os.makedirs(OUT, exist_ok=True)

def fetch_json(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="ignore"))

def fetch_text(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")

# ── 1. WORLD BANK API ──────────────────────────────────────────────────────
WB_INDICATORS = {
    # Agriculture & terre
    "AG.LND.FRST.ZS":     "wb_forest_area_pct",
    "AG.LND.ARBL.HA.PC":  "wb_arable_per_capita",
    "AG.LND.IRIG.AG.ZS":  "wb_irrigated_pct",
    "AG.PRD.FOOD.XD":     "wb_food_production_index",
    "AG.YLD.CREL.KG":     "wb_cereal_yield",
    "AG.CON.FERT.ZS":     "wb_fertilizer_consumption",
    # Économie & secteurs
    "NV.AGR.TOTL.ZS":     "wb_agri_value_pct_gdp",
    "NV.IND.MANF.ZS":     "wb_manuf_value_pct_gdp",
    "NV.SRV.TOTL.ZS":     "wb_services_value_pct_gdp",
    "NY.GDP.PETR.RT.ZS":  "wb_oil_rents_pct_gdp",
    "NY.GDP.MINR.RT.ZS":  "wb_mineral_rents_pct_gdp",
    "NY.GDP.NGAS.RT.ZS":  "wb_natgas_rents_pct_gdp",
    "NY.GDP.COAL.RT.ZS":  "wb_coal_rents_pct_gdp",
    "NY.GDP.FRST.RT.ZS":  "wb_forest_rents_pct_gdp",
    # Santé & alimentation
    "EN.ATM.PM25.MC.M3":  "wb_pm25_annual",
    "SH.STA.STNT.ZS":     "wb_stunting_pct",
    "SH.STA.WAST.ZS":     "wb_wasting_pct",
    "SH.STA.OWGH.ZS":     "wb_overweight_pct",
    "SH.H2O.SMDW.ZS":     "wb_safe_water_pct",
    "SH.STA.SMSS.ZS":     "wb_sanitation_pct",
    "SH.MED.PHYS.ZS":     "wb_physicians_per_1000",
    "SH.DTH.IMRT":        "wb_infant_deaths_total",
    "SP.DYN.AMRT.MA":     "wb_adult_mortality_male",
    "SP.DYN.AMRT.FE":     "wb_adult_mortality_female",
    # Éducation
    "SE.PRM.ENRR":        "wb_school_primary_enrollment",
    "SE.SEC.ENRR":        "wb_school_secondary_enrollment",
    "SE.ADT.LITR.ZS":     "wb_adult_literacy_pct",
    # Emploi
    "SL.AGR.EMPL.ZS":     "wb_employ_agri_pct",
    "SL.IND.EMPL.ZS":     "wb_employ_industry_pct",
    "SL.SRV.EMPL.ZS":     "wb_employ_services_pct",
    # Ressources
    "ER.H2O.INTR.PC":     "wb_freshwater_internal_per_cap",
    "ER.H2O.FWTL.K3":     "wb_freshwater_withdraw_total",
    # CO2 & énergie détaillée
    "EN.ATM.CO2E.PC":     "wb_co2_per_capita",
    "EG.USE.PCAP.KG.OE":  "wb_energy_use_per_cap",
    # Tech
    "IT.NET.BBND.P2":     "wb_broadband_per_100",
    # Pop additionnel
    "SP.POP.DPND":        "wb_dependency_ratio",
    "SP.POP.DPND.YG":     "wb_dependency_young",
    "SP.POP.DPND.OL":     "wb_dependency_old",
    "SP.POP.65UP.TO.ZS":  "wb_pop_65_plus_pct",
    "SP.POP.0014.TO.ZS":  "wb_pop_under14_pct",
    "SP.URB.GROW":        "wb_urban_growth",
    "SM.POP.REFG":        "wb_refugees_origin",
    "SM.POP.REFG.OR":     "wb_refugees_destination",
}

def wb_fetch(ind):
    """Récupère un indicateur WB pour TOUS les pays + années via l'API REST."""
    url = f"https://api.worldbank.org/v2/country/all/indicator/{ind}?format=json&per_page=20000&date=1960:2024"
    try:
        data = fetch_json(url, timeout=120)
        if len(data) < 2 or not data[1]:
            return None
        rows = []
        for r in data[1]:
            if r.get("value") is None: continue
            try:
                rows.append({
                    "Pays": r["country"]["value"],
                    "ISO3": r["countryiso3code"],
                    "Annee": int(r["date"]),
                    "Valeur": float(r["value"]),
                })
            except Exception:
                continue
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"   ✗ {ind}: {e}")
        return None

print("[1] World Bank — 35+ indicateurs supplémentaires…")
ok = 0
for code, colname in WB_INDICATORS.items():
    df = wb_fetch(code)
    if df is None or df.empty:
        print(f"   ✗ {code}")
        continue
    out = df[["Pays", "Annee", "Valeur"]]
    out.to_csv(f"{OUT}/{colname}.csv", index=False)
    print(f"   {code:25s} -> {colname:35s} {len(out):,} lignes")
    ok += 1
print(f"   → {ok}/{len(WB_INDICATORS)} OK")


# ── 2. OWID grapher CSVs supplémentaires ────────────────────────────────────
OWID_EXTRAS = [
    ("mean-years-of-schooling-long-run",         "owid_schooling_years"),
    ("number-of-deaths-from-air-pollution",      "owid_deaths_air_pollution"),
    ("share-of-deaths-air-pollution",            "owid_share_deaths_air_pollution"),
    ("per-capita-meat-consumption-by-type-kilograms-per-year", "owid_meat_consumption"),
    ("per-capita-fish-and-seafood-consumption-fao", "owid_fish_consumption"),
    ("share-of-the-population-with-undernourishment", "owid_undernourishment_pct"),
    ("per-capita-energy-from-fossil-fuels-nuclear-and-renewables", "owid_energy_breakdown"),
    ("annual-co-emissions-by-region",            "owid_co_emissions"),
    ("annual-methane-emissions-by-region",       "owid_methane_emissions"),
    ("annual-no2-emissions-by-region",           "owid_nox_emissions"),
    ("cereal-production",                        "owid_cereal_production"),
    ("agricultural-area",                        "owid_agri_area"),
    ("share-of-population-with-access-to-electricity", "owid_electricity_access_extra"),
    ("share-of-the-population-using-the-internet", "owid_internet_extra"),
    ("number-of-people-without-access-to-clean-water", "owid_no_clean_water"),
    ("primary-energy-consumption-per-capita",    "owid_energy_per_cap"),
    ("share-of-population-living-in-extreme-poverty", "owid_extreme_poverty"),
    ("life-expectancy-vs-gdp-per-capita",        None),  # composé
]

def owid_csv(slug):
    url = f"https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=true"
    txt = fetch_text(url, timeout=120)
    df = pd.read_csv(io.StringIO(txt))
    df.columns = [c.strip() for c in df.columns]
    ent = "Entity" if "Entity" in df.columns else df.columns[0]
    yr  = "Year"   if "Year"   in df.columns else df.columns[2]
    val_cols = [c for c in df.columns if c not in (ent, "Code", yr)]
    num_cols = [c for c in val_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return None
    # Si plusieurs valeurs : on garde les top-3 colonnes numériques
    out = df[[ent, yr] + num_cols[:3]].rename(columns={ent: "Pays", yr: "Annee"})
    out = out.dropna(subset=["Pays", "Annee"])
    return out

print("\n[2] OWID extras…")
ok = 0
for slug, colname in OWID_EXTRAS:
    if not colname:
        continue
    try:
        out = owid_csv(slug)
        if out is None or out.empty: continue
        # Standardisation : 1 colonne valeur principale
        val_col = [c for c in out.columns if c not in ("Pays", "Annee")][0]
        out2 = out[["Pays", "Annee", val_col]].rename(columns={val_col: "Valeur"})
        out2.to_csv(f"{OUT}/{colname}.csv", index=False)
        print(f"   {slug:55s} -> {colname:38s} {len(out2):,} lignes")
        ok += 1
    except Exception as e:
        print(f"   ✗ {slug}: {e}")
print(f"   → {ok}/{len(OWID_EXTRAS)} OK")


# ── 3. FAO Food Balance Sheets (FBS) bulk via FAOSTAT direct ───────────────
# FAOSTAT bulk download : https://www.fao.org/faostat/en/#data/FBS
# Direct CSV : https://bulks-faostat.fao.org/production/FoodBalanceSheets_E_All_Data.zip
# Trop lourd (zip). On utilise l'API JSON pour les indicateurs principaux.
def fao_api(domain, indicator, area="all", item="all", year_from=1990, year_to=2024):
    base = "https://faostatservices.fao.org/api/v1/en/data"
    url = f"{base}/{domain}?area_cs=all&element={indicator}&year={year_from}-{year_to}&output_type=json&pretty=false&show_codes=true&page_size=200000"
    try:
        data = fetch_json(url, timeout=180)
        if "data" not in data: return None
        rows = []
        for r in data["data"]:
            rows.append({
                "Pays": r.get("Area"),
                "ISO3": r.get("Area Code (ISO3)"),
                "Element": r.get("Element"),
                "Item": r.get("Item"),
                "Annee": int(r.get("Year")) if r.get("Year") else None,
                "Valeur": float(r["Value"]) if r.get("Value") else None,
                "Unite": r.get("Unit"),
            })
        return pd.DataFrame(rows).dropna(subset=["Annee", "Valeur"])
    except Exception as e:
        print(f"   ✗ {domain}/{indicator}: {e}")
        return None

print("\n[3] FAO indicators via FAOSTAT API…")
# Note : le service FAOSTAT peut être lent ; on tente quelques cibles importantes
# Si ça échoue ou est trop lent on skippe
fao_targets = [
    # (domain, element_code, friendly_name)
    # FBS Food supply (kg/cap/yr) — element 645
    # Trade Index — TI
    # Investment - capital stock - CS
]
# On skippe FAO bulk pour cette itération : peut être ajouté avec /optional_loaders.py
print("   (FAO bulk skipped — gros volumes, à activer manuellement via FAOSTAT bulk-download)")


print("\n[DONE] Tous les downloads dispos terminés.")
