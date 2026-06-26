"""
organize_and_process.py — Réorganise data/raw et data/cleaned par catégorie
+ traite les nouveaux fichiers (IRENA, Ember, CMM, WAHIS, SE4ALL).

Catégories :
  - atmosphere     : climat dynamique, émissions, qualité air, ozone
  - hydrologie     : eau, mer, glace, hydrologie
  - sol_ecologie   : sol, biodiversité, forêt, feux, environnement
  - agriculture    : cultures végétales, intrants, suitabilité
  - elevage        : animaux, viande, lait, œufs
  - peche          : poisson, aquaculture
  - energie        : production, consommation, renouvelables, fossile
  - geologie       : minéraux, séismes, volcans
  - demographie    : population, santé, vie, fécondité
  - economie       : PIB, commerce, valeur production
  - climat_indices : ENSO, NAO, AMO, PDO, indices
  - worldclim      : WorldClim BIO 1-19, T, P, etc.
  - shared         : dataset_final_*, country_clusters, EcoCrop
  - misc           : utilitaires, autres
"""
import os, sys, io, shutil, glob, zipfile
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

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
# CATÉGORISATION : mapping pattern → catégorie
# ════════════════════════════════════════════════════════════════════════
def categorize(filename):
    """Renvoie la catégorie pour un fichier."""
    n = filename.lower()
    # Atmosphere
    if any(k in n for k in ["edgar_","ch4","co2","n2o","f-gas","atmosphere_climat",
                              "atmosphere_extras","emissions","co2_","methane","ozone",
                              "climate_watch","temperature","pm25","mean_temperature",
                              "variation_temperature","global_co2","berkeley_earth",
                              "owid_co_per","owid_co2","owid_methane","owid_n2o",
                              "owid_avg_monthly_temp","owid_temp_anomaly","worldclim_bio_extra",
                              "download_cmm","mean_temperature","variation_temperature"]):
        return "atmosphere"
    # Hydrologie
    if any(k in n for k in ["aqueduct","aquastat","hydro","water","sea_level","ocean",
                              "psmsl","cobe","grace","wri_","groundwater","tide",
                              "wb_freshwater","wb_safe_water","wb_sanitation",
                              "owid_water"]):
        return "hydrologie"
    # Sol & Écologie
    if any(k in n for k in ["forest","tree_cover","ecologie","biodiv","iucn","epi_",
                              "firms","fire","ndvi","modis","viirs","gbif","biomass",
                              "pedologie","sols","soil","owid_forest","wb_forest",
                              "bilan_nutritif","wood_density","deforest",
                              "ecologie_biomasse"]):
        return "sol_ecologie"
    # Agriculture
    if any(k in n for k in ["spam","crop","yield","fertili","pesticid","land_use_input",
                              "landuse","intrants","arable","agricultur","cereal",
                              "vegetable","fruit","plant","sugar","cotton","tobacco",
                              "owid_agri","owid_cereal","owid_food","wb_agri","wb_cereal",
                              "wb_arable","wb_food_production","wb_fertili","wb_irrigat",
                              "production_cultures","ecocrop","gaez","crop_calendar"]):
        return "agriculture"
    # Élevage
    if any(k in n for k in ["livestock","meat","milk","egg","cattle","poultry",
                              "sheep","pig","glw","glw4","production_animaux",
                              "owid_meat","owid_milk","owid_egg","wahis","wahis_",
                              "donnees quantitatives","wb_employ_agri"]):
        return "elevage"
    # Pêche
    if any(k in n for k in ["fish","aqua_culture","aquaculture","fishery","peche",
                              "marine_protected","fishstat","globalproductionfish"]):
        return "peche"
    # Énergie
    if any(k in n for k in ["irena","ember","electric","energy","power","grid","solar",
                              "wind","hydroelec","coal","oil","gas_prod","fossil",
                              "renewable","ireccap","relecgen","reshare","heatgen",
                              "r-elec","r_elec","monthly_capacity","yearly_full_release",
                              "se4all","sustainable_energy","targets_download",
                              "wb_electric","wb_energy","wb_renewable","wb_coal_rents",
                              "wb_oil_rents","wb_natgas_rents","wb_mineral_rents",
                              "wb_forest_rents","ember_yearly","energie_extras",
                              "global_power_plants"]):
        return "energie"
    # Géologie
    if any(k in n for k in ["mrds","mineral","earthquake","seismic","volcano","volcan",
                              "tectonic","geolog","iron_mine","oil_gas","coal_mine"]):
        return "geologie"
    # Démographie
    if any(k in n for k in ["population","mortality","fertility","birth","death",
                              "child_mort","life_exp","migrant","hunger","stunting",
                              "wasting","owid_birth","owid_death","owid_population",
                              "owid_child_mortality","owid_pop","wb_population","wb_birth",
                              "wb_death","wb_child_mortality","wb_life_exp","wb_dependency",
                              "wb_pop_","wb_stunting","wb_wasting","wb_overweight",
                              "wb_adult_mortality","wb_infant","wb_school","wb_adult_literacy",
                              "wb_employ_","wb_unemployment","wb_urban","wb_internet",
                              "wb_mobile","wb_broadband","wb_hospital","wb_physic",
                              "wb_health","wb_hiv","wb_malaria","wb_deaths_communic",
                              "global_hunger","who_pm25","who_ambient"]):
        return "demographie"
    # Economie
    if any(k in n for k in ["gdp","econom","trade","commerce","finance","debt",
                              "inflation","gini","poverty","value_of_production",
                              "value_production","wb_gdp","wb_trade","wb_inflation",
                              "wb_gini","wb_poverty","wb_public_debt","wb_rd_expenditure",
                              "wb_manuf","wb_services","wb_agri_value","fao_value_of",
                              "fao_value_production"]):
        return "economie"
    # Climat indices
    if any(k in n for k in ["climate_index","enso","nao","amo","pdo","soi","oni",
                              "ao_lag","ao_index"]):
        return "climat_indices"
    # WorldClim
    if "worldclim" in n or n.startswith(("bio","prec","tmax","tmin","tvag","wind_","solrad","vapr","elev")):
        return "worldclim"
    # Shared (dataset finals, country clusters, etc.)
    if n.startswith(("dataset_final","country_","ecocrop","datset_final")):
        return "shared"
    # Topographie/géo
    if any(k in n for k in ["topographie","relief","elevation","slope","centroid",
                              "physical_features"]):
        return "geographie"
    # Misc / default
    return "misc"


# ════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 : RÉORGANISATION
# ════════════════════════════════════════════════════════════════════════
print("\n══════ ÉTAPE 1 — RÉORGANISATION ══════\n")

CATEGORIES = ["atmosphere","hydrologie","sol_ecologie","agriculture","elevage",
              "peche","energie","geologie","demographie","economie",
              "climat_indices","worldclim","geographie","shared","misc"]

def organize_dir(base_dir, dry_run=False):
    """Réorganise tous les fichiers d'un dossier par catégorie."""
    print(f"\n→ Réorganisation {base_dir}/")
    moved_count = {c: 0 for c in CATEGORIES}
    skipped = 0

    for fname in os.listdir(base_dir):
        src = os.path.join(base_dir, fname)
        # Skip si c'est déjà un dossier de catégorie
        if os.path.isdir(src) and fname in CATEGORIES:
            continue
        # Skip __pycache__ et fichiers cachés
        if fname.startswith("__") or fname.startswith("."):
            continue

        cat = categorize(fname)
        dst_dir = os.path.join(base_dir, cat)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, fname)

        if not dry_run:
            try:
                if os.path.isdir(src):
                    # Si c'est un sous-dossier (e.g., spam2020_v2r2_yield, epi_2024, etc.)
                    # On déplace tout le sous-dossier
                    if os.path.exists(dst):
                        # déjà existe : skip
                        skipped += 1
                        continue
                    shutil.move(src, dst)
                else:
                    shutil.move(src, dst)
                moved_count[cat] += 1
            except Exception as e:
                print(f"   ⚠️ {fname}: {str(e)[:80]}")

    print(f"   Bilan par catégorie :")
    for c, n in moved_count.items():
        if n > 0: print(f"     {c:20s} {n:4d} items")
    if skipped > 0: print(f"   Skipped (déjà existant) : {skipped}")
    return moved_count


# Réorganisation
organize_dir(RAW)
organize_dir(CLN)


# ════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 : TRAITEMENT DES NOUVEAUX FICHIERS
# ════════════════════════════════════════════════════════════════════════
print("\n\n══════ ÉTAPE 2 — TRAITEMENT DES NOUVEAUX FICHIERS ══════\n")

# ── 2.1 IRENA ELECCAP (Electricity Capacity by Region/Tech) ───────────────
print("[A] IRENA — Capacity, Generation, Renewable Share, Heat…")
def parse_irena_hierarchical(filepath, value_name):
    """IRENA exports CSV avec hiérarchie textuelle : Region | Tech | ... | Year | Value."""
    if not os.path.exists(filepath): return None
    try:
        # Read RAW lines (les colonnes sont compactées en une seule)
        with open(filepath, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        # La 1ère ligne est le titre. La 2ème les noms de colonnes mais imprimés en hiérarchie
        # Approach : ligne par ligne, splitter par double-espace
        records = []
        current = {}
        # Skip premier titre (ligne 0)
        for ln in lines[1:]:
            ln_strip = ln.rstrip("\n").rstrip()
            if not ln_strip: continue
            # On split par chunks d'espaces 2+
            import re
            parts = re.split(r"\s{2,}", ln_strip.strip())
            if len(parts) >= 2:
                try:
                    val = float(parts[-1].replace(",",""))
                    # Avant-dernier = Year
                    year = parts[-2]
                    if year.isdigit() and 1900 <= int(year) <= 2030:
                        records.append({"Year": int(year), "Value": val,
                                          "Hierarchy": " | ".join(parts[:-2])})
                except ValueError:
                    pass
        return pd.DataFrame(records)
    except Exception as e:
        print(f"   ⚠️ {filepath}: {str(e)[:80]}")
        return None


for fname, out_name in [
    ("R-ELECCAP_20260624-114520.csv", "irena_electricity_capacity"),
    ("R-ELECGEN_20260624-114636.csv", "irena_electricity_generation"),
    ("RESHARE_20260624-114707.csv",   "irena_renewable_share"),
    ("HEATGEN_20260624-114228.csv",   "irena_heat_generation"),
]:
    src = f"{RAW}/energie/{fname}"
    if not os.path.exists(src):
        # Maybe still in root
        src = f"{RAW}/{fname}"
    if not os.path.exists(src):
        print(f"   ✗ {fname} absent")
        continue
    df = parse_irena_hierarchical(src, out_name)
    if df is not None and len(df) > 0:
        # Extraire pays/région depuis Hierarchy : "World | Total renewable... | OnGrid"
        # Pour simplifier : la 1ère partie = Region/Country, le reste = technologie
        df["Pays"] = df["Hierarchy"].str.split(" | ").str[0]
        # Filtrer pays valides (pas "World", "Africa", etc. agrégats)
        non_country = {"World","Africa","Asia","Europe","Oceania","North America",
                        "South America","European Union","OECD","Non-OECD"}
        df_country = df[~df["Pays"].isin(non_country)]
        df_country["ISO"] = df_country["Pays"].apply(name_to_iso2)
        df_country = df_country.dropna(subset=["ISO"])
        if len(df_country) > 0:
            # Agréger par ISO/Year (somme sur techs)
            agg = df_country.groupby(["ISO","Year"])["Value"].sum().reset_index()
            agg = agg.rename(columns={"Year":"Annee", "Value": out_name + "_total"})
            agg.to_csv(f"{CLN}/energie/{out_name}.csv", index=False)
            print(f"   ✓ {out_name}: {len(agg)} lignes → {CLN}/energie/{out_name}.csv")
        else:
            print(f"   ⚠️ {fname}: aucun pays mappé")


# ── 2.2 CMM (Coal Mine Methane) ──────────────────────────────────────────
print("\n[B] Coal Mine Methane (CMM Emissions + Satellite)…")
for sheet_name, out_name in [
    ("emissions", "cmm_emissions_ch4"),
]:
    src = f"{RAW}/atmosphere/download_cmm_emissions.xlsx"
    if not os.path.exists(src):
        src = f"{RAW}/download_cmm_emissions.xlsx"
    if os.path.exists(src):
        try:
            df = pd.read_excel(src, sheet_name=sheet_name, engine="openpyxl")
            print(f"   {sheet_name}: {df.shape}, cols={list(df.columns)[:8]}")
            iso_col = next((c for c in df.columns if "COUNTRY_CODE" in c or "ISO" in c.upper()), None)
            year_col = next((c for c in df.columns if c.upper() == "YEAR"), None)
            val_col = next((c for c in df.columns if "EMISSIONS_CH4" in c.upper()), None)
            if iso_col and year_col and val_col:
                df["ISO"] = df[iso_col].apply(lambda c: iso3_to_iso2(c) if isinstance(c,str) and len(c)==3 else None)
                df = df.dropna(subset=["ISO"])
                agg = df.groupby(["ISO", year_col]).agg(
                    cmm_ch4_kt=(val_col, "sum")
                ).reset_index().rename(columns={year_col:"Annee"})
                agg.to_csv(f"{CLN}/atmosphere/{out_name}.csv", index=False)
                print(f"   ✓ → {CLN}/atmosphere/{out_name}.csv ({len(agg)} lignes)")
        except Exception as e:
            print(f"   ⚠️ {str(e)[:100]}")


# ── 2.3 WAHIS (Données quantitatives) ────────────────────────────────────
print("\n[C] WAHIS Animal Disease Outbreaks…")
for src in [f"{RAW}/elevage/Données quantitatives 2026-06-24.csv",
             f"{RAW}/Données quantitatives 2026-06-24.csv"]:
    if not os.path.exists(src): continue
    try:
        df = pd.read_csv(src, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(src, low_memory=False, encoding="latin-1")
    print(f"   {df.shape}, cols={list(df.columns)[:10]}")
    df["ISO"] = df["Pays"].apply(name_to_iso2)
    df = df.dropna(subset=["ISO","Année"])
    df["Année"] = pd.to_numeric(df["Année"], errors="coerce")
    df = df.dropna(subset=["Année"])
    df["Année"] = df["Année"].astype(int)
    # Agg : nombre d'outbreaks + cas + morts par pays/an
    num_cols = ["Cas","Mis à mort et éliminés","Morts","Vaccinés","Sensibles"]
    num_cols = [c for c in num_cols if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    agg = df.groupby(["ISO","Année"]).agg(
        wahis_outbreaks_total=("Outbreak_id", "nunique"),
        wahis_diseases_unique=("Maladie", "nunique"),
        wahis_cases=("Cas", "sum"),
        wahis_deaths=("Morts", "sum"),
    ).reset_index().rename(columns={"Année":"Annee"})
    agg.to_csv(f"{CLN}/elevage/wahis_animal_disease.csv", index=False)
    print(f"   ✓ → {CLN}/elevage/wahis_animal_disease.csv ({len(agg)} lignes)")
    break


# ── 2.4 IRENA targets_download.xlsx ──────────────────────────────────────
print("\n[D] IRENA Renewable Targets…")
for src in [f"{RAW}/energie/targets_download.xlsx", f"{RAW}/targets_download.xlsx"]:
    if not os.path.exists(src): continue
    try:
        df = pd.read_excel(src, sheet_name="raw_data_long", engine="openpyxl")
        print(f"   {df.shape}, cols={list(df.columns)[:10]}")
        df["ISO"] = df["COUNTRY_CODE"].apply(lambda c: iso3_to_iso2(c) if isinstance(c,str) and len(c)==3 else None)
        df = df.dropna(subset=["ISO"])
        # Agréger : nombre de cibles et année moyenne cible
        agg = df.groupby("ISO").agg(
            irena_target_year_avg=("TARGET_YEAR", "mean"),
            irena_target_n=("VALUE", "count"),
            irena_target_max_value=("VALUE", "max"),
        ).reset_index()
        # Année courante = 2024 (dataset téléchargé en 2024+)
        agg["Annee"] = 2024
        agg.to_csv(f"{CLN}/energie/irena_renewable_targets.csv", index=False)
        print(f"   ✓ → {CLN}/energie/irena_renewable_targets.csv ({len(agg)} pays)")
    except Exception as e:
        print(f"   ⚠️ {str(e)[:100]}")
    break


# ── 2.5 Ember Monthly Wind/Solar ─────────────────────────────────────────
print("\n[E] Ember Monthly Wind/Solar Capacity…")
for src in [f"{RAW}/energie/monthly_capacity_wind_solar_public_release_file.csv",
             f"{RAW}/monthly_capacity_wind_solar_public_release_file.csv"]:
    if not os.path.exists(src): continue
    df = pd.read_csv(src, low_memory=False)
    print(f"   {df.shape}, cols={list(df.columns)[:10]}")
    df["ISO"] = df["ISO 3 Code"].apply(lambda c: iso3_to_iso2(c) if isinstance(c,str) and len(c)==3 else None)
    df = df.dropna(subset=["ISO"])
    # Agréger par année (annualiser le mensuel)
    df["Capacity additions (year-to-date)"] = pd.to_numeric(
        df["Capacity additions (year-to-date)"], errors="coerce")
    df["Installed Capacity"] = pd.to_numeric(df["Installed Capacity"], errors="coerce")
    # Prendre la dernière valeur Year-to-date par (ISO, Year, Source)
    agg = df.sort_values(["ISO","Year","Source","Month"]).groupby(["ISO","Year","Source"]).last().reset_index()
    piv = agg.pivot_table(index=["ISO","Year"], columns="Source",
                            values="Installed Capacity", aggfunc="last").reset_index()
    piv = piv.rename(columns={"Year":"Annee"})
    rename = {c: f"ember_capacity_{str(c).lower().replace(' ','_')[:25]}_mw"
               for c in piv.columns if c not in ["ISO","Annee"]}
    piv = piv.rename(columns=rename)
    piv.to_csv(f"{CLN}/energie/ember_monthly_capacity_yearend.csv", index=False)
    print(f"   ✓ → {CLN}/energie/ember_monthly_capacity_yearend.csv ({len(piv)} lignes)")
    break


# ── 2.6 SE4ALL zip (déjà téléchargé) ─────────────────────────────────────
print("\n[F] SE4ALL P_Data_Extract zip…")
for src in [f"{RAW}/energie/P_Data_Extract_From_Sustainable_Energy_for_All.zip",
             f"{RAW}/P_Data_Extract_From_Sustainable_Energy_for_All.zip"]:
    if not os.path.exists(src): continue
    try:
        with zipfile.ZipFile(src) as z:
            members = z.namelist()
            print(f"   {len(members)} files in zip: {members[:5]}")
            for m in members:
                if m.endswith(".csv"):
                    with z.open(m) as f:
                        df = pd.read_csv(f, low_memory=False)
                    print(f"   {m}: {df.shape}, cols={list(df.columns)[:10]}")
                    # Format WB Databank : Country Code, Country Name, Series Name, Series Code, Year columns
                    year_cols = [c for c in df.columns if isinstance(c,str)
                                  and "[YR" in c]
                    if year_cols:
                        # Long format
                        id_cols = [c for c in df.columns if c not in year_cols]
                        long = df.melt(id_vars=id_cols, value_vars=year_cols,
                                        var_name="year_col", value_name="value")
                        long["Annee"] = long["year_col"].str.extract(r"YR(\d{4})").astype(float)
                        long = long.dropna(subset=["Annee","value"])
                        long["Annee"] = long["Annee"].astype(int)
                        long["value"] = pd.to_numeric(long["value"], errors="coerce")
                        long = long.dropna(subset=["value"])
                        if "Country Code" in long.columns:
                            long["ISO"] = long["Country Code"].apply(
                                lambda c: iso3_to_iso2(c) if isinstance(c,str) and len(c)==3 else None)
                            long = long.dropna(subset=["ISO"])
                            piv = long.pivot_table(index=["ISO","Annee"], columns="Series Name",
                                                     values="value", aggfunc="mean").reset_index()
                            rename = {c: f"se4all_{str(c).lower().replace(' ','_').replace(',','')[:35]}"
                                       for c in piv.columns if c not in ["ISO","Annee"]}
                            piv = piv.rename(columns=rename)
                            piv.to_csv(f"{CLN}/energie/se4all_indicators.csv", index=False)
                            print(f"   ✓ → {CLN}/energie/se4all_indicators.csv ({len(piv)} lignes × {piv.shape[1]-2} indicateurs)")
                            break
    except Exception as e:
        print(f"   ⚠️ {str(e)[:100]}")
    break


# ── 2.7 GLW4 cattle TIF ──────────────────────────────────────────────────
print("\n[G] GLW4 Cattle TIF…")
for src in [f"{RAW}/elevage/GLW4-2020.D-DA.GLEAM3-ALL-LU.tif",
             f"{RAW}/GLW4-2020.D-DA.GLEAM3-ALL-LU.tif"]:
    if not os.path.exists(src): continue
    print(f"   ⚠️ GLW4 cattle = GeoTIFF raster, nécessite rasterio + shapefile pays")
    print(f"   ⚠️ Skip pour l'instant — à intégrer plus tard")
    break


print("\n\n══════════════════════════════════════════════════════════════")
print("📊 STRUCTURE FINALE")
print("══════════════════════════════════════════════════════════════")
for base in [RAW, CLN]:
    print(f"\n{base}/")
    for c in CATEGORIES:
        d = f"{base}/{c}"
        if os.path.isdir(d):
            files = os.listdir(d)
            n_csv = sum(1 for f in files if f.endswith(".csv"))
            n_zip = sum(1 for f in files if f.endswith(".zip"))
            n_dir = sum(1 for f in files if os.path.isdir(os.path.join(d,f)))
            n_other = len(files) - n_csv - n_zip - n_dir
            print(f"  {c:18s} : {n_csv:3d} csv | {n_zip:2d} zip | {n_dir:2d} dirs | {n_other:2d} autres")
