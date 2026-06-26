"""
process_all_new.py — Traite TOUS les nouveaux datasets téléchargés.

1. FAO FRA 2025 bulk download (130+ CSV forêt détaillée)
2. OECD-FAO Outlook 2023-2032 (carcasses, viandes)
3. PMP Toolkit (Plant Mitigation Pathways)
4. USGS NEIC séismes historiques 1900-2025 (recalcul)
5. GBIF (déjà clean, vérifier)
6. EuroStat livestock (cattle/pig/goat)
7. FAO Crops+Livestock bundle (34 MB)
8. FAO Fertilizers by Product
9. FAO Emissions Intensities

Crée des cibles + features ISO/Année par pays pour intégration v14.
"""
import os, sys, io, glob, zipfile, warnings
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
print("\n[1] 🌳 FAO FRA 2025 — bulk download (130+ CSV)…")
# ════════════════════════════════════════════════════════════════════════
fra_dir = f"{RAW}/bulk-download_fra_2025"
if os.path.isdir(fra_dir):
    # Les fichiers les plus importants pour notre pipeline
    KEY_FILES = {
        "FRA_Years_variables/1a_forestArea_2026-06-25.csv":          "fra_forest_area",
        "FRA_Years_variables/1a_landArea_2026-06-25.csv":            "fra_land_area",
        "FRA_Years_variables/1a_otherWoodedLand_2026-06-25.csv":     "fra_other_wooded_land",
        "FRA_Years_variables/1b_primary_2026-06-25.csv":              "fra_primary_forest",
        "FRA_Years_variables/1b_naturallyRegeneratingForest_2026-06-25.csv": "fra_natural_regen",
        "FRA_Years_variables/1b_plantedForest_2026-06-25.csv":        "fra_planted_forest",
        "FRA_Years_variables/2c_agb_total_2026-06-25.csv":            "fra_biomass_agb_total",
        "FRA_Years_variables/2d_carbon_agb_total_2026-06-25.csv":     "fra_carbon_agb_total",
        "FRA_Years_variables/3b_protected_2026-06-25.csv":            "fra_forest_protected",
        "Annual_variables/5b_fire_forest_2026-06-25.csv":             "fra_fire_forest_annual",
        "Annual_variables/5a_diseases_2026-06-25.csv":                "fra_diseases_annual",
        "Annual_variables/5a_insect_2026-06-25.csv":                  "fra_insect_annual",
        "Intervals_variables/1d_deforestation_2026-06-25.csv":        "fra_deforestation",
        "Intervals_variables/1d_afforestation_2026-06-25.csv":        "fra_afforestation",
        "Intervals_variables/1d_nat_exp_2026-06-25.csv":              "fra_natural_expansion",
    }

    all_fra = []
    for relpath, name in KEY_FILES.items():
        p = f"{fra_dir}/{relpath}"
        if not os.path.exists(p): continue
        try:
            df = pd.read_csv(p, low_memory=False)
            # Format FRA : iso3, year, value (souvent)
            iso_col = next((c for c in df.columns if c.lower() in ("iso3","countryiso3code","iso","country code")), None)
            yr_col = next((c for c in df.columns if c.lower() in ("year","fra_year")), None)
            val_col = next((c for c in df.columns if c.lower() in ("value","val")), None)
            if not (iso_col and yr_col and val_col):
                # Try first 3 cols
                if df.shape[1] >= 3:
                    iso_col, yr_col, val_col = df.columns[0], df.columns[1], df.columns[2]
            df["ISO"] = df[iso_col].apply(lambda c: iso3_to_iso2(c) if isinstance(c,str) and len(c)==3 else None)
            df = df.dropna(subset=["ISO"])
            df["Annee"] = pd.to_numeric(df[yr_col], errors="coerce")
            df = df.dropna(subset=["Annee"])
            df["Annee"] = df["Annee"].astype(int)
            df["value"] = pd.to_numeric(df[val_col], errors="coerce")
            df = df.dropna(subset=["value"])
            agg = df.groupby(["ISO","Annee"])["value"].mean().reset_index()
            agg = agg.rename(columns={"value": name})
            all_fra.append(agg)
            print(f"   + {name}: {len(agg)} obs")
        except Exception as e:
            print(f"   ⚠️ {name}: {str(e)[:80]}")

    # Outer merge sur (ISO, Annee)
    if all_fra:
        fra_merged = all_fra[0]
        for d in all_fra[1:]:
            fra_merged = fra_merged.merge(d, on=["ISO","Annee"], how="outer")
        os.makedirs(f"{CLN}/sol_ecologie", exist_ok=True)
        fra_merged.to_csv(f"{CLN}/sol_ecologie/fao_fra2025_forest_indicators.csv", index=False)
        print(f"   → {CLN}/sol_ecologie/fao_fra2025_forest_indicators.csv ({len(fra_merged)} lignes × {fra_merged.shape[1]-2} indic)")


# ════════════════════════════════════════════════════════════════════════
print("\n[2] 🐄 OECD-FAO Outlook 2023-2032…")
# ════════════════════════════════════════════════════════════════════════
oecd_path = f"{RAW}/OECD.TAD.ATM,DSD_AGR@DF_OUTLOOK_2023_2032,+OECD.A.CPC_0111....csv"
if not os.path.exists(oecd_path):
    cands = glob.glob(f"{RAW}/OECD*.csv") + glob.glob(f"{RAW}/elevage/OECD*.csv") + glob.glob(f"{RAW}/economie/OECD*.csv")
    oecd_path = cands[0] if cands else None
if oecd_path and os.path.exists(oecd_path):
    df = pd.read_csv(oecd_path, low_memory=False)
    print(f"   {df.shape}, cols: {list(df.columns)[:10]}")
    # Format OECD SDMX : multiples dimensions
    # Cherche country code, year, value, mesure
    iso_col = next((c for c in df.columns if "REF_AREA" in c.upper() or "COUNTRY" in c.upper()), None)
    yr_col = next((c for c in df.columns if c.upper() in ("TIME_PERIOD","YEAR","TIME")), None)
    val_col = next((c for c in df.columns if c.upper() in ("OBS_VALUE","VALUE","OBSERVATION_VALUE")), None)
    meas_col = next((c for c in df.columns if "MEASURE" in c.upper() or "COMMODITY" in c.upper()), None)
    print(f"   iso={iso_col}, year={yr_col}, val={val_col}, meas={meas_col}")
    if iso_col and yr_col and val_col:
        df[yr_col] = pd.to_numeric(df[yr_col], errors="coerce")
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=[yr_col, val_col])
        df["Annee"] = df[yr_col].astype(int)
        df["ISO"] = df[iso_col].apply(lambda c: iso3_to_iso2(c) if isinstance(c,str) and len(c)==3 else None)
        df = df.dropna(subset=["ISO"])
        # Pivot par measure si dispo
        if meas_col and df[meas_col].nunique() < 50:
            piv = df.pivot_table(index=["ISO","Annee"], columns=meas_col, values=val_col,
                                   aggfunc="mean").reset_index()
            rename = {c: f"oecd_{str(c).lower()[:25]}" for c in piv.columns if c not in ["ISO","Annee"]}
            piv = piv.rename(columns=rename)
        else:
            piv = df.groupby(["ISO","Annee"])[val_col].mean().reset_index().rename(
                columns={val_col: "oecd_outlook_value"})
        os.makedirs(f"{CLN}/elevage", exist_ok=True)
        piv.to_csv(f"{CLN}/elevage/oecd_fao_outlook_2023.csv", index=False)
        print(f"   → {CLN}/elevage/oecd_fao_outlook_2023.csv ({len(piv)} lignes)")


# ════════════════════════════════════════════════════════════════════════
print("\n[3] 🌱 forest_area summary 1990-2025…")
# ════════════════════════════════════════════════════════════════════════
forest_csv = f"{RAW}/forest_area,_1990_-_2025,_million_ha.csv"
if os.path.exists(forest_csv):
    try:
        df = pd.read_csv(forest_csv, low_memory=False)
        print(f"   {df.shape}, cols: {list(df.columns)[:10]}")
    except Exception as e:
        print(f"   ⚠️ {str(e)[:80]}")


# ════════════════════════════════════════════════════════════════════════
print("\n[4] 🌍 USGS NEIC séismes 95k — recalcul par pays/année…")
# ════════════════════════════════════════════════════════════════════════
eq_csv = f"{RAW}/geologie/usgs_neic_earthquakes_M5plus_1900-2025.csv"
if os.path.exists(eq_csv):
    df = pd.read_csv(eq_csv, low_memory=False)
    print(f"   {len(df):,} séismes M≥5")
    try:
        import reverse_geocoder as rg
        # Reverse-geocoding par batch
        coords = list(zip(df["latitude"].fillna(0), df["longitude"].fillna(0)))
        print(f"   Reverse-geocoding {len(coords):,} séismes…")
        results = rg.search(coords, mode=1, verbose=False)
        df["ISO"] = [r["cc"] for r in results]
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["Annee"] = df["time"].dt.year
        df = df.dropna(subset=["Annee"])
        df["Annee"] = df["Annee"].astype(int)
        df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
        agg = df.groupby(["ISO","Annee"]).agg(
            eq_count_m5=("mag","count"),
            eq_max_mag_hist=("mag","max"),
            eq_mean_mag_hist=("mag","mean"),
            eq_mag_ge6_hist=("mag", lambda s: (s>=6).sum()),
            eq_mag_ge7_hist=("mag", lambda s: (s>=7).sum()),
        ).reset_index()
        agg.to_csv(f"{CLN}/geologie/earthquakes_M5plus_by_country_year.csv", index=False)
        print(f"   → {CLN}/geologie/earthquakes_M5plus_by_country_year.csv ({len(agg)} lignes, {agg['ISO'].nunique()} pays)")
    except Exception as e:
        print(f"   ⚠️ {str(e)[:120]}")


# ════════════════════════════════════════════════════════════════════════
print("\n[5] 🐄 EuroStat livestock (cattle/pig/goat TSV)…")
# ════════════════════════════════════════════════════════════════════════
euro_dir = f"{RAW}/elevage/eurostat"
if os.path.isdir(euro_dir):
    euro_files = glob.glob(f"{euro_dir}/*.tsv")
    print(f"   {len(euro_files)} fichiers TSV")
    for f in euro_files:
        try:
            # EuroStat TSV : 1ère col = dims séparées par ',', puis colonnes années
            df = pd.read_csv(f, sep="\t", low_memory=False)
            print(f"   {os.path.basename(f)}: {df.shape}, col1 = {df.columns[0]}")
            # 1ère col contient des dimensions séparées par ',' + ISO2 en dernier
            first_col = df.columns[0]
            # Split la 1ère col
            df_split = df[first_col].str.split(",", expand=True)
            df_split.columns = [f"dim{i}" for i in range(df_split.shape[1])]
            # Le pays ISO est typiquement la dernière dimension
            iso_col_idx = df_split.shape[1] - 1
            df["ISO_eu"] = df_split.iloc[:, iso_col_idx]
            df["ISO"] = df["ISO_eu"].apply(lambda c: c if isinstance(c,str) and len(c)==2 else None)
            df = df.dropna(subset=["ISO"])
            # Année cols : noms numériques
            year_cols = [c for c in df.columns if c.strip().isdigit()]
            if not year_cols:
                year_cols = [c for c in df.columns if c.strip().split()[0].isdigit()]
            if year_cols:
                long = df.melt(id_vars=["ISO"], value_vars=year_cols,
                                var_name="Annee", value_name="value")
                long["Annee"] = pd.to_numeric(long["Annee"].astype(str).str.strip(), errors="coerce")
                long = long.dropna(subset=["Annee"])
                long["Annee"] = long["Annee"].astype(int)
                # Filtrer ":" (missing) et convertir
                long["value"] = pd.to_numeric(long["value"].astype(str).str.replace(":","").str.strip()
                                                .str.split(" ").str[0], errors="coerce")
                long = long.dropna(subset=["value"])
                agg = long.groupby(["ISO","Annee"])["value"].mean().reset_index()
                name = "eurostat_" + os.path.basename(f).replace(".tsv","").replace("eurostat_","")
                agg = agg.rename(columns={"value": f"{name}_count"})
                agg.to_csv(f"{CLN}/elevage/{name}.csv", index=False)
                print(f"     → {CLN}/elevage/{name}.csv ({len(agg)} lignes)")
        except Exception as e:
            print(f"     ⚠️ {os.path.basename(f)}: {str(e)[:80]}")


# ════════════════════════════════════════════════════════════════════════
print("\n[6] 🌾 FAO Crops+Livestock bundle (34 MB)…")
# ════════════════════════════════════════════════════════════════════════
zf = f"{RAW}/agriculture/FAO_crops_livestock.zip"
if os.path.exists(zf):
    try:
        with zipfile.ZipFile(zf) as z:
            csv_names = [n for n in z.namelist() if n.endswith(".csv") and "Normalized" in n]
            print(f"   Fichiers : {csv_names[:3]}")
            if csv_names:
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f, low_memory=False, encoding="latin-1")
                print(f"   {df.shape}, cols={list(df.columns)[:10]}")
                # Format FAOSTAT normalisé
                if "Element" in df.columns and "Item" in df.columns:
                    print(f"   Items uniques : {df['Item'].nunique()}")
                    print(f"   Elements : {df['Element'].unique()[:8]}")
                    # Garder yield + production pour quelques items spécifiques
                    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
                    df = df.dropna(subset=["Value"])
                    df["Area"] = df["Area"].astype(str)
                    df["ISO"] = df["Area"].apply(name_to_iso2)
                    df = df.dropna(subset=["ISO"])
                    # Filtrer Yield only
                    df_y = df[df["Element"].str.contains("Yield", case=False, na=False)]
                    print(f"   Yield lignes : {len(df_y)}")
                    if len(df_y):
                        # Pivot Item x ISO/Year
                        # Trop d'items → garde top 30 par couverture
                        top_items = df_y["Item"].value_counts().head(30).index
                        df_y = df_y[df_y["Item"].isin(top_items)]
                        piv = df_y.pivot_table(index=["ISO","Year"], columns="Item",
                                                values="Value", aggfunc="mean").reset_index()
                        piv = piv.rename(columns={"Year":"Annee"})
                        rename = {c: f"fao_cl_yield_{str(c).lower().replace(' ','_').replace(',','')[:25]}"
                                   for c in piv.columns if c not in ["ISO","Annee"]}
                        piv = piv.rename(columns=rename)
                        piv.to_csv(f"{CLN}/agriculture/fao_crops_livestock_yields_top30.csv", index=False)
                        print(f"   → fao_crops_livestock_yields_top30.csv ({len(piv)} lignes × {piv.shape[1]-2} items)")
    except Exception as e:
        print(f"   ⚠️ {str(e)[:120]}")


# ════════════════════════════════════════════════════════════════════════
print("\n[7] 🌱 FAO Fertilizers by Product (2.7 MB)…")
# ════════════════════════════════════════════════════════════════════════
zf = f"{RAW}/agriculture/FAO_fert_product.zip"
if os.path.exists(zf):
    try:
        with zipfile.ZipFile(zf) as z:
            csv = [n for n in z.namelist() if n.endswith(".csv") and "Normalized" in n][0]
            with z.open(csv) as f:
                df = pd.read_csv(f, low_memory=False, encoding="latin-1")
        print(f"   {df.shape}, items uniques: {df['Item'].nunique() if 'Item' in df else '?'}")
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna(subset=["Value"])
        df["ISO"] = df["Area"].apply(name_to_iso2)
        df = df.dropna(subset=["ISO"])
        df_filt = df[df["Element"].str.contains("Agricultural Use", case=False, na=False)]
        piv = df_filt.pivot_table(index=["ISO","Year"], columns="Item", values="Value",
                                    aggfunc="sum").reset_index()
        piv = piv.rename(columns={"Year":"Annee"})
        rename = {c: f"fao_fertp_{str(c).lower().replace(' ','_')[:25]}"
                   for c in piv.columns if c not in ["ISO","Annee"]}
        piv = piv.rename(columns=rename)
        piv.to_csv(f"{CLN}/agriculture/fao_fertilizers_by_product.csv", index=False)
        print(f"   → fao_fertilizers_by_product.csv ({len(piv)} lignes × {piv.shape[1]-2} produits)")
    except Exception as e:
        print(f"   ⚠️ {str(e)[:120]}")


# ════════════════════════════════════════════════════════════════════════
print("\n[8] 🌫️ FAO Emissions Intensities…")
# ════════════════════════════════════════════════════════════════════════
zf = f"{RAW}/agriculture/FAO_emissions_intensities.zip"
if os.path.exists(zf):
    try:
        with zipfile.ZipFile(zf) as z:
            csv = [n for n in z.namelist() if n.endswith(".csv") and "Normalized" in n][0]
            with z.open(csv) as f:
                df = pd.read_csv(f, low_memory=False, encoding="latin-1")
        print(f"   {df.shape}, items: {df['Item'].nunique() if 'Item' in df else '?'}")
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna(subset=["Value"])
        df["ISO"] = df["Area"].apply(name_to_iso2)
        df = df.dropna(subset=["ISO"])
        # Agg moyenne globale par pays/an
        agg = df.groupby(["ISO","Year"]).agg(
            fao_emissions_intensity_mean=("Value","mean"),
            fao_emissions_intensity_max=("Value","max"),
        ).reset_index().rename(columns={"Year":"Annee"})
        agg.to_csv(f"{CLN}/atmosphere/fao_emissions_intensities.csv", index=False)
        print(f"   → fao_emissions_intensities.csv ({len(agg)} lignes)")
    except Exception as e:
        print(f"   ⚠️ {str(e)[:120]}")


# ════════════════════════════════════════════════════════════════════════
print("\n[9] 🌿 PMP Toolkit (xlsx) — Plant Mitigation Pathways…")
# ════════════════════════════════════════════════════════════════════════
pmp = f"{RAW}/PMP-TAB_Toolkit_July_2024.xlsx"
if not os.path.exists(pmp):
    cands = glob.glob(f"{RAW}/**/PMP*.xlsx", recursive=True)
    pmp = cands[0] if cands else None
if pmp and os.path.exists(pmp):
    try:
        sheets = pd.read_excel(pmp, sheet_name=None, engine="openpyxl")
        print(f"   Sheets ({len(sheets)}): {list(sheets.keys())[:6]}")
        for sn, df in sheets.items():
            if df.shape[0] >= 10:
                print(f"     {sn}: {df.shape}, cols sample = {list(df.columns)[:6]}")
        # PMP est probablement un toolkit, pas un dataset par pays
        print("   ⚠️ PMP Toolkit = guide méthodologique, pas dataset utilisable directement")
    except Exception as e:
        print(f"   ⚠️ {str(e)[:120]}")


print("\n══════════════════════════════════════════════════════════════")
print("📊 BILAN — Nouveaux fichiers cleaned créés")
print("══════════════════════════════════════════════════════════════")
new_keywords = ["fao_fra2025","oecd_fao_outlook","earthquakes_M5plus","eurostat_eurostat",
                 "fao_crops_livestock","fao_fertilizers_by_product","fao_emissions_intensities",
                 "gbif_occurrences"]
for cat in ["atmosphere","hydrologie","sol_ecologie","agriculture","elevage","geologie"]:
    d = f"{CLN}/{cat}"
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if any(k in f for k in new_keywords):
                sz = os.path.getsize(f"{d}/{f}") / 1024
                print(f"  ✓ [{cat:13s}] {f}  ({sz:.0f} KB)")
