"""
process_all_new_data.py — Décompresse + inspecte + traite TOUS les nouveaux datasets.

Phase 1 : Décompression dans data/raw/<categorie>/<dataset>/
Phase 2 : Inspection (format, colonnes, taille, ISO/Année si présents)
Phase 3 : Agrégats par (ISO, Année) pour intégration cascade

Datasets traités :
  - EDGAR 2025 GHGs by sector
  - SPAM 2020 V2r2 (yield + harvested area) ⭐
  - WRI Aqueduct 4.0 water risk
  - FAO Fertilizers Nutrient v2
  - FAO Investment Machinery
  - FAO Value of Production
  - AQUASTAT (FAO water)
  - EPI 2024 Yale
  - NASA FIRMS (MODIS + VIIRS)
  - IUCN Red List (4 exports)
  - GBIF biodiversity
  - PSMSL rlr_annual (sea level)
  - bulk_eng (à identifier)
"""
import os, sys, io, zipfile, glob, time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

RAW = "data/raw"
CLN = "data/cleaned"
os.makedirs(CLN, exist_ok=True)


def safe_unzip(zip_path, out_dir):
    """Decompresse un zip dans un dossier dédié."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            members = z.namelist()
            z.extractall(out_dir)
        return True, len(members), members[:5]
    except Exception as e:
        return False, 0, [str(e)[:80]]


def try_read_csv(path, **kwargs):
    """Lit un CSV avec fallback latin-1."""
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False, **kwargs)


print("══════════════════════════════════════════════════════════════")
print("📦 PHASE 1 — DÉCOMPRESSION")
print("══════════════════════════════════════════════════════════════\n")

zips_to_unzip = [
    # (zip_path, out_dir, label)
    (f"{RAW}/EDGAR_2025_GHGs_CO2eq_AR5_NUTS2_by_country_sector_1990-2024_b.zip",
     f"{RAW}/edgar_ghgs_2025", "EDGAR 2025 GHGs"),
    (f"{RAW}/spam2020V2r2_global_yield.csv.zip",
     f"{RAW}/spam2020_v2r2_yield", "SPAM 2020 V2r2 Yield"),
    (f"{RAW}/spam2020V2r2_global_harvested_area.csv.zip",
     f"{RAW}/spam2020_v2r2_harvested", "SPAM 2020 V2r2 Harvested"),
    (f"{RAW}/aqueduct-4-0-water-risk-data.zip",
     f"{RAW}/aqueduct_40", "WRI Aqueduct 4.0"),
    (f"{RAW}/epi2024raw.zip",
     f"{RAW}/epi_2024", "EPI 2024 Yale"),
    (f"{RAW}/Inputs_FertilizersNutrient_E_All_Data.zip",
     f"{RAW}/fao_fertilizers_v2", "FAO Fertilizers Nutrient v2"),
    (f"{RAW}/Investment_Machinery_E_All_Data_(Normalized).zip",
     f"{RAW}/fao_machinery", "FAO Investment Machinery"),
    (f"{RAW}/Value_of_Production_E_All_Data.zip",
     f"{RAW}/fao_value_production", "FAO Value of Production"),
]

# IUCN Red List : 4 zips à fusionner
iucn_zips = sorted(glob.glob(f"{RAW}/redlist_species_data_*.zip"))
for i, zp in enumerate(iucn_zips):
    zips_to_unzip.append((zp, f"{RAW}/iucn_redlist_{i+1}", f"IUCN Red List #{i+1}"))

unzip_results = []
for zp, out, label in zips_to_unzip:
    if not os.path.exists(zp):
        print(f"  ✗ {label:35s} zip absent")
        unzip_results.append({"dataset": label, "status": "missing"})
        continue
    if os.path.exists(out) and len(os.listdir(out)) > 0:
        n = len(os.listdir(out))
        print(f"  ✓ {label:35s} déjà décompressé ({n} fichiers)")
        unzip_results.append({"dataset": label, "status": "exists"})
        continue
    ok, n, sample = safe_unzip(zp, out)
    sym = "✓" if ok else "✗"
    print(f"  {sym} {label:35s} {n} fichiers extraits → {out}")
    if not ok: print(f"    Error: {sample}")
    unzip_results.append({"dataset": label, "status": "ok" if ok else "fail",
                           "n_files": n})

print(f"\n{sum(1 for r in unzip_results if r['status'] in ('ok','exists'))}/{len(unzip_results)} zips OK")


print("\n══════════════════════════════════════════════════════════════")
print("📊 PHASE 2 — INSPECTION + TRAITEMENT PAR CATÉGORIE")
print("══════════════════════════════════════════════════════════════")


# ── 1. SPAM 2020 V2r2 YIELD ───────────────────────────────────────────────
print("\n[1] 🌾 SPAM 2020 V2r2 Yield (PRIORITÉ AGRICULTURE)…")
spam_dir = f"{RAW}/spam2020_v2r2_yield"
if os.path.isdir(spam_dir):
    csvs = glob.glob(f"{spam_dir}/*.csv") + glob.glob(f"{spam_dir}/**/*.csv", recursive=True)
    print(f"   {len(csvs)} CSV trouvés")
    # SPAM standard : 1 fichier par technologie (all=A, irrig=I, low-input=L, rainfed-high=H, subsistence=S)
    for csv in csvs[:3]:
        print(f"   - {os.path.basename(csv)} ({os.path.getsize(csv)/1e6:.1f} MB)")

    # On prend "all technologies" si dispo
    main_csv = None
    for c in csvs:
        bn = os.path.basename(c).lower()
        if "_a.csv" in bn or "_all" in bn or "all_technologies" in bn:
            main_csv = c; break
    if not main_csv and csvs:
        main_csv = csvs[0]

    if main_csv:
        print(f"\n   → Traite : {os.path.basename(main_csv)}")
        df = try_read_csv(main_csv, nrows=5)
        print(f"   Cols sample : {list(df.columns)[:15]}")

        # Lire en entier (peut être gros)
        try:
            df = try_read_csv(main_csv)
            print(f"   Shape complète : {df.shape}")

            # Identifier colonne ISO
            iso_cols = [c for c in df.columns if c.lower() in
                         ("iso3", "iso", "name_cntr", "name_cntr_a3", "cntry_code")]
            iso_col = iso_cols[0] if iso_cols else None
            print(f"   ISO col : {iso_col}")

            # Identifier colonnes culture (codes 4-letter)
            crop_cols = [c for c in df.columns if len(c) == 4 and c.lower() in
                          ("whea","rice","maiz","barl","pmil","smil","sorg","ocer",
                           "pota","swpo","yams","cass","orts","bean","chic","cowp",
                           "pige","lent","opul","soyb","grou","cnut","oilp","sunf",
                           "rape","sesa","ooil","sugc","sugb","cott","ofib","acof",
                           "rcof","coco","teas","toba","bana","plnt","trof","temf",
                           "vege","rest")]
            print(f"   {len(crop_cols)} colonnes culture détectées")

            if iso_col and crop_cols:
                # Agrégation par ISO (moyenne pondérée par cellule, mais simple mean ici)
                agg = df.groupby(iso_col)[crop_cols].mean().reset_index()
                agg = agg.rename(columns={iso_col: "ISO3"})
                # Préfixer
                rename = {c: f"spam_yield_{c.lower()}" for c in crop_cols}
                agg = agg.rename(columns=rename)
                # Mapper ISO3 → ISO2
                import pycountry
                def iso3_to_iso2(c):
                    try:
                        co = pycountry.countries.get(alpha_3=c)
                        return co.alpha_2 if co else None
                    except: return None
                agg["ISO"] = agg["ISO3"].apply(iso3_to_iso2)
                agg = agg.dropna(subset=["ISO"])
                # Ajouter Annee = 2020 (SPAM 2020 est statique)
                agg["Annee"] = 2020
                cols_out = ["ISO", "Annee"] + [c for c in agg.columns if c.startswith("spam_yield_")]
                agg[cols_out].to_csv(f"{CLN}/spam2020_v2_yield_by_country.csv", index=False)
                print(f"   → {CLN}/spam2020_v2_yield_by_country.csv ({len(agg)} pays × {len(crop_cols)} cultures)")
        except Exception as e:
            print(f"   ⚠️ Erreur lecture : {str(e)[:100]}")


# ── 2. EDGAR 2025 GHGs by sector ──────────────────────────────────────────
print("\n[2] 🌫️ EDGAR 2025 GHGs by sector…")
edgar_dir = f"{RAW}/edgar_ghgs_2025"
if os.path.isdir(edgar_dir):
    files = glob.glob(f"{edgar_dir}/**/*", recursive=True)
    csv_xlsx = [f for f in files if f.lower().endswith((".csv",".xlsx"))]
    print(f"   {len(csv_xlsx)} fichiers data trouvés")
    for f in csv_xlsx[:5]:
        sz = os.path.getsize(f) / 1024
        print(f"   - {os.path.basename(f)} ({sz:.0f} KB)")

    # Prendre le fichier principal
    for f in csv_xlsx:
        bn = os.path.basename(f).lower()
        try:
            if bn.endswith(".xlsx"):
                df = pd.read_excel(f, sheet_name=None)
                print(f"   {bn}: sheets = {list(df.keys())[:5]}")
                # Process première sheet utile
                for sheet_name, sheet_df in df.items():
                    if sheet_df.shape[0] < 10: continue
                    print(f"     Sheet '{sheet_name}': {sheet_df.shape}, cols = {list(sheet_df.columns)[:8]}")
                    # Si format pivot avec années en colonnes
                    year_cols = [c for c in sheet_df.columns if isinstance(c, (int, float)) or (isinstance(c, str) and c.isdigit())]
                    if len(year_cols) > 10:
                        # Long format
                        id_cols = [c for c in sheet_df.columns if c not in year_cols]
                        long = sheet_df.melt(id_vars=id_cols, value_vars=year_cols,
                                              var_name="Annee", value_name="Value")
                        long["Annee"] = pd.to_numeric(long["Annee"], errors="coerce")
                        long = long.dropna(subset=["Annee", "Value"])
                        long["Annee"] = long["Annee"].astype(int)
                        out_name = f"edgar_2025_{sheet_name.lower().replace(' ','_')[:20]}.csv"
                        long.to_csv(f"{CLN}/{out_name}", index=False)
                        print(f"     → {CLN}/{out_name} ({len(long)} lignes)")
                        break
                break
            elif bn.endswith(".csv"):
                df = try_read_csv(f, nrows=5)
                print(f"   {bn}: cols = {list(df.columns)[:10]}")
        except Exception as e:
            print(f"   ⚠️ {bn}: {str(e)[:80]}")


# ── 3. AQUEDUCT 4.0 ──────────────────────────────────────────────────────
print("\n[3] 💧 WRI Aqueduct 4.0…")
aq_dir = f"{RAW}/aqueduct_40"
if os.path.isdir(aq_dir):
    files = []
    for ext in ("*.csv","*.xlsx","*.geojson","*.gpkg","*.shp"):
        files.extend(glob.glob(f"{aq_dir}/**/{ext}", recursive=True))
    print(f"   {len(files)} fichiers data")
    csvs = [f for f in files if f.endswith(".csv")]
    xlsxs = [f for f in files if f.endswith(".xlsx")]
    for f in csvs[:5]: print(f"   CSV: {os.path.basename(f)} ({os.path.getsize(f)/1e6:.1f} MB)")
    for f in xlsxs[:5]: print(f"   XLSX: {os.path.basename(f)} ({os.path.getsize(f)/1e6:.1f} MB)")

    # Cherche le fichier country-level
    for f in csvs + xlsxs:
        bn = os.path.basename(f).lower()
        if "country" in bn or "summary" in bn or "rank" in bn:
            try:
                if bn.endswith(".csv"):
                    df = try_read_csv(f)
                else:
                    df = pd.read_excel(f)
                print(f"   {bn}: {df.shape}, cols = {list(df.columns)[:10]}")
                # Sauvegarder tel quel
                out = f"{CLN}/wri_aqueduct40_{bn.replace('.xlsx','').replace('.csv','')[:30]}.csv"
                df.to_csv(out, index=False)
                print(f"   → {out}")
                break
            except Exception as e:
                print(f"   ⚠️ {bn}: {str(e)[:80]}")


# ── 4. EPI 2024 ──────────────────────────────────────────────────────────
print("\n[4] 🌿 EPI 2024 Yale…")
epi_dir = f"{RAW}/epi_2024"
if os.path.isdir(epi_dir):
    files = glob.glob(f"{epi_dir}/**/*", recursive=True)
    data_files = [f for f in files if f.lower().endswith((".csv",".xlsx"))]
    print(f"   {len(data_files)} fichiers data")
    for f in data_files[:8]:
        print(f"   - {os.path.basename(f)} ({os.path.getsize(f)/1024:.0f} KB)")

    # Cherche les EPI scores
    for f in data_files:
        bn = os.path.basename(f).lower()
        if "epi" in bn and ("results" in bn or "score" in bn or "indicator" in bn):
            try:
                if bn.endswith(".csv"):
                    df = try_read_csv(f)
                else:
                    df = pd.read_excel(f, sheet_name=0)
                print(f"   {bn}: {df.shape}")
                out = f"{CLN}/epi_2024_{bn.replace('.xlsx','').replace('.csv','')[:30]}.csv"
                df.to_csv(out, index=False)
                print(f"   → {out}")
            except Exception as e:
                print(f"   ⚠️ {str(e)[:80]}")


# ── 5. NASA FIRMS (fires) ─────────────────────────────────────────────────
print("\n[5] 🔥 NASA FIRMS Active Fires…")
firms_files = [f"{RAW}/MODIS_C6_1_Global_24h.csv", f"{RAW}/MODIS_C6_1_Global_7d.csv",
                f"{RAW}/J1_VIIRS_C2_Global_24h.csv", f"{RAW}/J1_VIIRS_C2_Global_48h.csv",
                f"{RAW}/J2_VIIRS_C2_Global_24h.csv"]
fires_dfs = []
for f in firms_files:
    if not os.path.exists(f): continue
    try:
        df = pd.read_csv(f, low_memory=False)
        print(f"   {os.path.basename(f)}: {df.shape}, cols={list(df.columns)[:8]}")
        fires_dfs.append(df)
    except Exception as e:
        print(f"   ⚠️ {os.path.basename(f)}: {str(e)[:80]}")

if fires_dfs:
    # NB : FIRMS donne juste un snapshot ponctuel (24h/7d récent), pas historique
    # On peut compter les feux par pays récents et faire un proxy "fire activity"
    all_fires = pd.concat(fires_dfs, ignore_index=True)
    print(f"   Total feux récents combinés : {len(all_fires):,}")
    if "latitude" in all_fires.columns and "longitude" in all_fires.columns:
        # Pour assigner à un pays, il faudrait un shapefile. À défaut, on save brut.
        all_fires.to_csv(f"{CLN}/firms_active_fires_recent.csv", index=False)
        print(f"   → {CLN}/firms_active_fires_recent.csv (à géoréférencer plus tard)")
        print(f"   ⚠️ NOTE : FIRMS est un snapshot temps réel, pas une série historique par pays.")


# ── 6. IUCN Red List ─────────────────────────────────────────────────────
print("\n[6] 🦎 IUCN Red List…")
iucn_dfs = []
for i in range(1, 10):
    d = f"{RAW}/iucn_redlist_{i}"
    if not os.path.isdir(d): continue
    files = glob.glob(f"{d}/**/*.csv", recursive=True)
    # Le fichier principal IUCN s'appelle souvent "assessments.csv" ou "simple_summary.csv"
    for f in files:
        bn = os.path.basename(f).lower()
        try:
            df = pd.read_csv(f, low_memory=False)
            if "scientificName" in df.columns or "scientific_name" in df.columns or "scientificname" in df.columns:
                if "redlistCategory" in df.columns or "category" in df.columns or "redlistcategory" in df.columns:
                    print(f"   iucn_{i}/{bn}: {df.shape}")
                    iucn_dfs.append(df)
                    break
        except Exception:
            pass

if iucn_dfs:
    # Concat all
    iucn = pd.concat(iucn_dfs, ignore_index=True)
    # Dédupliquer
    if "scientificName" in iucn.columns:
        iucn = iucn.drop_duplicates(subset=["scientificName"])
    print(f"   Total espèces uniques: {len(iucn):,}")
    iucn.to_csv(f"{CLN}/iucn_redlist_species.csv", index=False)
    print(f"   → {CLN}/iucn_redlist_species.csv")
    print(f"   ⚠️ Pour agrégat par pays : nécessite jointure avec range data (autre dataset IUCN)")


# ── 7. AQUASTAT Dissemination System ─────────────────────────────────────
print("\n[7] 💦 AQUASTAT…")
aqua_csv = f"{RAW}/AQUASTAT Dissemination System.csv"
if os.path.exists(aqua_csv):
    df = try_read_csv(aqua_csv)
    print(f"   Shape : {df.shape}")
    print(f"   Cols : {list(df.columns)[:10]}")

    # Format typique : Country | Variable Name | Year | Value
    if "Country" in df.columns and "Variable Name" in df.columns and "Year" in df.columns:
        # Pivot par variable
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.dropna(subset=["Year", "Value"])
        df["Year"] = df["Year"].astype(int)

        # Garder seulement variables clés (sinon trop de cols)
        keep_vars = df["Variable Name"].value_counts().head(30).index
        df_k = df[df["Variable Name"].isin(keep_vars)]
        piv = df_k.pivot_table(index=["Country", "Year"], columns="Variable Name",
                                 values="Value", aggfunc="mean").reset_index()
        piv.columns = ["Pays" if c=="Country" else
                        "Annee" if c=="Year" else
                        f"aquastat_{str(c).lower().replace(' ','_').replace(',','')[:30]}"
                        for c in piv.columns]
        piv.to_csv(f"{CLN}/aquastat_top30_variables.csv", index=False)
        print(f"   → {CLN}/aquastat_top30_variables.csv ({len(piv)} lignes, top 30 variables)")


# ── 8. FAO Machinery v2 ──────────────────────────────────────────────────
print("\n[8] 🚜 FAO Investment Machinery…")
mach_dir = f"{RAW}/fao_machinery"
if os.path.isdir(mach_dir):
    csvs = glob.glob(f"{mach_dir}/*.csv")
    for f in csvs:
        try:
            df = try_read_csv(f)
            print(f"   {os.path.basename(f)}: {df.shape}")
            if "Element" in df.columns:
                print(f"   Elements: {df['Element'].unique()[:5]}")
                print(f"   Items: {df['Item'].unique()[:10]}")
            # Pivot Item × Year
            if "Element" in df.columns and "Item" in df.columns:
                df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
                df = df.dropna(subset=["Value"])
                area_col = "Area" if "Area" in df.columns else "Country"
                piv = df.pivot_table(index=[area_col, "Year"], columns="Item",
                                       values="Value", aggfunc="mean").reset_index()
                piv = piv.rename(columns={area_col: "Pays", "Year": "Annee"})
                rename = {c: f"machinery_{str(c).lower().replace(' ','_')[:25]}"
                           for c in piv.columns if c not in ["Pays","Annee"]}
                piv = piv.rename(columns=rename)
                piv.to_csv(f"{CLN}/fao_machinery.csv", index=False)
                print(f"   → {CLN}/fao_machinery.csv ({len(piv)} lignes)")
                break
        except Exception as e:
            print(f"   ⚠️ {str(e)[:80]}")


# ── 9. FAO Value of Production ───────────────────────────────────────────
print("\n[9] 💵 FAO Value of Production…")
val_dir = f"{RAW}/fao_value_production"
if os.path.isdir(val_dir):
    csvs = glob.glob(f"{val_dir}/*.csv")
    for f in csvs:
        try:
            df = try_read_csv(f)
            print(f"   {os.path.basename(f)}: {df.shape}")
            if "Element" in df.columns:
                print(f"   Elements: {df['Element'].unique()[:5]}")
                # Garder une seule élément (Gross Production Value typique)
                df_filt = df[df["Element"].str.contains("Gross Production Value", case=False, na=False)]
                if len(df_filt) == 0:
                    df_filt = df[df["Element"] == df["Element"].iloc[0]]
                # Agréger par pays/an (somme sur tous Items)
                df_filt["Value"] = pd.to_numeric(df_filt["Value"], errors="coerce")
                area_col = "Area" if "Area" in df_filt.columns else "Country"
                agg = df_filt.groupby([area_col, "Year"]).agg(
                    val_production_total=("Value", "sum"),
                    val_production_n_items=("Item", "nunique"),
                ).reset_index()
                agg = agg.rename(columns={area_col: "Pays", "Year": "Annee"})
                agg.to_csv(f"{CLN}/fao_value_of_production.csv", index=False)
                print(f"   → {CLN}/fao_value_of_production.csv ({len(agg)} lignes)")
                break
        except Exception as e:
            print(f"   ⚠️ {str(e)[:80]}")


# ── 10. GBIF datasets (tsv) ──────────────────────────────────────────────
print("\n[10] 🌿 GBIF datasets…")
gbif = f"{RAW}/gbif_datasets.tsv"
if os.path.exists(gbif):
    try:
        df = pd.read_csv(gbif, sep="\t", low_memory=False)
        print(f"   Shape: {df.shape}, cols={list(df.columns)[:10]}")
        # Pas un dataset par pays/année à proprement parler — c'est un catalogue
        print(f"   ⚠️ Catalogue de datasets GBIF (pas observation), à utiliser pour référence")
    except Exception as e:
        print(f"   ⚠️ {str(e)[:80]}")


# ── 11. PSMSL Sea Level (rlr_annual) ─────────────────────────────────────
print("\n[11] 🌊 PSMSL Sea Level…")
rlr_dir = f"{RAW}/rlr_annual"
if os.path.isdir(rlr_dir):
    files = glob.glob(f"{rlr_dir}/**/*", recursive=True)
    print(f"   {len(files)} fichiers")
    # PSMSL format = data/ + filelist.txt
    flist = f"{rlr_dir}/filelist.txt"
    if not os.path.exists(flist):
        flist = next((f for f in files if "filelist" in f.lower()), None)
    if flist and os.path.exists(flist):
        print(f"   filelist trouvé : {flist}")
        try:
            # Format PSMSL: ID; LAT; LON; NAME; COUNTRY; YEAR_START; YEAR_END
            meta = pd.read_csv(flist, sep=";", header=None, encoding="latin-1",
                                names=["id","lat","lon","name","country","coastline","stationcode","quality"])
            print(f"   {len(meta)} stations")
            # Données : data/<id>.rlrdata
            station_dfs = []
            data_files = glob.glob(f"{rlr_dir}/**/data/*.rlrdata", recursive=True)
            print(f"   {len(data_files)} fichiers de données")
            for df_path in data_files[:300]:  # limiter
                sid = int(os.path.basename(df_path).replace(".rlrdata", ""))
                try:
                    d = pd.read_csv(df_path, sep=";", header=None,
                                      names=["year_frac","value","interpolated","flag"])
                    d["station_id"] = sid
                    d["year"] = d["year_frac"].astype(int)
                    d = d[d["value"] > -30000]  # PSMSL = -99999 for missing
                    station_dfs.append(d)
                except Exception: pass
            if station_dfs:
                all_data = pd.concat(station_dfs, ignore_index=True)
                merged = all_data.merge(meta[["id","country","lat"]],
                                          left_on="station_id", right_on="id")
                # Agréger par pays/année
                agg = merged.groupby(["country", "year"]).agg(
                    sea_level_mean=("value", "mean"),
                    sea_level_min=("value", "min"),
                    sea_level_max=("value", "max"),
                    n_stations=("station_id", "nunique"),
                ).reset_index().rename(columns={"country": "Pays", "year": "Annee"})
                agg.to_csv(f"{CLN}/psmsl_sea_level_by_country.csv", index=False)
                print(f"   → {CLN}/psmsl_sea_level_by_country.csv ({len(agg)} lignes)")
        except Exception as e:
            print(f"   ⚠️ {str(e)[:120]}")


# ── 12. bulk_eng(in).csv ─────────────────────────────────────────────────
print("\n[12] 📄 bulk_eng(in).csv (à identifier)…")
bulk = f"{RAW}/bulk_eng(in).csv"
if os.path.exists(bulk):
    try:
        df = try_read_csv(bulk, nrows=5)
        print(f"   Cols : {list(df.columns)[:15]}")
        print(f"   Première ligne : {df.iloc[0].to_dict()}")
    except Exception as e:
        print(f"   ⚠️ {str(e)[:80]}")


# ── 13. mrds.csv (USGS Mineral Resources) ────────────────────────────────
print("\n[13] ⛏️ MRDS USGS Mineral Resources…")
mrds = f"{RAW}/mrds.csv"
if os.path.exists(mrds):
    try:
        df = try_read_csv(mrds, nrows=5)
        print(f"   Cols : {list(df.columns)[:12]}")
    except Exception as e:
        print(f"   ⚠️ {str(e)[:80]}")


# ════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════════════════════════")
print("📊 BILAN FICHIERS CLEANED CRÉÉS")
print("══════════════════════════════════════════════════════════════")
new_files = []
for f in os.listdir(CLN):
    if any(k in f for k in ["spam2020_v2", "edgar_2025", "wri_aqueduct", "epi_2024",
                              "firms_active", "iucn_redlist_species", "aquastat_top30",
                              "fao_machinery", "fao_value_of_production",
                              "psmsl_sea_level"]):
        sz = os.path.getsize(f"{CLN}/{f}") / 1024
        new_files.append((f, sz))
        print(f"  ✓ {f} ({sz:.0f} KB)")
print(f"\n{len(new_files)} nouveaux fichiers cleaned créés")
