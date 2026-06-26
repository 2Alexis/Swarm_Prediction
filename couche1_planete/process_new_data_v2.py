"""
process_new_data_v2.py — Correction des datasets non traités v1.

Fixes :
  - EDGAR avec openpyxl
  - SPAM 2020 V2 (mapping nom→ISO car pas de colonne ISO3 directe)
  - EPI 2024 (210 fichiers BCA_raw.csv format wide → long fusionné)
  - IUCN Red List (compter espèces menacées par pays)
  - WRI Aqueduct 4.0 (filtre country aggregates)
  - AQUASTAT (format m49 wide)
  - FAO Value of Production
  - bulk_eng (AQUASTAT alternative)
"""
import os, sys, io, glob, json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from build_dataset import custom_mappings, get_english_iso

RAW = "data/raw"
CLN = "data/cleaned"

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
    code = get_english_iso(name)
    return code


# ════════════════════════════════════════════════════════════════════════
print("\n[1] 🌫️ EDGAR 2025 GHGs (Excel multi-sheets)…")
# ════════════════════════════════════════════════════════════════════════
edgar_xlsx = glob.glob(f"{RAW}/edgar_ghgs_2025/**/*.xlsx", recursive=True)
if edgar_xlsx:
    f = edgar_xlsx[0]
    print(f"   Loading: {os.path.basename(f)}")
    try:
        sheets = pd.read_excel(f, sheet_name=None, engine="openpyxl")
        print(f"   Sheets: {list(sheets.keys())}")
        for sheet_name, df in sheets.items():
            if df.shape[0] < 10: continue
            print(f"\n   Sheet '{sheet_name}': {df.shape}")
            print(f"     Cols (10): {list(df.columns)[:10]}")
            # EDGAR format probable : Country, Sector, year_cols (Y_1990 ... Y_2024)
            year_cols = [c for c in df.columns
                          if (isinstance(c, str) and c.startswith(("Y_","y_")) and c[2:].isdigit())
                          or (isinstance(c, int) and 1900 < c < 2050)]
            if len(year_cols) > 5:
                # Long format
                id_cols = [c for c in df.columns if c not in year_cols]
                long = df.melt(id_vars=id_cols, value_vars=year_cols,
                                var_name="Annee", value_name="Value")
                long["Annee"] = long["Annee"].astype(str).str.replace("Y_","").str.replace("y_","")
                long["Annee"] = pd.to_numeric(long["Annee"], errors="coerce")
                long = long.dropna(subset=["Annee","Value"])
                long["Annee"] = long["Annee"].astype(int)

                # Détecter colonne pays
                country_col = next((c for c in id_cols if any(k in str(c).lower()
                                     for k in ["country","name","iso"])), None)
                sector_col = next((c for c in id_cols if "sector" in str(c).lower()), None)
                print(f"     Country col: {country_col}, Sector col: {sector_col}")

                if country_col and sector_col:
                    # Pivot par secteur
                    long["ISO"] = long[country_col].apply(name_to_iso2) if "iso" not in str(country_col).lower() \
                                   else long[country_col].apply(lambda c: iso3_to_iso2(c) if isinstance(c,str) and len(c)==3 else c)
                    long = long.dropna(subset=["ISO"])
                    long[sector_col] = long[sector_col].astype(str).str[:30]
                    piv = long.pivot_table(index=["ISO","Annee"], columns=sector_col,
                                            values="Value", aggfunc="sum").reset_index()
                    rename = {c: f"edgar_ghg_{str(c).lower().replace(' ','_').replace(',','')[:35]}"
                               for c in piv.columns if c not in ["ISO","Annee"]}
                    piv = piv.rename(columns=rename)
                    out = f"{CLN}/edgar_2025_ghg_by_sector.csv"
                    piv.to_csv(out, index=False)
                    print(f"     → {out} ({len(piv)} lignes, {len([c for c in piv.columns if c.startswith('edgar_')])} secteurs)")
                    break
    except Exception as e:
        print(f"   ⚠️ {str(e)[:200]}")


# ════════════════════════════════════════════════════════════════════════
print("\n[2] 🌾 SPAM 2020 V2r2 yield (mapping ADM0_NAME→ISO)…")
# ════════════════════════════════════════════════════════════════════════
# Y_TA = All technologies. Très gros fichier (447 MB). On lit par chunks.
spam_csv = f"{RAW}/spam2020_v2r2_yield/spam2020V2r2_global_Y_TA.csv"
if os.path.exists(spam_csv):
    print(f"   Lecture chunks (447 MB)…")
    crop_cols = ['whea','rice','maiz','barl','mill','sorg','ocer','pota','swpo','yams',
                 'cass','orts','bean','chic','cowp','pige','lent','opul','soyb','grou',
                 'cnut','oilp','sunf','rape','sesa','ooil','sugc','sugb','cott','ofib',
                 'acof','rcof','coco','teas','toba','bana','plnt','trof','temf','vege','rest']
    keep_cols = ["ADM0_NAME"] + crop_cols
    chunks = []
    for chunk in pd.read_csv(spam_csv, usecols=lambda c: c in keep_cols,
                              chunksize=200000, low_memory=False):
        # Agrégation immédiate par pays
        agg = chunk.groupby("ADM0_NAME")[crop_cols].agg(["mean","count"])
        chunks.append(agg)
    # Combiner les chunks
    print(f"   {len(chunks)} chunks lus")
    combined = pd.concat(chunks)
    # Re-agréger
    final = combined.groupby("ADM0_NAME").apply(
        lambda g: pd.Series({c: (g[(c,"mean")] * g[(c,"count")]).sum() / max(g[(c,"count")].sum(),1)
                              for c in crop_cols}),
        include_groups=False
    ).reset_index()
    # Mapper pays → ISO
    final["ISO"] = final["ADM0_NAME"].apply(name_to_iso2)
    final = final.dropna(subset=["ISO"])
    final["Annee"] = 2020
    rename = {c: f"spam_yield_{c}" for c in crop_cols}
    final = final.rename(columns=rename)
    cols_out = ["ISO","Annee"] + [c for c in final.columns if c.startswith("spam_yield_")]
    final[cols_out].to_csv(f"{CLN}/spam2020_v2_yield_by_country.csv", index=False)
    print(f"   → {CLN}/spam2020_v2_yield_by_country.csv ({len(final)} pays × {len(crop_cols)} cultures)")


# ════════════════════════════════════════════════════════════════════════
print("\n[3] 🌿 EPI 2024 Yale (210 fichiers wide → long fusionné)…")
# ════════════════════════════════════════════════════════════════════════
epi_dir = f"{RAW}/epi_2024/Raw"
if os.path.isdir(epi_dir):
    csvs = [f for f in glob.glob(f"{epi_dir}/*.csv") if "_na" not in os.path.basename(f).lower()]
    print(f"   {len(csvs)} indicateurs (sans _na variants)")

    all_indicators = []
    for f in csvs:
        try:
            df = pd.read_csv(f, low_memory=False)
            # Trouver code indicateur
            code = os.path.basename(f).replace("_raw.csv","").replace(".csv","")
            # Colonnes années
            year_cols = [c for c in df.columns if "." in str(c)
                          and str(c).split(".")[-1].isdigit() and 1990 <= int(str(c).split(".")[-1]) <= 2030]
            if not year_cols: continue
            # Long format
            id_cols = ["code","iso","country"] if "iso" in df.columns else ["iso","country"]
            id_cols = [c for c in id_cols if c in df.columns]
            long = df.melt(id_vars=id_cols, value_vars=year_cols,
                            var_name="year_col", value_name="value")
            long["Annee"] = long["year_col"].astype(str).str.split(".").str[-1].astype(int)
            long["Indicator"] = code
            # Remplacer -9999 (missing) par NaN
            long.loc[long["value"] <= -9000, "value"] = np.nan
            long = long.dropna(subset=["value"])
            if "iso" in long.columns:
                long["ISO"] = long["iso"].apply(iso3_to_iso2)
                long = long.dropna(subset=["ISO"])
            all_indicators.append(long[["ISO","Annee","Indicator","value"]])
        except Exception as e:
            pass

    if all_indicators:
        combined = pd.concat(all_indicators, ignore_index=True)
        print(f"   {len(combined):,} observations totales, {combined['Indicator'].nunique()} indicateurs")
        # Pivot indicateurs en colonnes
        # Trop d'indicateurs ? Garder top 30 + EPI score
        top_indicators = combined["Indicator"].value_counts().head(30).index
        # Forcer inclusion EPI/COV principaux
        priority = ["EPI","ECO","BHV","CCH","AIR","H2O","WMG","BCA"]
        for p in priority:
            for ind in combined["Indicator"].unique():
                if ind.startswith(p) and ind not in top_indicators:
                    top_indicators = top_indicators.append(pd.Index([ind]))

        combined_keep = combined[combined["Indicator"].isin(top_indicators)]
        piv = combined_keep.pivot_table(index=["ISO","Annee"], columns="Indicator",
                                          values="value", aggfunc="mean").reset_index()
        rename = {c: f"epi_{str(c).lower()}" for c in piv.columns if c not in ["ISO","Annee"]}
        piv = piv.rename(columns=rename)
        piv.to_csv(f"{CLN}/epi_2024_indicators.csv", index=False)
        print(f"   → {CLN}/epi_2024_indicators.csv ({len(piv)} lignes, {piv.shape[1]-2} indicateurs)")


# ════════════════════════════════════════════════════════════════════════
print("\n[4] 🦎 IUCN Red List — points par pays…")
# ════════════════════════════════════════════════════════════════════════
# Combiner les 4 zips
iucn_files = []
for i in range(1, 5):
    pf = f"{RAW}/iucn_redlist_{i}/points_data.csv"
    if os.path.exists(pf):
        iucn_files.append(pf)
print(f"   {len(iucn_files)} fichiers points_data")

if iucn_files:
    iucn_dfs = []
    for pf in iucn_files:
        try:
            df = pd.read_csv(pf, low_memory=False)
            iucn_dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ {pf}: {str(e)[:80]}")
    if iucn_dfs:
        iucn = pd.concat(iucn_dfs, ignore_index=True)
        print(f"   Total points : {len(iucn):,}, espèces uniques : {iucn['sci_name'].nunique():,}")

        # Géoréférencer chaque point → pays via lat/lon
        # Méthode simple : utiliser reverse_geocoder (offline) ou un shapefile
        # Sans dépendance externe : on garde juste lat/lon et le total

        # Pour avoir un compte par pays, on utilise reverse_geocoder
        try:
            import reverse_geocoder as rg
            # Échantillonner pour gagner du temps
            sample = iucn.dropna(subset=["dec_lat","dec_long"])
            print(f"   Reverse-geocoding {len(sample):,} points...")
            coords = list(zip(sample["dec_lat"], sample["dec_long"]))
            results = rg.search(coords, mode=1, verbose=False)
            sample["ISO"] = [r["cc"] for r in results]
            # Compter espèces uniques par pays
            agg = sample.groupby("ISO").agg(
                iucn_species_count=("sci_name", "nunique"),
                iucn_observations=("sci_name", "count"),
            ).reset_index()
            agg.to_csv(f"{CLN}/iucn_species_by_country.csv", index=False)
            print(f"   → {CLN}/iucn_species_by_country.csv ({len(agg)} pays)")
        except ImportError:
            print("   ⚠️ reverse_geocoder pas installé : pip install reverse_geocoder")
            # Sauvegarder bruts
            iucn.to_csv(f"{CLN}/iucn_redlist_points_raw.csv", index=False)
            print(f"   → {CLN}/iucn_redlist_points_raw.csv (raw, à géoréférencer)")


# ════════════════════════════════════════════════════════════════════════
print("\n[5] 💧 WRI Aqueduct 4.0 (filtrer agrégats pays)…")
# ════════════════════════════════════════════════════════════════════════
aq_csv = f"{RAW}/aqueduct_40/Aqueduct40_baseline_annual_y2023m07d05.csv"
if os.path.exists(aq_csv):
    print(f"   Lecture (202 MB)…")
    # Trop gros pour mémoire complète : chunks
    chunks = []
    cols_keep = None
    for ch in pd.read_csv(aq_csv, chunksize=200000, low_memory=False):
        if cols_keep is None:
            cols = list(ch.columns)
            print(f"   Cols ({len(cols)}): {cols[:15]}")
            # Filtrer cols utiles : gid_0/iso/cntry + indicateurs principaux
            id_cols = [c for c in cols if c.lower() in ("gid_0","iso","name_0","cntry_name","gid_1")]
            # Indicateurs Aqueduct : bws (baseline water stress), bwd (depletion), gtd (groundwater),
            # rfr (riverine flood risk), drr (drought risk)
            ind_cols = [c for c in cols if any(c.lower().startswith(p) for p in
                          ["bws_","bwd_","gtd_","rfr_","drr_","cfr_","ucw_","ucr_","cep_","udw_","usa_","iav_","sev_"])]
            ind_cols = [c for c in ind_cols if c.endswith("_score") or c.endswith("_raw")][:30]
            cols_keep = id_cols + ind_cols
            print(f"   Garde {len(id_cols)} ID cols + {len(ind_cols)} indicateurs")
        if cols_keep:
            chunks.append(ch[[c for c in cols_keep if c in ch.columns]])
    df = pd.concat(chunks, ignore_index=True)
    print(f"   Combined : {df.shape}")

    # Agréger par pays (gid_0)
    iso_col = next((c for c in df.columns if c.lower() in ("gid_0","iso","cntry_name")), None)
    if iso_col:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        agg = df.groupby(iso_col)[num_cols].mean().reset_index()
        agg = agg.rename(columns={iso_col: "ISO3"})
        if "ISO3" in agg.columns:
            agg["ISO"] = agg["ISO3"].apply(iso3_to_iso2)
            agg = agg.dropna(subset=["ISO"])
        agg["Annee"] = 2023
        rename = {c: f"aqueduct_{str(c).lower()[:30]}" for c in agg.columns
                   if c not in ["ISO","ISO3","Annee"]}
        agg = agg.rename(columns=rename)
        cols_out = ["ISO","Annee"] + sorted([c for c in agg.columns if c.startswith("aqueduct_")])
        agg[cols_out].to_csv(f"{CLN}/wri_aqueduct40_by_country.csv", index=False)
        print(f"   → {CLN}/wri_aqueduct40_by_country.csv ({len(agg)} pays × {len(cols_out)-2} indicateurs)")


# ════════════════════════════════════════════════════════════════════════
print("\n[6] 💦 AQUASTAT format m49 wide…")
# ════════════════════════════════════════════════════════════════════════
aqua_csv = f"{RAW}/AQUASTAT Dissemination System.csv"
if os.path.exists(aqua_csv):
    df = pd.read_csv(aqua_csv, low_memory=False)
    print(f"   {df.shape}, cols: {list(df.columns)[:10]}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Year","Value"])
    df["Year"] = df["Year"].astype(int)

    # Filtrer agrégats (IsAggregate=False = pays individuels)
    if "IsAggregate" in df.columns:
        df = df[df["IsAggregate"] == False]

    # Mapper Area → ISO
    df["ISO"] = df["Area"].apply(name_to_iso2)
    df = df.dropna(subset=["ISO"])

    # Garder top 30 variables
    top_vars = df["Variable"].value_counts().head(30).index
    df = df[df["Variable"].isin(top_vars)]

    piv = df.pivot_table(index=["ISO","Year"], columns="Variable",
                          values="Value", aggfunc="mean").reset_index()
    piv = piv.rename(columns={"Year":"Annee"})
    rename = {c: f"aquastat_{str(c).lower().replace(' ','_').replace(',','').replace('[','').replace(']','')[:35]}"
               for c in piv.columns if c not in ["ISO","Annee"]}
    piv = piv.rename(columns=rename)
    piv.to_csv(f"{CLN}/aquastat_top30_variables.csv", index=False)
    print(f"   → {CLN}/aquastat_top30_variables.csv ({len(piv)} lignes, {piv.shape[1]-2} variables)")


# ════════════════════════════════════════════════════════════════════════
print("\n[7] 💵 FAO Value of Production (format normalisé)…")
# ════════════════════════════════════════════════════════════════════════
fvp = f"{RAW}/fao_value_production/Value_of_Production_E_All_Data_NOFLAG.csv"
if os.path.exists(fvp):
    # Format wide avec colonnes Y1961 ... Y2023
    try:
        df = pd.read_csv(fvp, encoding="latin-1", low_memory=False)
    except:
        df = pd.read_csv(fvp, encoding="utf-8", low_memory=False)
    print(f"   {df.shape}, cols (15): {list(df.columns)[:15]}")

    year_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Y") and c[1:].isdigit()]
    print(f"   {len(year_cols)} year cols")

    if year_cols:
        # Filtrer Element = Gross Production Value en USD
        df_filt = df[df["Element"].str.contains("Gross Production Value.*current.*US", regex=True, na=False)]
        if len(df_filt) == 0:
            df_filt = df[df["Element"].str.contains("Gross Production Value.*constant.*US", regex=True, na=False)]
        print(f"   Filtré : {df_filt.shape}")

        # Long
        id_cols = [c for c in df_filt.columns if c not in year_cols]
        long = df_filt.melt(id_vars=["Area","Item"], value_vars=year_cols,
                              var_name="Annee", value_name="Value")
        long["Annee"] = long["Annee"].str.replace("Y","").astype(int)
        long["Value"] = pd.to_numeric(long["Value"], errors="coerce")
        long = long.dropna(subset=["Value"])

        # Agréger par pays/an (somme sur items)
        agg = long.groupby(["Area","Annee"]).agg(
            val_prod_total_kusd=("Value","sum"),
            val_prod_n_items=("Item","nunique"),
        ).reset_index()
        agg["ISO"] = agg["Area"].apply(name_to_iso2)
        agg = agg.dropna(subset=["ISO"])
        agg[["ISO","Annee","val_prod_total_kusd","val_prod_n_items"]].to_csv(
            f"{CLN}/fao_value_of_production.csv", index=False)
        print(f"   → {CLN}/fao_value_of_production.csv ({len(agg)} lignes)")


# ════════════════════════════════════════════════════════════════════════
print("\n[8] 📄 bulk_eng (AQUASTAT bulk)…")
# ════════════════════════════════════════════════════════════════════════
bulk = f"{RAW}/bulk_eng(in).csv"
if os.path.exists(bulk):
    df = pd.read_csv(bulk, low_memory=False)
    print(f"   {df.shape}")
    print(f"   Cols : {list(df.columns)[:6]}")
    # Format : aquastatElement, AREA, timePointYears, Value
    df["Year"] = pd.to_numeric(df["timePointYears"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Year","Value"])
    df["Year"] = df["Year"].astype(int)
    df["ISO"] = df["AREA"].apply(name_to_iso2)
    df = df.dropna(subset=["ISO"])
    # Pivot par variable
    var_col = "aquastatElement.1"  # nom textuel
    top_vars = df[var_col].value_counts().head(40).index
    df = df[df[var_col].isin(top_vars)]
    piv = df.pivot_table(index=["ISO","Year"], columns=var_col,
                          values="Value", aggfunc="mean").reset_index()
    piv = piv.rename(columns={"Year":"Annee"})
    rename = {c: f"aqua_bulk_{str(c).lower().replace(' ','_').replace('[','').replace(']','').replace(',','')[:35]}"
               for c in piv.columns if c not in ["ISO","Annee"]}
    piv = piv.rename(columns=rename)
    piv.to_csv(f"{CLN}/aquastat_bulk_eng.csv", index=False)
    print(f"   → {CLN}/aquastat_bulk_eng.csv ({len(piv)} lignes, {piv.shape[1]-2} variables)")


# ════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════════════════════════")
print("📊 BILAN FINAL — TOUS LES CLEANED")
print("══════════════════════════════════════════════════════════════")
keywords = ["edgar_2025","spam2020_v2","epi_2024","iucn","wri_aqueduct40","aquastat",
            "fao_value","aqua_bulk","fao_machinery","firms_active","psmsl_sea_level"]
for f in sorted(os.listdir(CLN)):
    if any(k in f for k in keywords):
        sz = os.path.getsize(f"{CLN}/{f}") / 1024
        print(f"  ✓ {f}  ({sz:.0f} KB)")
