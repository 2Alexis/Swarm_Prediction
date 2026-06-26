"""
process_user_datasets.py — Traite les datasets téléchargés avec succès.

Catégories traitées :
  - FAO Fertilizers by Nutrient (N, P, K séparés)
  - FAO Pesticides Use (par catégorie)
  - FAO Land Use Inputs
  - Ember Electricity (yearly + monthly)
  - WB SE4ALL (renewable %, electricity access)
  - Re-tentatives intelligentes pour les URLs cassées
"""
import os, sys, io, zipfile, urllib.request, json
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

D = "data/cleaned"
RAW_EXTRA = "data/raw"


# ── 1. FAO Fertilizers by Nutrient ────────────────────────────────────────
print("[1] FAO Fertilizers by Nutrient (N, P, K)…")
zf = "data/raw/agriculture_extras/FAO_fertilizers_nutrient.zip"
if os.path.exists(zf):
    with zipfile.ZipFile(zf) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(csv_name) as f:
            try:
                df = pd.read_csv(f, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                f.close()
                with z.open(csv_name) as f2:
                    df = pd.read_csv(f2, encoding="latin-1", low_memory=False)
    print(f"   Loaded {csv_name}: {df.shape}")
    print(f"   Cols : {list(df.columns)[:10]}")

    # Elements + Items pour comprendre
    if "Element" in df.columns:
        print(f"   Elements: {df['Element'].unique()[:10]}")
    if "Item" in df.columns:
        print(f"   Items: {df['Item'].unique()[:10]}")

    # Filtrer pour Agricultural Use uniquement (kg/ha si dispo) et 3 nutriments
    el_col = "Element"
    item_col = "Item"
    if el_col in df.columns and item_col in df.columns:
        # Garder usage agricole
        df_filt = df[df[el_col].str.contains("Agricultural Use", case=False, na=False)]
        print(f"   After 'Agricultural Use' filter: {df_filt.shape}")

        # Mapper Item → nutriment
        def nutrient(item):
            it = str(item).lower()
            if "nitrog" in it or " n " in it or "nitrate" in it: return "N"
            if "phospha" in it or " p " in it: return "P"
            if "potass" in it or " k " in it: return "K"
            return "Other"
        df_filt["Nutrient"] = df_filt[item_col].apply(nutrient)
        df_filt = df_filt[df_filt["Nutrient"] != "Other"]

        # Pivot par (pays, année, nutriment)
        area_col = "Area" if "Area" in df_filt.columns else "Country"
        year_col = "Year"
        val_col = "Value"
        df_filt[val_col] = pd.to_numeric(df_filt[val_col], errors="coerce")
        df_filt = df_filt.dropna(subset=[val_col])

        piv = df_filt.pivot_table(index=[area_col, year_col],
                                    columns="Nutrient",
                                    values=val_col,
                                    aggfunc="sum").reset_index()
        piv.columns = ["Pays", "Annee", "fertilizer_K_t", "fertilizer_N_t", "fertilizer_P_t"][:len(piv.columns)]
        piv.to_csv(f"{D}/fao_fertilizers_NPK.csv", index=False)
        print(f"   → {D}/fao_fertilizers_NPK.csv ({len(piv)} lignes, N/P/K séparés)")
else:
    print("   ✗ fichier zip absent")


# ── 2. FAO Pesticides Use ─────────────────────────────────────────────────
print("\n[2] FAO Pesticides Use…")
zf = "data/raw/agriculture_extras/FAO_pesticides_use.zip"
if os.path.exists(zf):
    with zipfile.ZipFile(zf) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(csv_name) as f:
            try:
                df = pd.read_csv(f, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                f.close()
                with z.open(csv_name) as f2:
                    df = pd.read_csv(f2, encoding="latin-1", low_memory=False)
    print(f"   Loaded {csv_name}: {df.shape}")
    if "Item" in df.columns:
        print(f"   Items uniques: {df['Item'].nunique()}")
        print(f"   Top items: {df['Item'].value_counts().head(8).to_dict()}")

    # Mapper Item → catégorie pesticide
    def pest_cat(item):
        it = str(item).lower()
        if "insectic" in it: return "Insecticide"
        if "herbic" in it: return "Herbicide"
        if "fungic" in it or "bacter" in it: return "Fungicide"
        if "rodent" in it: return "Rodenticide"
        if "total" in it or "pesticide" in it: return "Total"
        return "Other"
    df["Category"] = df["Item"].apply(pest_cat) if "Item" in df.columns else "Total"

    # Pivot par catégorie
    area_col = "Area" if "Area" in df.columns else "Country"
    year_col = "Year"
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df_filt = df.dropna(subset=["Value"])
    df_filt = df_filt[df_filt["Element"].str.contains("Use", case=False, na=False)] if "Element" in df_filt.columns else df_filt

    piv = df_filt.pivot_table(index=[area_col, year_col],
                                columns="Category", values="Value",
                                aggfunc="sum").reset_index()
    piv = piv.rename(columns={area_col: "Pays", year_col: "Annee"})
    rename_pest = {c: f"pest_{c.lower()}_t" for c in piv.columns if c not in ["Pays", "Annee"]}
    piv = piv.rename(columns=rename_pest)
    piv.to_csv(f"{D}/fao_pesticides_categories.csv", index=False)
    print(f"   → {D}/fao_pesticides_categories.csv ({len(piv)} lignes)")
else:
    print("   ✗ fichier zip absent")


# ── 3. FAO Land Use Inputs ────────────────────────────────────────────────
print("\n[3] FAO Land Use Inputs…")
zf = "data/raw/agriculture_extras/FAO_land_use_inputs.zip"
if os.path.exists(zf):
    with zipfile.ZipFile(zf) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(csv_name) as f:
            try:
                df = pd.read_csv(f, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                f.close()
                with z.open(csv_name) as f2:
                    df = pd.read_csv(f2, encoding="latin-1", low_memory=False)
    print(f"   Loaded {csv_name}: {df.shape}")
    if "Item" in df.columns:
        print(f"   Top items: {df['Item'].value_counts().head(8).to_dict()}")

    # Filtrer items intéressants
    interesting = ["Arable land", "Permanent crops", "Forest land", "Permanent meadows",
                    "Other land", "Cropland", "Agricultural land",
                    "Land area equipped for irrigation"]
    if "Item" in df.columns:
        df_filt = df[df["Item"].isin(interesting)].copy()
    else:
        df_filt = df.copy()

    area_col = "Area" if "Area" in df.columns else "Country"
    df_filt["Value"] = pd.to_numeric(df_filt["Value"], errors="coerce")
    df_filt = df_filt.dropna(subset=["Value"])

    piv = df_filt.pivot_table(index=[area_col, "Year"], columns="Item",
                                values="Value", aggfunc="sum").reset_index()
    piv.columns = [c if c in [area_col, "Year"] else f"landuse_{c.lower().replace(' ','_')[:30]}_kha"
                    for c in piv.columns]
    piv = piv.rename(columns={area_col: "Pays", "Year": "Annee"})
    piv.to_csv(f"{D}/fao_landuse_inputs.csv", index=False)
    print(f"   → {D}/fao_landuse_inputs.csv ({len(piv)} lignes)")
else:
    print("   ✗ fichier zip absent")


# ── 4. Ember Electricity (yearly) ─────────────────────────────────────────
print("\n[4] Ember Electricity Yearly…")
emb = "data/raw/energie_extras/ember_yearly_electricity.csv"
if os.path.exists(emb):
    df = pd.read_csv(emb, low_memory=False)
    print(f"   Shape: {df.shape}")
    print(f"   Cols: {list(df.columns)[:12]}")
    print(f"   Categories: {df.get('Category', pd.Series(['?'])).value_counts().head().to_dict()}")
    print(f"   Variables: {df.get('Variable', pd.Series(['?'])).unique()[:15]}")

    # Filtrer génération électrique par source
    df_filt = df[(df.get("Category", "") == "Electricity generation") &
                  (df.get("Unit", "") == "TWh")] if "Category" in df.columns else df

    if len(df_filt):
        country_col = "Area" if "Area" in df.columns else "Country"
        piv = df_filt.pivot_table(index=[country_col, "Year"], columns="Variable",
                                     values="Value", aggfunc="sum").reset_index()
        piv = piv.rename(columns={country_col: "Pays", "Year": "Annee"})
        rename_cols = {c: f"ember_{c.lower().replace(' ','_')[:25]}_twh"
                        for c in piv.columns if c not in ["Pays", "Annee"]}
        piv = piv.rename(columns=rename_cols)
        piv.to_csv(f"{D}/ember_electricity_by_source.csv", index=False)
        print(f"   → {D}/ember_electricity_by_source.csv ({len(piv)} lignes, {piv.shape[1]-2} sources)")
    else:
        print("   ✗ Format inattendu")
else:
    print("   ✗ fichier absent")


# ── 5. WB SE4ALL (renewable + access) ─────────────────────────────────────
print("\n[5] WB SE4ALL — renewable share + electricity access…")
for src, name in [
    ("wb_renewable_pct.json",       "wb_renewable_elec_pct"),
    ("wb_renewable_final_pct.json", "wb_renewable_final_pct"),
    ("wb_elec_access.json",         "wb_elec_access_pct"),
]:
    p = f"data/raw/energie_extras/{src}"
    if not os.path.exists(p): continue
    with open(p) as f:
        data = json.load(f)
    if len(data) < 2: continue
    rows = []
    for r in data[1]:
        if r.get("value") is None: continue
        rows.append({"Pays": r["country"]["value"], "ISO3": r["countryiso3code"],
                     "Annee": int(r["date"]), "Valeur": float(r["value"])})
    df = pd.DataFrame(rows).rename(columns={"Valeur": name})
    if len(df) > 0:
        df[["Pays", "Annee", name]].to_csv(f"{D}/{name}.csv", index=False)
        print(f"   → {D}/{name}.csv ({len(df)} lignes)")


# ── 6. Re-tentatives URLs critiques ───────────────────────────────────────
print("\n[6] Re-tentatives URLs critiques…")

retries = [
    # EDGAR — vraies URLs depuis la doc
    ("https://edgar.jrc.ec.europa.eu/php/downl_pubdata.php?dataset=v90_GHG", "EDGAR webform", "atmosphere_extras"),
    # WRI Aqueduct — URL alternative
    ("https://files.wri.org/d8/s3fs-public/aqueduct-30-summary-rankings-2019.xlsx", "WRI Aqueduct rankings", "hydrologie_extras"),
    # Yale EPI 2024 — vraie URL
    ("https://epi.yale.edu/measure/2024/EPI", "Yale EPI page", "sol_ecologie_extras"),
    ("https://epi.yale.edu/downloads", "Yale EPI downloads", "sol_ecologie_extras"),
    # GFW alternative
    ("https://data.globalforestwatch.org/datasets/63f9425c45404c36a23495ed7bef1314_2.csv", "GFW countries tree loss", "sol_ecologie_extras"),
    # FAO Machinery alternative
    ("https://bulks-faostat.fao.org/production/Inputs_AgriculturalMachinery_E_All_Data_(Normalized).zip", "FAO Machinery alt", "agriculture_extras"),
]
for url, name, cat in retries:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
            if len(data) > 5000:  # >5KB pour considérer comme valide
                fname = os.path.basename(url).split("?")[0] or f"{name}.bin"
                out = f"data/raw/{cat}/{fname}"
                with open(out, "wb") as f:
                    f.write(data)
                print(f"   ✓ {name} ({len(data)/1024:.0f}KB) → {out}")
            else:
                print(f"   ✗ {name} : réponse trop petite ({len(data)} bytes)")
    except Exception as e:
        print(f"   ✗ {name} : {str(e)[:60]}")


print("\n\n══════════════════════════════════════════════════════════════")
print("📊 BILAN PROCESSING")
print("══════════════════════════════════════════════════════════════")
new_files = []
for f in os.listdir(D):
    if any(k in f for k in ["fao_fertilizers", "fao_pesticides", "fao_landuse",
                              "ember_electricity", "wb_renewable", "wb_elec_access"]):
        sz = os.path.getsize(f"{D}/{f}") / 1024
        new_files.append((f, sz))
        print(f"  ✓ {f} ({sz:.0f} KB)")
print(f"\n{len(new_files)} nouveaux fichiers cleaned créés")
