"""process_new_data_v3.py — Fix EDGAR + SPAM + Aqueduct + bulk_eng."""
import os, sys, io, glob
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
    return get_english_iso(name)


# ════════════════════════════════════════════════════════════════════════
print("[1] 🌫️ EDGAR 2025 GHGs (skip 9 header rows EDGAR)…")
# ════════════════════════════════════════════════════════════════════════
edgar_xlsx = glob.glob(f"{RAW}/edgar_ghgs_2025/**/*.xlsx", recursive=True)
if edgar_xlsx:
    f = edgar_xlsx[0]
    # EDGAR format : 9 lignes de header notes, puis vraie table
    for sheet in ["Fossil CO2", "CH4 AR5", "N2O AR5", "F-gases AR5"]:
        try:
            df = pd.read_excel(f, sheet_name=sheet, skiprows=9, engine="openpyxl")
            print(f"   Sheet {sheet}: {df.shape}, cols={list(df.columns)[:8]}")
            # Trouver colonne country + year columns Y_xxxx
            year_cols = [c for c in df.columns if isinstance(c, (int, str)) and
                          str(c).startswith(("Y_","y_")) and str(c).replace("Y_","").replace("y_","").isdigit()]
            # Si pas de Y_ prefix, années seraient des int
            if not year_cols:
                year_cols = [c for c in df.columns if isinstance(c, int) and 1990 <= c <= 2030]
            # Si toujours pas, cols str ressemblant à années
            if not year_cols:
                year_cols = [c for c in df.columns if isinstance(c, str) and c.isdigit()
                              and 1990 <= int(c) <= 2030]
            print(f"     {len(year_cols)} year cols (sample : {year_cols[:5]})")
            if not year_cols: continue

            id_cols = [c for c in df.columns if c not in year_cols]
            country_col = next((c for c in id_cols if "Country" in str(c) or "Name" in str(c)
                                  or "ISO" in str(c)), id_cols[0] if id_cols else None)
            sector_col = next((c for c in id_cols if "Sector" in str(c) or "IPCC" in str(c)), None)
            print(f"     country={country_col}, sector={sector_col}")

            long = df.melt(id_vars=id_cols, value_vars=year_cols,
                            var_name="Annee", value_name="Value")
            long["Annee"] = long["Annee"].astype(str).str.replace("Y_","").str.replace("y_","")
            long["Annee"] = pd.to_numeric(long["Annee"], errors="coerce")
            long["Value"] = pd.to_numeric(long["Value"], errors="coerce")
            long = long.dropna(subset=["Annee","Value"])
            long["Annee"] = long["Annee"].astype(int)

            # ISO
            if country_col:
                if "ISO" in str(country_col).upper() or "CODE" in str(country_col).upper():
                    long["ISO"] = long[country_col].apply(
                        lambda c: iso3_to_iso2(c) if isinstance(c,str) and len(c)==3 else None)
                else:
                    long["ISO"] = long[country_col].apply(name_to_iso2)
                long = long.dropna(subset=["ISO"])

            # Agréger par sector ou prendre total
            if sector_col:
                long[sector_col] = long[sector_col].astype(str).str[:20]
                piv = long.pivot_table(index=["ISO","Annee"], columns=sector_col,
                                        values="Value", aggfunc="sum").reset_index()
                gas = sheet.split()[0].lower()
                rename = {c: f"edgar_{gas}_{str(c).lower().replace(' ','_').replace(',','')[:25]}"
                           for c in piv.columns if c not in ["ISO","Annee"]}
                piv = piv.rename(columns=rename)
            else:
                # Sum sur tout
                piv = long.groupby(["ISO","Annee"])["Value"].sum().reset_index()
                gas = sheet.split()[0].lower()
                piv = piv.rename(columns={"Value": f"edgar_{gas}_total"})

            out = f"{CLN}/edgar_2025_{sheet.lower().replace(' ','_')}.csv"
            piv.to_csv(out, index=False)
            print(f"     → {out} ({len(piv)} lignes)")
        except Exception as e:
            print(f"   ⚠️ Sheet {sheet}: {str(e)[:120]}")


# ════════════════════════════════════════════════════════════════════════
print("\n[2] 🌾 SPAM 2020 V2r2 yield (chunks limités)…")
# ════════════════════════════════════════════════════════════════════════
spam_csv = f"{RAW}/spam2020_v2r2_yield/spam2020V2r2_global_Y_TA.csv"
if os.path.exists(spam_csv):
    print(f"   Lecture (447MB) par chunks…")
    crop_cols = ['whea','rice','maiz','barl','mill','sorg','ocer','pota','swpo','yams',
                 'cass','orts','bean','chic','cowp','pige','lent','opul','soyb','grou',
                 'cnut','oilp','sunf','rape','sesa','ooil','sugc','sugb','cott','ofib',
                 'acof','rcof','coco','teas','toba','bana','plnt','trof','temf','vege','rest']

    # Approche directe : accumuler somme et compte par pays par chunk
    sum_dict = {}    # {country: {crop: sum}}
    count_dict = {}  # {country: {crop: count_non_zero}}

    chunk_idx = 0
    for chunk in pd.read_csv(spam_csv, usecols=["ADM0_NAME"] + crop_cols,
                              chunksize=100000, low_memory=False):
        chunk_idx += 1
        # Pour chaque pays présent dans le chunk
        for country, sub in chunk.groupby("ADM0_NAME"):
            if country not in sum_dict:
                sum_dict[country] = {c: 0.0 for c in crop_cols}
                count_dict[country] = {c: 0 for c in crop_cols}
            for c in crop_cols:
                vals = sub[c]
                vals_nz = vals[vals > 0]
                sum_dict[country][c] += vals_nz.sum()
                count_dict[country][c] += len(vals_nz)
        if chunk_idx % 5 == 0:
            print(f"   chunk {chunk_idx}…")

    print(f"   {chunk_idx} chunks traités")
    # Construire DataFrame final (moyenne pondérée)
    rows = []
    for country in sum_dict:
        row = {"ADM0_NAME": country}
        for c in crop_cols:
            count = count_dict[country][c]
            row[f"spam_yield_{c}"] = (sum_dict[country][c] / count) if count > 0 else np.nan
        rows.append(row)
    df_spam = pd.DataFrame(rows)
    df_spam["ISO"] = df_spam["ADM0_NAME"].apply(name_to_iso2)
    df_spam = df_spam.dropna(subset=["ISO"])
    df_spam["Annee"] = 2020
    cols_out = ["ISO","Annee"] + [c for c in df_spam.columns if c.startswith("spam_yield_")]
    df_spam[cols_out].to_csv(f"{CLN}/spam2020_v2_yield_by_country.csv", index=False)
    print(f"   → {CLN}/spam2020_v2_yield_by_country.csv ({len(df_spam)} pays × {len(crop_cols)} cultures)")


# ════════════════════════════════════════════════════════════════════════
print("\n[3] 💧 WRI Aqueduct 4.0 (chunks robuste)…")
# ════════════════════════════════════════════════════════════════════════
aq_csv = f"{RAW}/aqueduct_40/Aqueduct40_baseline_annual_y2023m07d05.csv"
if os.path.exists(aq_csv):
    # Lire juste la première ligne pour ID des cols
    head = pd.read_csv(aq_csv, nrows=2, low_memory=False)
    print(f"   Total cols : {len(head.columns)}")
    print(f"   Sample cols : {list(head.columns)[:25]}")

    id_cols = [c for c in head.columns if c.lower() in
                ("gid_0","gid_1","name_0","name_1","cntry_name","aq30_id")]
    # Tous les indicators _score (32 indicateurs typiques)
    ind_cols = [c for c in head.columns if c.endswith("_score") and len(c) < 25][:25]
    print(f"   Garde : {id_cols} + {len(ind_cols)} _score cols")

    keep = list(set(id_cols + ind_cols))
    chunks = []
    for ch in pd.read_csv(aq_csv, usecols=keep, chunksize=200000, low_memory=False):
        chunks.append(ch)
    df = pd.concat(chunks, ignore_index=True)
    print(f"   Combined : {df.shape}")

    iso_col = "gid_0" if "gid_0" in df.columns else next((c for c in id_cols if "iso" in c.lower() or "cntry" in c.lower()), None)
    if iso_col:
        num_cols = [c for c in df.columns if c.endswith("_score")]
        # Garder seulement valid scores (Aqueduct utilise -1 pour no data, 0-5 valid)
        df_v = df.copy()
        for c in num_cols:
            df_v.loc[df_v[c] < 0, c] = np.nan
        agg = df_v.groupby(iso_col)[num_cols].mean().reset_index()
        agg = agg.rename(columns={iso_col: "ISO3"})
        agg["ISO"] = agg["ISO3"].apply(iso3_to_iso2)
        agg = agg.dropna(subset=["ISO"])
        agg["Annee"] = 2023
        rename = {c: f"aqueduct_{c.replace('_score','')[:30]}" for c in num_cols}
        agg = agg.rename(columns=rename)
        cols_out = ["ISO","Annee"] + [c for c in agg.columns if c.startswith("aqueduct_")]
        agg[cols_out].to_csv(f"{CLN}/wri_aqueduct40_by_country.csv", index=False)
        print(f"   → {CLN}/wri_aqueduct40_by_country.csv ({len(agg)} pays × {len(num_cols)} indicateurs)")


# ════════════════════════════════════════════════════════════════════════
print("\n[4] 📄 bulk_eng (encoding latin-1)…")
# ════════════════════════════════════════════════════════════════════════
bulk = f"{RAW}/bulk_eng(in).csv"
if os.path.exists(bulk):
    df = pd.read_csv(bulk, encoding="latin-1", low_memory=False)
    print(f"   {df.shape}, cols (6): {list(df.columns)[:6]}")
    df["Year"] = pd.to_numeric(df["timePointYears"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Year","Value"])
    df["Year"] = df["Year"].astype(int)
    df["ISO"] = df["AREA"].apply(name_to_iso2)
    df = df.dropna(subset=["ISO"])
    var_col = "aquastatElement.1"
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
print("📊 BILAN FINAL")
print("══════════════════════════════════════════════════════════════")
keywords = ["edgar_2025","spam2020_v2","epi_2024","iucn","wri_aqueduct40","aquastat",
            "fao_value","aqua_bulk","fao_machinery","firms_active","psmsl_sea_level"]
for f in sorted(os.listdir(CLN)):
    if any(k in f for k in keywords):
        sz = os.path.getsize(f"{CLN}/{f}") / 1024
        print(f"  ✓ {f}  ({sz:.0f} KB)")
