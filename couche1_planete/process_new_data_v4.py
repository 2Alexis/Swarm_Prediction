"""process_new_data_v4.py — Fix final EDGAR + SPAM + Aqueduct (chemins corrigés)."""
import os, sys, io, glob
import pandas as pd
import numpy as np
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")
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
print("[1] 🌫️ EDGAR (skiprows=6, header propre)…")
# ════════════════════════════════════════════════════════════════════════
edgar_xlsx = glob.glob(f"{RAW}/edgar_ghgs_2025/**/*.xlsx", recursive=True)
if edgar_xlsx:
    f = edgar_xlsx[0]
    for sheet in ["Fossil CO2", "CH4 AR5", "N2O AR5", "F-gases AR5"]:
        try:
            df = pd.read_excel(f, sheet_name=sheet, skiprows=6, engine="openpyxl")
            print(f"   {sheet}: {df.shape}, cols: {list(df.columns)[:8]}")

            year_cols = [c for c in df.columns if isinstance(c, str)
                          and c.startswith("Y_") and c[2:].isdigit()]
            print(f"     {len(year_cols)} year cols")

            if year_cols and "ISO" in df.columns and "Sector" in df.columns:
                # Long
                long = df.melt(id_vars=["ISO","Country","Sector"], value_vars=year_cols,
                                var_name="Annee", value_name="Value")
                long["Annee"] = long["Annee"].str.replace("Y_","").astype(int)
                long["Value"] = pd.to_numeric(long["Value"], errors="coerce")
                long = long.dropna(subset=["Value"])

                # ISO3 → ISO2
                long["ISO2"] = long["ISO"].apply(iso3_to_iso2)
                long = long.dropna(subset=["ISO2"])

                # Pivot par secteur (sum sur sub-national)
                piv = long.pivot_table(index=["ISO2","Annee"], columns="Sector",
                                        values="Value", aggfunc="sum").reset_index()
                piv = piv.rename(columns={"ISO2":"ISO"})

                gas = sheet.replace(" AR5","").replace(" ","_").lower()
                rename = {c: f"edgar_{gas}_{str(c).lower().replace(' ','_')[:25]}"
                           for c in piv.columns if c not in ["ISO","Annee"]}
                piv = piv.rename(columns=rename)
                out = f"{CLN}/edgar_2025_{gas}.csv"
                piv.to_csv(out, index=False)
                print(f"     → {out} ({len(piv)} lignes × {piv.shape[1]-2} secteurs)")
        except Exception as e:
            print(f"   ⚠️ {sheet}: {str(e)[:120]}")


# ════════════════════════════════════════════════════════════════════════
print("\n[2] 🌾 SPAM 2020 V2r2 — chemin corrigé…")
# ════════════════════════════════════════════════════════════════════════
spam_csv = f"{RAW}/spam2020_v2r2_yield/spam2020V2r2_global_yield/spam2020V2r2_global_Y_TA.csv"
print(f"   Path: {spam_csv}")
print(f"   Exists: {os.path.exists(spam_csv)}")

if os.path.exists(spam_csv):
    crop_cols = ['whea','rice','maiz','barl','mill','sorg','ocer','pota','swpo','yams',
                 'cass','orts','bean','chic','cowp','pige','lent','opul','soyb','grou',
                 'cnut','oilp','sunf','rape','sesa','ooil','sugc','sugb','cott','ofib',
                 'acof','rcof','coco','teas','toba','bana','plnt','trof','temf','vege','rest']

    sum_dict, count_dict = {}, {}
    chunk_idx = 0
    for chunk in pd.read_csv(spam_csv, usecols=["ADM0_NAME"] + crop_cols,
                              chunksize=100000, low_memory=False):
        chunk_idx += 1
        for country, sub in chunk.groupby("ADM0_NAME"):
            if country not in sum_dict:
                sum_dict[country] = {c: 0.0 for c in crop_cols}
                count_dict[country] = {c: 0 for c in crop_cols}
            for c in crop_cols:
                vals = sub[c]
                vals_nz = vals[vals > 0]
                sum_dict[country][c] += vals_nz.sum()
                count_dict[country][c] += len(vals_nz)
        if chunk_idx % 3 == 0:
            print(f"     chunk {chunk_idx}…")
    print(f"   {chunk_idx} chunks traités")

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
print("\n[3] 💧 WRI Aqueduct 4.0 — chemin corrigé…")
# ════════════════════════════════════════════════════════════════════════
aq_csv = f"{RAW}/aqueduct_40/Aqueduct40_waterrisk_download_Y2023M07D05/CVS/Aqueduct40_baseline_annual_y2023m07d05.csv"
print(f"   Path: {aq_csv}")
print(f"   Exists: {os.path.exists(aq_csv)}")

if os.path.exists(aq_csv):
    head = pd.read_csv(aq_csv, nrows=2, low_memory=False)
    print(f"   Total cols: {len(head.columns)}, sample: {list(head.columns)[:18]}")

    # Aqueduct 4.0 : cols principales gid_0, gid_1, name_0, name_1
    # Indicateurs : bws_cat, bwd_cat, gtd_cat, rfr_cat, drr_cat, etc. + _label + _score
    id_cols = [c for c in head.columns if c in ("gid_0","gid_1","name_0","name_1","aq30_id")]
    # Score columns (float)
    ind_cols = [c for c in head.columns if c.endswith(("_score","_raw","_cat")) and len(c) < 28]
    # Limiter
    ind_cols = ind_cols[:40]
    print(f"   Keep: {id_cols} + {len(ind_cols)} indicateurs")

    keep = list(set(id_cols + ind_cols))
    chunks = []
    for ch in pd.read_csv(aq_csv, usecols=keep, chunksize=200000, low_memory=False):
        chunks.append(ch)
    df = pd.concat(chunks, ignore_index=True)
    print(f"   Combined: {df.shape}")

    iso_col = "gid_0" if "gid_0" in df.columns else None
    if iso_col:
        # Garder seulement scores numériques
        num_cols = [c for c in df.columns if c.endswith(("_score","_raw"))]
        df_v = df.copy()
        for c in num_cols:
            df_v[c] = pd.to_numeric(df_v[c], errors="coerce")
            df_v.loc[df_v[c] < 0, c] = np.nan
        agg = df_v.groupby(iso_col)[num_cols].mean().reset_index()
        agg = agg.rename(columns={iso_col: "ISO3"})
        agg["ISO"] = agg["ISO3"].apply(iso3_to_iso2)
        agg = agg.dropna(subset=["ISO"])
        agg["Annee"] = 2023
        rename = {c: f"aqueduct_{c[:30]}" for c in num_cols}
        agg = agg.rename(columns=rename)
        cols_out = ["ISO","Annee"] + [c for c in agg.columns if c.startswith("aqueduct_")]
        agg[cols_out].to_csv(f"{CLN}/wri_aqueduct40_by_country.csv", index=False)
        print(f"   → {CLN}/wri_aqueduct40_by_country.csv ({len(agg)} pays × {len(num_cols)} indicateurs)")


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
