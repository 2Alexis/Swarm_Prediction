"""process_spam_aqueduct.py — Traite SPAM V2 + Aqueduct (chemins corrigés, cols dynamiques)."""
import os, sys, io, glob
import pandas as pd
import numpy as np
import warnings

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
print("[1] 🌾 SPAM 2020 V2r2 (cols dynamiques)…")
# ════════════════════════════════════════════════════════════════════════
spam_csv = f"{RAW}/spam2020_v2r2_yield/spam2020V2r2_global_yield/spam2020V2r2_global_Y_TA.csv"
if os.path.exists(spam_csv):
    head = pd.read_csv(spam_csv, nrows=1)
    crop_cols = [c for c in head.columns if c not in
                  ("grid_code","x","y","FIPS0","FIPS1","FIPS2","ADM0_NAME","ADM1_NAME","ADM2_NAME")]
    print(f"   {len(crop_cols)} cultures : {crop_cols}")

    sum_d, cnt_d = {}, {}
    chunk_idx = 0
    for chunk in pd.read_csv(spam_csv, usecols=["ADM0_NAME"]+crop_cols,
                              chunksize=100000, low_memory=False):
        chunk_idx += 1
        for country, sub in chunk.groupby("ADM0_NAME"):
            if country not in sum_d:
                sum_d[country] = {c: 0.0 for c in crop_cols}
                cnt_d[country] = {c: 0 for c in crop_cols}
            for c in crop_cols:
                vals = sub[c]
                vals_nz = vals[vals > 0]
                sum_d[country][c] += vals_nz.sum()
                cnt_d[country][c] += len(vals_nz)
        if chunk_idx % 3 == 0: print(f"     chunk {chunk_idx}…")
    print(f"   {chunk_idx} chunks traités, {len(sum_d)} pays")

    rows = []
    for country in sum_d:
        row = {"ADM0_NAME": country}
        for c in crop_cols:
            cnt = cnt_d[country][c]
            row[f"spam_yield_{c}"] = (sum_d[country][c] / cnt) if cnt > 0 else np.nan
        rows.append(row)
    df_spam = pd.DataFrame(rows)
    df_spam["ISO"] = df_spam["ADM0_NAME"].apply(name_to_iso2)
    df_spam = df_spam.dropna(subset=["ISO"])
    df_spam["Annee"] = 2020
    cols_out = ["ISO","Annee"] + [c for c in df_spam.columns if c.startswith("spam_yield_")]
    df_spam[cols_out].to_csv(f"{CLN}/spam2020_v2_yield_by_country.csv", index=False)
    print(f"   → {CLN}/spam2020_v2_yield_by_country.csv ({len(df_spam)} pays × {len(crop_cols)} cultures)")


# ════════════════════════════════════════════════════════════════════════
print("\n[2] 💧 WRI Aqueduct 4.0 baseline annual…")
# ════════════════════════════════════════════════════════════════════════
aq_csv = f"{RAW}/aqueduct_40/Aqueduct40_waterrisk_download_Y2023M07D05/CVS/Aqueduct40_baseline_annual_y2023m07d05.csv"
if os.path.exists(aq_csv):
    head = pd.read_csv(aq_csv, nrows=2, low_memory=False)
    print(f"   {len(head.columns)} cols total")

    id_cols = [c for c in head.columns if c in ("gid_0","gid_1","name_0","name_1","aq30_id")]
    ind_cols = [c for c in head.columns if c.endswith(("_score","_raw")) and len(c) < 28]
    ind_cols = ind_cols[:40]
    keep = list(set(id_cols + ind_cols))
    print(f"   Keep {len(id_cols)} id + {len(ind_cols)} indicateurs")

    chunks = []
    for ch in pd.read_csv(aq_csv, usecols=keep, chunksize=200000, low_memory=False):
        chunks.append(ch)
    df = pd.concat(chunks, ignore_index=True)
    print(f"   Combined: {df.shape}")

    iso_col = "gid_0" if "gid_0" in df.columns else None
    if iso_col:
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
        print(f"   → {CLN}/wri_aqueduct40_by_country.csv ({len(agg)} pays × {len(num_cols)} ind)")


print("\n══════════════════════════════════════════════════════════════")
print("📊 BILAN — TOUS LES CLEANED")
print("══════════════════════════════════════════════════════════════")
keywords = ["edgar_2025","spam2020_v2","epi_2024","iucn","wri_aqueduct40","aquastat",
            "fao_value","aqua_bulk","fao_machinery","firms_active","psmsl_sea_level"]
for f in sorted(os.listdir(CLN)):
    if any(k in f for k in keywords):
        sz = os.path.getsize(f"{CLN}/{f}") / 1024
        print(f"  ✓ {f}  ({sz:.0f} KB)")
