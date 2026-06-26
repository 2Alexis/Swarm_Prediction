"""
clean_fish.py — Nettoyage du dataset FAO Global Production Fish 2024.1.0

Pipeline :
  1. Charger Global_production_quantity.csv (1.15M lignes brutes)
  2. Filtrer MEASURE=Q_tlw (poids vivant en tonnes)
  3. Joindre UN_Code → ISO2_Code (codes pays)
  4. Joindre SPECIES.ALPHA_3_CODE → ISSCAAP_Group + Scientific_Name
  5. Joindre PRODUCTION_SOURCE_DET → libellés (capture, aquaculture marine/eau douce/saumâtre)
  6. Détecter et winsorizer les outliers (P99.5)
  7. Agrégations multiples :
     - par (ISO, Année, Source)            → tonnes par source
     - par (ISO, Année)                    → total + diversité
     - par (ISO, Année, ISSCAAP_Group)     → top groupes par pays
  8. Sauvegarder en CSV pour intégration dans dataset_final_v13_couche1
"""
import os, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
RAW_DIR  = "data/raw/GlobalProductionFish_2024.1.0"
OUT_DIR  = "data/cleaned"
os.makedirs(OUT_DIR, exist_ok=True)


# ── 1. Chargement données brutes ──────────────────────────────────────────
print("[1] Chargement Global_production_quantity.csv…")
df = pd.read_csv(f"{RAW_DIR}/Global_production_quantity.csv", low_memory=False)
print(f"   {len(df):,} lignes brutes, années {df['PERIOD'].min()}–{df['PERIOD'].max()}")


# ── 2. Filtrage MEASURE=Q_tlw (tonnes poids vivant) ────────────────────────
print("\n[2] Filtrage MEASURE=Q_tlw (tonnes)…")
before = len(df)
df = df[df["MEASURE"] == "Q_tlw"].copy()
print(f"   {len(df):,} lignes après filtre (était {before:,})")
df = df.drop(columns=["MEASURE"])


# ── 3. Jointure pays UN → ISO2 ────────────────────────────────────────────
print("\n[3] Jointure pays (UN_Code → ISO2)…")
cn = pd.read_csv(f"{RAW_DIR}/CL_FI_COUNTRY_GROUPS.csv", low_memory=False)
cn = cn[["UN_Code", "ISO2_Code", "ISO3_Code", "Name_En",
         "Continent_Group_En", "EcoClass_Group_En"]].rename(columns={
            "UN_Code": "COUNTRY.UN_CODE",
            "ISO2_Code": "ISO",
            "ISO3_Code": "ISO3",
            "Name_En": "Pays",
            "Continent_Group_En": "Continent",
            "EcoClass_Group_En": "EcoClass"
         })
df = df.merge(cn, on="COUNTRY.UN_CODE", how="left")
n_no_iso = df["ISO"].isna().sum()
print(f"   {n_no_iso:,} lignes sans ISO ({n_no_iso/len(df)*100:.1f}%) — agrégats régionaux FAO")
df = df.dropna(subset=["ISO"])
print(f"   {len(df):,} lignes avec ISO valide, {df['ISO'].nunique()} pays")


# ── 4. Jointure espèces (3A_Code → ISSCAAP + Name) ────────────────────────
print("\n[4] Jointure espèces…")
sp = pd.read_csv(f"{RAW_DIR}/CL_FI_SPECIES_GROUPS.csv", low_memory=False)
sp = sp[["3A_Code", "Name_En", "Scientific_Name", "Major_Group",
         "ISSCAAP_Group_En"]].rename(columns={
            "3A_Code": "SPECIES.ALPHA_3_CODE",
            "Name_En": "Espece_En",
            "Major_Group": "MajorGroup",
            "ISSCAAP_Group_En": "ISSCAAP"
         })
df = df.merge(sp, on="SPECIES.ALPHA_3_CODE", how="left")
print(f"   {df['ISSCAAP'].isna().sum():,} sans ISSCAAP → comblées avec 'Unknown'")
df["ISSCAAP"] = df["ISSCAAP"].fillna("Unknown")
df["MajorGroup"] = df["MajorGroup"].fillna("Unknown")


# ── 5. Jointure source production (CAPTURE / AQUACULTURE) ─────────────────
print("\n[5] Jointure source production…")
ps = pd.read_csv(f"{RAW_DIR}/CL_FI_PRODUCTION_SOURCE_DET.csv", low_memory=False)
ps = ps[["Code", "Name_En"]].rename(columns={"Code": "PRODUCTION_SOURCE_DET.CODE",
                                              "Name_En": "Source"})
df = df.merge(ps, on="PRODUCTION_SOURCE_DET.CODE", how="left")
# Simplifier
def simplify_source(s):
    if pd.isna(s): return "Unknown"
    s = str(s).lower()
    if "capture" in s: return "Capture"
    if "freshwater" in s: return "Aquaculture_Freshwater"
    if "brackish" in s: return "Aquaculture_Brackish"
    if "marine" in s: return "Aquaculture_Marine"
    return "Other"
df["SourceType"] = df["Source"].apply(simplify_source)
print(f"   Sources : {df['SourceType'].value_counts().to_dict()}")


# ── 6. Détection outliers ─────────────────────────────────────────────────
print("\n[6] Outliers détectés…")
print(f"   Valeurs > P99.9 : {(df['VALUE'] > df['VALUE'].quantile(0.999)).sum():,}")
p99 = df["VALUE"].quantile(0.999)
print(f"   P99.9 = {p99:,.0f} tonnes")
print(f"   Max observé = {df['VALUE'].max():,.0f} tonnes")
# Pas de winsorize : les vraies productions massives (anchois pérou, etc.) sont valides
# On filtre seulement les zéros pour les agrégations
df_nonzero = df[df["VALUE"] > 0].copy()
print(f"   {len(df_nonzero):,} lignes avec VALUE > 0 ({len(df_nonzero)/len(df)*100:.1f}%)")


# ── 7. Renommage final ────────────────────────────────────────────────────
df = df.rename(columns={
    "PERIOD": "Annee",
    "VALUE": "Tonnes",
    "AREA.CODE": "Zone_FAO",
})
df = df[["ISO", "ISO3", "Pays", "Continent", "EcoClass", "Annee",
         "SPECIES.ALPHA_3_CODE", "Espece_En", "Scientific_Name", "MajorGroup", "ISSCAAP",
         "Zone_FAO", "SourceType", "Source", "Tonnes", "STATUS"]]
df.to_csv(f"{OUT_DIR}/fish_production_raw_clean.csv", index=False)
print(f"\n[OK] {OUT_DIR}/fish_production_raw_clean.csv (lignes complètes : {len(df):,})")


# ── 8. Agrégations utiles ──────────────────────────────────────────────────
print("\n[8] Agrégations par (ISO, Année)…")

# A. Par source → tonnes par catégorie
df_pos = df[df["Tonnes"] > 0]
by_src = df_pos.groupby(["ISO", "Annee", "SourceType"])["Tonnes"].sum().unstack(fill_value=0).reset_index()
by_src.columns = ["ISO", "Annee"] + [f"fish_{c}_t" for c in by_src.columns[2:]]
print(f"   Par source : {by_src.shape}")
by_src.to_csv(f"{OUT_DIR}/fish_production_by_source.csv", index=False)

# B. Total + diversité espèces
by_total = df_pos.groupby(["ISO", "Annee"]).agg(
    fish_total_t=("Tonnes", "sum"),
    fish_species_count=("SPECIES.ALPHA_3_CODE", "nunique"),
    fish_isscaap_count=("ISSCAAP", "nunique"),
).reset_index()
print(f"   Totaux : {by_total.shape}")
by_total.to_csv(f"{OUT_DIR}/fish_production_total.csv", index=False)

# C. Top ISSCAAP groups par pays/année (wide)
isscaap_top = (df_pos.groupby(["ISO", "Annee", "ISSCAAP"])["Tonnes"].sum()
                .unstack(fill_value=0).reset_index())
# Garder seulement les 10 groupes ISSCAAP les plus produits mondialement
top_groups = df_pos.groupby("ISSCAAP")["Tonnes"].sum().sort_values(ascending=False).head(10).index
keep_cols = ["ISO", "Annee"] + [g for g in top_groups if g in isscaap_top.columns]
isscaap_top = isscaap_top[keep_cols].copy()
# Nommer proprement
def safe_col(s):
    return "fish_isscaap_" + "".join(c if c.isalnum() else "_" for c in s.lower())[:40]
isscaap_top.columns = ["ISO", "Annee"] + [safe_col(c) for c in isscaap_top.columns[2:]]
print(f"   Top 10 ISSCAAP : {isscaap_top.shape}")
isscaap_top.to_csv(f"{OUT_DIR}/fish_production_by_isscaap.csv", index=False)


# ── 9. Stats finales ──────────────────────────────────────────────────────
print("\n[9] Stats finales :")
print(f"   Pays couverts : {by_total['ISO'].nunique()}")
print(f"   Années : {by_total['Annee'].min()}–{by_total['Annee'].max()}")
print(f"   Production totale 2022 : {by_total[by_total['Annee']==2022]['fish_total_t'].sum()/1e6:.1f} M tonnes")
print(f"   Top 10 pays 2022 (M tonnes) :")
top2022 = by_total[by_total["Annee"]==2022].sort_values("fish_total_t", ascending=False).head(10)
for _, r in top2022.iterrows():
    print(f"     {r['ISO']}: {r['fish_total_t']/1e6:.2f} M t  ({int(r['fish_species_count'])} espèces)")

print(f"\n   Outputs créés :")
for f in ["fish_production_raw_clean.csv","fish_production_by_source.csv",
          "fish_production_total.csv","fish_production_by_isscaap.csv"]:
    p = f"{OUT_DIR}/{f}"
    sz = os.path.getsize(p) / 1e6
    print(f"     {f}  ({sz:.1f} MB)")
