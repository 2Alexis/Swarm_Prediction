"""
enrich_v5.py — dataset_final_v5.csv → dataset_final_v6.csv

  + Rendements par culture spécifique (soja, colza, palmier, pomme, banane, etc.)
  + CRU TS / OWID temp anomalies historiques
  + Catastrophes ventilées par type (drought/flood/storm/wildfire/earthquake/...)
  + Variables extras OWID (undernourishment, mental health, refugees, IDPs...)
"""
import os, sys, io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = "data/cleaned"

from build_dataset import custom_mappings, get_english_iso, get_iso_from_alpha3, clean_columns

def to_iso(series):
    out = []
    for p in series:
        if pd.isna(p): out.append(None); continue
        s = str(p).strip().lower()
        code = custom_mappings.get(s)
        if not code:
            code = get_english_iso(p)
        out.append(code)
    return out

# ── 1. Charger v5 ──────────────────────────────────────────────────────────
print("[1] Chargement dataset_final_v5.csv…")
df = pd.read_csv(f"{D}/dataset_final_v5.csv")
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
print(f"   départ : {df.shape}")


# ── 2. Décomposition par culture spécifique (FAOSTAT) ─────────────────────
print("[2] Rendements par culture spécifique…")
df_prod = clean_columns(pd.read_csv(f"{D}/production_cultures.csv"))
df_prod["Pays_Clean"] = df_prod["Pays"].str.strip().str.lower()
df_prod["ISO"] = df_prod["Pays_Clean"].map(custom_mappings)
df_prod = df_prod.dropna(subset=["ISO"])
df_prod["Valeur"] = pd.to_numeric(df_prod["Valeur"], errors="coerce")

SPECIFIC_CROPS = {
    # Oléagineux
    "yield_soybeans":   "Fèves de soja",
    "yield_rapeseed":   "Graines de navette ou de colza",
    "yield_sunflower":  "Graines de tournesol",
    "yield_groundnut":  "Arachides non décortiquées",
    "yield_olives":     "Olives",
    "yield_sesame":     "Sésame",
    "yield_coconut":    "Noix de coco",
    "yield_cotton":     "Coton en graine, non égrené",
    # Fruits spécifiques
    "yield_apple":      "Pommes",
    "yield_banana":     "Bananes",
    "yield_orange":     "Oranges",
    "yield_mango":      "Mangues, mangoustans et goyaves",
    "yield_pineapple":  "Ananas",
    "yield_grape":      "Raisins",
    "yield_avocado":    "Avocats",
    "yield_lemon":      "Citrons et limes",
    "yield_peach":      "Pêches et nectarines",
    "yield_pear":       "Poires",
    "yield_strawberry": "Fraises",
    "yield_watermelon": "Pastèques",
    "yield_dates":      "Dattes",
    "yield_apricot":    "Abricots",
    "yield_cherry":     "Cerises",
    "yield_plum":       "Prunes et prunelles",
    # Légumes spécifiques
    "yield_tomato":     "Tomates, fraiches",
    "yield_potato":     "Pommes de terre",
    "yield_onion":      "Oignons, echalotes, frais",
    "yield_cabbage":    "Choux",
    "yield_carrot":     "Carottes et navets",
    "yield_cucumber":   "Concombres, cornichons",
    "yield_eggplant":   "Aubergines",
    "yield_cauliflower":"Choux-fleurs et brocolis",
    "yield_lettuce":    "Laitue et chicorée",
    # Légumineuses spécifiques
    "yield_chickpea":   "Pois chiches, secs",
    "yield_drybean":    "Haricots secs",
    "yield_drypea":     "Pois secs",
}

yield_rendement = df_prod[df_prod["Element"] == "Rendement"]
for new_col, prod_name in SPECIFIC_CROPS.items():
    sub = yield_rendement[yield_rendement["Produit"] == prod_name].copy()
    sub = sub[(sub["Valeur"] > 0) & (sub["Valeur"] <= 200000)]
    sub = sub.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(
        columns={"Valeur": new_col})
    df = df.merge(sub, on=["ISO", "Annee"], how="left")
print(f"   + {len(SPECIFIC_CROPS)} rendements spécifiques")

# Cibles ML : log1p sur chaque (mais on les définit après)


# ── 3. CRU TS / OWID temp historiques ─────────────────────────────────────
print("[3] CRU TS via OWID…")
for f, name in [
    ("owid_avg_monthly_temp.csv",           "owid_monthly_temp"),
    ("owid_annual_temp_anomaly.csv",        "owid_temp_anomaly"),
    ("owid_annual_precip_cru.csv",          "owid_precip_cru"),
    ("owid_co2_per_capita.csv",             "owid_co2_pc"),
    ("owid_methane.csv",                    "owid_methane"),
    ("owid_nitrous_oxide.csv",              "owid_n2o"),
    ("owid_water_stress_by_country.csv",    "owid_water_stress"),
    ("owid_ghg_by_sector.csv",              "owid_ghg_by_sector"),
]:
    p = f"{D}/{f}"
    if not os.path.exists(p): continue
    try:
        sub = pd.read_csv(p)
        if "Pays" not in sub.columns: continue
        sub["ISO"] = to_iso(sub["Pays"])
        sub = sub.dropna(subset=["ISO"])
        sub["Annee"] = pd.to_numeric(sub["Annee"], errors="coerce")
        sub = sub.dropna(subset=["Annee"])
        sub["Annee"] = sub["Annee"].astype(int)
        # Si plusieurs colonnes numériques (ex: ghg_by_sector), garder la première
        if "Valeur" in sub.columns:
            sub["Valeur"] = pd.to_numeric(sub["Valeur"], errors="coerce")
            sub = sub.dropna(subset=["Valeur"])
            sub = sub.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": name})
        else:
            num_cols = [c for c in sub.columns if pd.api.types.is_numeric_dtype(sub[c]) and c not in ("Annee",)]
            if not num_cols: continue
            sub["Valeur"] = sub[num_cols[0]]
            sub = sub.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": name})
        df = df.merge(sub, on=["ISO", "Annee"], how="left")
        print(f"   + {name}")
    except Exception as e:
        print(f"   ✗ {f}: {e}")


# ── 4. Catastrophes par type (équivalent EM-DAT ventilé) ──────────────────
print("[4] Catastrophes par type via OWID…")
import glob
disaster_files = glob.glob(f"{D}/disaster_deaths_*.csv") + glob.glob(f"{D}/disaster_affected_*.csv") + glob.glob(f"{D}/disaster_damage_*.csv")
for fp in disaster_files:
    f = os.path.basename(fp)
    name = f.replace(".csv", "")
    try:
        sub = pd.read_csv(fp)
        if "Pays" not in sub.columns: continue
        sub["ISO"] = to_iso(sub["Pays"])
        sub = sub.dropna(subset=["ISO"])
        sub["Annee"] = pd.to_numeric(sub["Annee"], errors="coerce")
        sub = sub.dropna(subset=["Annee"])
        sub["Annee"] = sub["Annee"].astype(int)
        sub["Valeur"] = pd.to_numeric(sub.get("Valeur", sub.iloc[:, -1]), errors="coerce")
        sub = sub.dropna(subset=["Valeur"])
        sub = sub.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(
            columns={"Valeur": name})
        df = df.merge(sub, on=["ISO", "Annee"], how="left")
        df[name] = df[name].fillna(0)
    except Exception as e:
        print(f"   ✗ {f}: {e}")
print(f"   + {len(disaster_files)} catastrophes ventilées")


# ── 5. OWID extras ─────────────────────────────────────────────────────────
print("[5] OWID extras (undernourishment, mental health, refugees…)…")
for f, name in [
    ("owid_undernourishment_share.csv",  "undernourishment_share"),
    ("owid_mental_health.csv",            "mental_health_share"),
    ("owid_rural_share.csv",              "rural_share"),
    ("owid_idps_conflict.csv",            "idps_conflict"),
    ("owid_refugees_origin.csv",          "refugees_origin_owid"),
    ("owid_elec_rural.csv",               "elec_rural_pct"),
    ("owid_oos_children.csv",             "out_of_school_pct"),
    ("owid_services_share.csv",           "services_share_owid"),
    ("owid_ghi.csv",                      "ghi_owid"),
    ("owid_agri_output_usd.csv",          "agri_output_usd"),
]:
    p = f"{D}/{f}"
    if not os.path.exists(p): continue
    try:
        sub = pd.read_csv(p)
        if "Pays" not in sub.columns: continue
        sub["ISO"] = to_iso(sub["Pays"])
        sub = sub.dropna(subset=["ISO"])
        sub["Annee"] = pd.to_numeric(sub["Annee"], errors="coerce")
        sub = sub.dropna(subset=["Annee"])
        sub["Annee"] = sub["Annee"].astype(int)
        sub["Valeur"] = pd.to_numeric(sub["Valeur"], errors="coerce")
        sub = sub.dropna(subset=["Valeur"])
        sub = sub.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": name})
        df = df.merge(sub, on=["ISO", "Annee"], how="left")
        print(f"   + {name}")
    except Exception as e:
        print(f"   ✗ {f}: {e}")


# ── 6. Cibles ML pour les rendements spécifiques ─────────────────────────
print("[6] Création des cibles ML…")
for col in SPECIFIC_CROPS.keys():
    if col in df.columns:
        df[f"target_{col}"] = np.log1p(df[col])

# ── 7. Lags sur principales nouvelles features ────────────────────────────
print("[7] Lags…")
df = df.sort_values(["ISO", "Annee"])
key_new = ["owid_temp_anomaly", "owid_precip_cru", "owid_water_stress",
           "owid_co2_pc", "owid_methane",
           "disaster_deaths_drought", "disaster_deaths_flood", "disaster_deaths_storm",
           "disaster_affected_drought", "disaster_affected_flood", "disaster_affected_storm",
           "undernourishment_share", "rural_share", "idps_conflict",
           "agri_output_usd", "services_share_owid"]
for v in key_new:
    if v in df.columns:
        for k in (1, 3, 5):
            df[f"{v}_lag{k}"] = df.groupby("ISO")[v].shift(k)
        df[f"{v}_roll5"] = df.groupby("ISO")[v].transform(lambda s: s.rolling(5, min_periods=2).mean())

# ── 8. Save v6 ────────────────────────────────────────────────────────────
out = f"{D}/dataset_final_v6.csv"
df.to_csv(out, index=False)
print(f"\n[OK] {out}")
print(f"     shape = {df.shape}  (+{df.shape[1] - 470} cols vs v5)")

num = df.select_dtypes(include="number").columns.tolist()
dyn = sum(1 for c in num if df.groupby("ISO")[c].std().mean() >= 0.01)
print(f"     DYNAMIC: {dyn}/{len(num)}")
