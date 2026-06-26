"""
enrich_v4.py — Enrichit dataset_final_v4.csv pour produire dataset_final_v5.csv

  + NASA POWER 12 variables annuelles (T réelles, soil moisture, humidité…)
  + WHO PM2.5 & ambient deaths
  + OWID forêt (forest_share, change, tree_cover_loss, deforestation, forest_area_km)
  + OWID démographie détaillée (births, deaths, crude_birth, crude_death, pop_density, pop_by_age)
  + Anomalies NASA (vs référence pays 1990-2000)
  + Lags sur NASA + forêt + démo détaillée
  + Décomposition pop_growth → 3 cibles (birth_rate, death_rate, net_migration)
"""
import os, sys, io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = "data/cleaned"

from build_dataset import custom_mappings, get_english_iso, get_iso_from_alpha3

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

# ── 1. Charger v4 ──────────────────────────────────────────────────────────
print("[1] Chargement dataset_final_v4.csv…")
df = pd.read_csv(f"{D}/dataset_final_v4.csv")
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
print(f"   départ : {df.shape}")

# ── 2. NASA POWER ──────────────────────────────────────────────────────────
print("[2] NASA POWER (climat réel + soil moisture)…")
nasa = pd.read_csv(f"{D}/nasa_power_yearly.csv")
nasa["ISO"] = nasa["ISO"].astype(str)
df = df.merge(nasa, on=["ISO", "Annee"], how="left")
print(f"   + {nasa.shape[1]-2} colonnes NASA")

# Anomalies NASA (vs référence pays 1990-2000)
nasa_cols = [c for c in nasa.columns if c.startswith("nasa_")]
ref = df[(df["Annee"] >= 1990) & (df["Annee"] <= 2000)].groupby("ISO")[nasa_cols].mean().reset_index()
ref.columns = ["ISO"] + [f"{c}_ref" for c in nasa_cols]
df = df.merge(ref, on="ISO", how="left")
for c in nasa_cols:
    df[f"{c}_anomaly"] = df[c] - df[f"{c}_ref"]

# Lags sur NASA-clés (T, P, soil moisture, solar)
df = df.sort_values(["ISO", "Annee"])
key_nasa = ["nasa_t2m", "nasa_prectotcorr", "nasa_gwetroot", "nasa_gwettop",
            "nasa_gwetprof", "nasa_allsky_sfc_sw_dwn", "nasa_rh2m",
            "nasa_t2m_anomaly", "nasa_prectotcorr_anomaly", "nasa_gwetroot_anomaly"]
for v in key_nasa:
    if v in df.columns:
        for k in (1, 3, 5):
            df[f"{v}_lag{k}"] = df.groupby("ISO")[v].shift(k)
        df[f"{v}_roll5"] = df.groupby("ISO")[v].transform(lambda s: s.rolling(5, min_periods=2).mean())
print(f"   + anomalies + lags NASA")

# ── 3. WHO Air Quality ─────────────────────────────────────────────────────
print("[3] WHO Air Quality…")
for f, name in [("who_pm25_mean.csv", "who_pm25_mean"),
                ("who_ambient_air_deaths_rate.csv", "who_ambient_air_deaths_rate")]:
    p = f"{D}/{f}"
    if not os.path.exists(p): continue
    sub = pd.read_csv(p)
    if "ISO" not in sub.columns: continue
    sub["Annee"] = sub["Annee"].astype(int)
    df = df.merge(sub, on=["ISO", "Annee"], how="left")
    print(f"   + {name}")

# ── 4. OWID forêt ──────────────────────────────────────────────────────────
print("[4] OWID forêt…")
for f, name in [
    ("owid_forest_share.csv",            "forest_share_pct"),
    ("owid_forest_change.csv",           "forest_change"),
    ("owid_tree_cover_loss.csv",         "tree_cover_loss_ha"),
    ("owid_annual_deforestation.csv",    "deforestation_annual"),
    ("owid_forest_area_km.csv",          "forest_area_km2"),
]:
    p = f"{D}/{f}"
    if not os.path.exists(p): continue
    sub = pd.read_csv(p)
    sub["ISO"] = to_iso(sub["Pays"])
    sub = sub.dropna(subset=["ISO"])
    sub["Annee"] = pd.to_numeric(sub["Annee"], errors="coerce")
    sub = sub.dropna(subset=["Annee"])
    sub["Annee"] = sub["Annee"].astype(int)
    sub["Valeur"] = pd.to_numeric(sub["Valeur"], errors="coerce")
    sub = sub.dropna(subset=["Valeur"])
    sub = sub.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": name})
    df = df.merge(sub, on=["ISO", "Annee"], how="left")
    print(f"   + {name}: {sub[name].notna().sum()}")

# Cumul déforestation 5y
if "tree_cover_loss_ha" in df.columns:
    df = df.sort_values(["ISO", "Annee"])
    df["tree_cover_loss_cumul5y"] = df.groupby("ISO")["tree_cover_loss_ha"].transform(
        lambda s: s.rolling(5, min_periods=1).sum())

# ── 5. OWID démographie détaillée ──────────────────────────────────────────
print("[5] OWID démographie détaillée (pour décomposition pop_growth)…")
for f, name in [
    ("owid_births.csv",            "owid_births_total"),
    ("owid_deaths.csv",            "owid_deaths_total"),
    ("owid_crude_birth_rate.csv",  "owid_crude_birth_rate"),
    ("owid_crude_death_rate.csv",  "owid_crude_death_rate"),
    ("owid_pop_density.csv",       "owid_pop_density"),
    ("owid_agri_land_share.csv",   "owid_agri_land_share"),
]:
    p = f"{D}/{f}"
    if not os.path.exists(p): continue
    sub = pd.read_csv(p)
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

# ── 6. Features dérivées V5 ────────────────────────────────────────────────
print("[6] Features dérivées V5…")
def safe_div(a, b, default=np.nan):
    return np.where((b > 0) & b.notna() & a.notna(), a / b, default)

# Déficit d'humidité du sol par rapport à référence
if "nasa_gwetroot" in df.columns and "nasa_gwetroot_ref" in df.columns:
    df["soil_moisture_deficit"] = df["nasa_gwetroot_ref"] - df["nasa_gwetroot"]

# Stress hydrique combiné : faible humidité sol × faible précip
if "nasa_gwetroot" in df.columns and "nasa_prectotcorr" in df.columns:
    df["combined_drought_index"] = (1 - df["nasa_gwetroot"]) * np.exp(-df["nasa_prectotcorr"] / 1000.0)

# Aridité (P / ET potentielle, approx via T)
if "nasa_t2m" in df.columns and "nasa_prectotcorr" in df.columns:
    pet_approx = 16 * (10 * df["nasa_t2m"] / 12).clip(lower=0) ** 0.5  # Thornthwaite simplifié
    df["aridity_index_real"] = safe_div(df["nasa_prectotcorr"], pet_approx * 12)

# Forêt par habitant
if "forest_area_km2" in df.columns and "Population" in df.columns:
    df["forest_per_capita_km2"] = safe_div(df["forest_area_km2"] * 1e6, df["Population"])

# Pourcentage déforestation annuel
if "tree_cover_loss_ha" in df.columns and "forest_area_km2" in df.columns:
    df["deforestation_pct_annual"] = safe_div(df["tree_cover_loss_ha"] / 100,  # ha->km2
                                              df["forest_area_km2"]) * 100

# Naissances/décès par jour (intensité réelle)
if "owid_crude_birth_rate" in df.columns:
    df["births_per_1000_minus_deaths"] = df["owid_crude_birth_rate"] - df.get("owid_crude_death_rate", 0)

# Saisonnalité hygrique NASA
if "nasa_gwetroot" in df.columns and "nasa_gwettop" in df.columns:
    df["soil_moisture_top_root_ratio"] = safe_div(df["nasa_gwettop"], df["nasa_gwetroot"])

# Aridité-T interaction (chaud + sec = stress)
if "nasa_t2m" in df.columns and "nasa_gwetroot" in df.columns:
    df["heat_drought_stress"] = (df["nasa_t2m"] - 15) * (1 - df["nasa_gwetroot"])

print("   ok features dérivées")

# ── 7. Décomposition de pop_growth en 3 cibles ──────────────────────────────
print("[7] Décomposition pop_growth → 3 cibles…")
# Birth_Rate, Death_Rate, Net_Migration sont déjà dans v4 (WB)
df["target_birth_rate"] = df.get("Birth_Rate")
df["target_death_rate"] = df.get("Death_Rate")
df["target_net_migration"] = df.get("Net_Migration")
# Cibles alternatives via OWID (cross-check)
if "owid_crude_birth_rate" in df.columns:
    df["target_birth_rate_owid"] = df["owid_crude_birth_rate"]
if "owid_crude_death_rate" in df.columns:
    df["target_death_rate_owid"] = df["owid_crude_death_rate"]

# Cibles NASA-spécifiques utiles
if "nasa_gwetroot" in df.columns:
    df["target_soil_moisture_root"] = df["nasa_gwetroot"]

# Cible forêt
if "forest_share_pct" in df.columns:
    df["target_forest_share"] = df["forest_share_pct"]
if "tree_cover_loss_ha" in df.columns:
    df["target_tree_cover_loss"] = np.log1p(df["tree_cover_loss_ha"].clip(lower=0))

# ── 8. Sauvegarde ─────────────────────────────────────────────────────────
out = f"{D}/dataset_final_v5.csv"
df.to_csv(out, index=False)
print(f"\n[OK] {out}")
print(f"     shape = {df.shape}")
print(f"     +{df.shape[1] - 364} colonnes vs v4")

num = df.select_dtypes(include="number").columns.tolist()
dyn = sum(1 for c in num if df.groupby("ISO")[c].std().mean() >= 0.01)
print(f"     DYNAMIC: {dyn}/{len(num)}")
