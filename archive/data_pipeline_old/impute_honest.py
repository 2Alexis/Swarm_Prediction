"""
impute_honest.py — Imputation HONNÊTE : on impute SEULEMENT les features explicatives,
JAMAIS les cibles ni leurs sources brutes.

Règles :
  - Garder les NaN sur target_* et leurs sources brutes (yield_cereals_kgha, Child_Mort, etc.)
  - Imputer les features explicatives :
    * Forward/backward fill par pays (NaN temporels)
    * Cluster-based pour features encore manquantes
  - Flags _cultivated : utiles pour cibles cultures mais ne pas utiliser comme feature dans modèles
  - Clusters basés sur features environnementales statiques

Sortie : data/cleaned/dataset_final_v8_honest.csv
"""
import os, sys, io
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = "data/cleaned"

print("[1] Chargement v7…")
df = pd.read_csv(f"{D}/dataset_final_v7.csv", low_memory=False)
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
print(f"   départ : {df.shape}")

# ── 2. Identifier les colonnes À NE PAS toucher (cibles + sources brutes) ─
print("\n[2] Identification des colonnes intouchables…")
target_cols = [c for c in df.columns if c.startswith("target_")]

# Sources brutes des cibles (ne pas imputer non plus)
TARGET_SOURCES = [
    "yield_cereals_kgha","yield_oilcrops_kgha","yield_pulses_kgha","yield_roots_kgha",
    "yield_fruits_kgha","yield_vegetables_kgha","yield_fibres_kgha","yield_citrus_kgha",
    "yield_treenuts_kgha",
    "Water_Withdrawal_pct","Bilan_sols_kgha","T_anomaly",
    "Child_Mort","Life_Exp","Pop_Growth","Birth_Rate","Death_Rate","Net_Migration",
    "disaster_deaths","disaster_affected","disaster_damages_usd","disaster_events",
    "stunting_pct","Fertility_Rate","nasa_gwetroot","forest_share_pct",
    "tree_cover_loss_ha","pm25_annual",
]
# Cultures spécifiques (sources)
SPECIFIC_CROP_SOURCES = [c for c in df.columns
                        if c.startswith("yield_") and not c.endswith("_kgha")
                        and not c.startswith("yield_cereals")
                        and c not in TARGET_SOURCES]

# Intouchables = targets + sources + lags/rolling des sources
INTOUCHABLE = set(target_cols) | set(TARGET_SOURCES) | set(SPECIFIC_CROP_SOURCES)
for src in list(TARGET_SOURCES) + list(SPECIFIC_CROP_SOURCES):
    INTOUCHABLE |= {c for c in df.columns if c.startswith(src + "_")}

# OOF predictions si elles existent
INTOUCHABLE |= {c for c in df.columns if c.startswith("oof_")}

print(f"   {len(INTOUCHABLE)} colonnes INTOUCHABLES (cibles + sources brutes + lags)")

# Colonnes à imputer = tout le reste (numérique)
imputable_cols = [c for c in df.columns
                  if c not in INTOUCHABLE
                  and c not in ("ISO","Annee")
                  and df[c].dtype in ("float64","int64")]
print(f"   {len(imputable_cols)} colonnes IMPUTABLES (features explicatives uniquement)")


# ── 3. Imputation temporelle (forward/backward fill par pays) ─────────────
print("\n[3] Imputation temporelle par pays (interpolation linéaire)…")
df = df.sort_values(["ISO", "Annee"]).reset_index(drop=True)
for c in imputable_cols:
    df[c] = df.groupby("ISO")[c].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both", limit=5)
    )
# Puis ffill/bfill pour ce qui reste (extrémités)
for c in imputable_cols:
    df[c] = df.groupby("ISO")[c].transform(lambda s: s.ffill().bfill())

n_nan_after_step3 = df[imputable_cols].isna().sum().sum()
print(f"   NaN restants dans imputables : {n_nan_after_step3:,}")


# ── 4. Clustering sur features environnementales (pour étape 5) ───────────
print("\n[4] Clustering des pays sur features environnementales statiques…")
static_feats = ["latitude", "longitude", "elevation", "temp_mean", "precip_mean",
                "precip_seasonality", "solar_radiation_mean", "wind_speed_mean",
                "vapor_pressure_mean", "dist_to_coast_km", "dist_to_river_km",
                "clay_pct", "sand_pct", "soil_pH", "organic_carbon_pct"]
static_feats = [c for c in static_feats if c in df.columns]
country_profile = df.groupby("ISO")[static_feats].mean().reset_index()

imp = SimpleImputer(strategy="median")
X_prof = pd.DataFrame(imp.fit_transform(country_profile[static_feats]),
                      index=country_profile["ISO"], columns=static_feats)
sc = StandardScaler()
X_scaled = sc.fit_transform(X_prof)

N_CLUSTERS = 8
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
country_profile["cluster"] = km.fit_predict(X_scaled)
iso_to_cluster = dict(zip(country_profile["ISO"], country_profile["cluster"]))
df["cluster"] = df["ISO"].map(iso_to_cluster).astype(int)
print(f"   {N_CLUSTERS} clusters créés")

# Description rapide
for c in range(N_CLUSTERS):
    members = country_profile[country_profile["cluster"] == c]
    means = members[static_feats].mean()
    n = len(members)
    print(f"   Cluster {c}: {n:3d} pays | T={means.get('temp_mean',np.nan):4.1f}°C | "
          f"P={means.get('precip_mean',np.nan):4.0f}mm | lat={means.get('latitude',np.nan):4.0f}°")


# ── 5. Imputation par cluster pour ce qui reste ────────────────────────────
print("\n[5] Imputation par cluster pour features encore NaN…")
for c in imputable_cols:
    if df[c].isna().sum() == 0: continue
    df[c] = df.groupby(["cluster", "Annee"])[c].transform(lambda s: s.fillna(s.median()))
    df[c] = df.groupby("Annee")[c].transform(lambda s: s.fillna(s.median()))
    df[c] = df[c].fillna(df[c].median())

n_nan_features = df[imputable_cols].isna().sum().sum()
print(f"   NaN restants dans features : {n_nan_features:,}")


# ── 6. Stats finales ──────────────────────────────────────────────────────
print("\n[6] Stats finales :")
n_nan_total = df.isna().sum().sum()
n_nan_targets = df[target_cols].isna().sum().sum()
n_nan_sources = df[[c for c in INTOUCHABLE if c in df.columns]].isna().sum().sum()
print(f"   NaN total       : {n_nan_total:,}  ({n_nan_total/(df.shape[0]*df.shape[1])*100:.1f}%)")
print(f"   NaN dans cibles : {n_nan_targets:,}  ← OK c'est voulu (anti-leak)")
print(f"   NaN dans sources/lags : {n_nan_sources:,}  ← OK c'est voulu")
print(f"   NaN dans features imputables : {n_nan_features:,}  ← devrait être 0")

print("\n   Couverture cibles principales :")
for t in sorted(target_cols)[:25]:
    n = df[t].notna().sum()
    print(f"     {t:35s} {n:6,d} ({n/len(df)*100:.0f}%)")


# ── 7. Sauvegarde ─────────────────────────────────────────────────────────
out_csv = f"{D}/dataset_final_v8_honest.csv"
df.to_csv(out_csv, index=False)
print(f"\n[OK] {out_csv}")

country_profile.to_csv(f"{D}/country_clusters.csv", index=False)
print(f"[OK] {D}/country_clusters.csv ({N_CLUSTERS} clusters)")
