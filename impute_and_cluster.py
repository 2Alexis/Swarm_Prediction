"""
impute_and_cluster.py — Nettoie les NaN et clusterise les pays.

Étapes :
  1. Filtrer les pays trop incomplets (>50% NaN) — micro-territoires
  2. Imputer les NaN restants intelligemment :
     - Forward/backward fill par pays (NaN temporels)
     - Pour cultures non cultivées : 0 + flag binaire 'crop_grown'
     - Pour features socio/économiques : interpolation linéaire dans le temps par pays
     - Pour features statiques : valeur médiane du cluster climatique le plus proche
  3. Clusteriser les pays via KMeans sur features environnementales
  4. Sauvegarder dataset_final_v8_clean.csv + mapping ISO → cluster
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

# ── 2. Audit NaN par pays (sans exclure) ───────────────────────────────────
print("\n[2] Audit des NaN (on garde tous les pays)…")
nan_by_country = df.groupby("ISO").apply(
    lambda g: g.isna().sum().sum() / (len(g) * df.shape[1]) * 100,
    include_groups=False
)
n_pathological = (nan_by_country > 50).sum()
print(f"   {df['ISO'].nunique()} pays au total")
print(f"   {n_pathological} pays avec >50% NaN (micro-territoires)")
print(f"   → on garde tout, on impute mieux pour les remplir")

# ── 3. Identifier les colonnes de yield (cultures spécifiques) ─────────────
print("\n[3] Création de flags 'cultivé' pour cultures spécifiques…")
yield_cols = [c for c in df.columns
              if (c.startswith("yield_") and c.endswith("_kgha"))
              or (c.startswith("yield_") and not c.endswith("_kgha")
                  and c not in ("yield_cereals_kgha","yield_oilcrops_kgha",
                                "yield_pulses_kgha","yield_roots_kgha",
                                "yield_fruits_kgha","yield_vegetables_kgha"))]

# Pour chaque culture, créer un flag binaire "cultivé par ce pays/année"
for c in yield_cols:
    # Si le pays a JAMAIS cultivé cette plante → ne pas imputer, garder NaN
    # Si le pays cultive parfois → flag=1 si valeur, 0 sinon
    pays_cultiv = df.groupby("ISO")[c].apply(lambda s: s.notna().any())
    df[f"{c}_cultivated"] = df["ISO"].map(pays_cultiv).astype(int)

print(f"   {len(yield_cols)} flags 'cultivated' créés")


# ── 4. Imputation temporelle par pays (forward/backward fill court terme) ─
print("\n[4] Imputation temporelle par pays…")
df = df.sort_values(["ISO", "Annee"]).reset_index(drop=True)

# Colonnes dynamiques temporelles (varient avec l'année)
dynamic_cols = []
for c in df.columns:
    if df[c].dtype not in ("float64", "int64"): continue
    if c in ("ISO", "Annee", "T_ref", "P_ref"): continue
    if c.endswith("_cultivated"): continue
    # Détection: vraiment dynamique au sein d'un pays ?
    sample_std = df.groupby("ISO")[c].std().mean()
    if pd.notna(sample_std) and sample_std > 0:
        dynamic_cols.append(c)

print(f"   {len(dynamic_cols)} colonnes dynamiques à imputer temporellement")

# Interpolation linéaire dans le temps par pays
for c in dynamic_cols:
    df[c] = df.groupby("ISO")[c].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both", limit=5)
    )

# Forward fill (au-delà de la dernière valeur connue)
for c in dynamic_cols:
    df[c] = df.groupby("ISO")[c].transform(lambda s: s.ffill().bfill())


# ── 5. Imputation des cibles cultures : 0 si jamais cultivée ──────────────
print("\n[5] Imputation cibles cultures non cultivées → 0…")
for c in yield_cols:
    flag_col = f"{c}_cultivated"
    # Si le pays NE cultive PAS cette plante → mettre 0
    mask_not_cult = df[flag_col] == 0
    df.loc[mask_not_cult, c] = 0
    # Pour ceux qui cultivent mais NaN ponctuels → garder l'interpolation faite
    target_col = f"target_{c}" if not c.endswith("_kgha") else f"target_{c.replace('_kgha','')}"
    if target_col in df.columns:
        df.loc[mask_not_cult, target_col] = 0


# ── 6. Imputation des statiques avec médiane régionale (cluster) ──────────
print("\n[6] Clustering des pays sur features environnementales…")
# Features statiques par pays (climat, sol, géographie)
static_feats = ["latitude", "longitude", "elevation", "temp_mean", "precip_mean",
                "precip_seasonality", "solar_radiation_mean", "wind_speed_mean",
                "vapor_pressure_mean", "dist_to_coast_km", "dist_to_river_km",
                "clay_pct", "sand_pct", "soil_pH", "organic_carbon_pct"]
static_feats = [c for c in static_feats if c in df.columns]

# Un point par pays (moyenne sur années)
country_profile = df.groupby("ISO")[static_feats].mean().reset_index()
print(f"   {len(country_profile)} pays profilés sur {len(static_feats)} features")

# Imputer les NaN du profil (rare mais possible)
imp = SimpleImputer(strategy="median")
X_prof = pd.DataFrame(imp.fit_transform(country_profile[static_feats]),
                      index=country_profile["ISO"], columns=static_feats)

# Standardiser
sc = StandardScaler()
X_scaled = sc.fit_transform(X_prof)

# KMeans 8 clusters climatiques
N_CLUSTERS = 8
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = km.fit_predict(X_scaled)
country_profile["cluster"] = cluster_labels
iso_to_cluster = dict(zip(country_profile["ISO"], cluster_labels))
df["cluster"] = df["ISO"].map(iso_to_cluster)

# Description des clusters (climat moyen)
print("\n[7] Profil des clusters climatiques :")
for c in range(N_CLUSTERS):
    members = country_profile[country_profile["cluster"] == c]
    means = members[static_feats].mean()
    n = len(members)
    examples = members["ISO"].head(5).tolist()
    print(f"\n  Cluster {c} ({n} pays) — exemples : {examples}")
    print(f"    T mean = {means.get('temp_mean', float('nan')):.1f}°C  "
          f"P annual = {means.get('precip_mean', float('nan')):.0f}mm  "
          f"latitude = {means.get('latitude', float('nan')):.0f}°")


# ── 8. Imputation par cluster pour les features encore NaN ────────────────
print("\n[8] Imputation par cluster pour features encore NaN…")
for c in dynamic_cols:
    n_nan_before = df[c].isna().sum()
    if n_nan_before == 0: continue
    # Médiane par cluster × année
    df[c] = df.groupby(["cluster", "Annee"])[c].transform(
        lambda s: s.fillna(s.median())
    )
    # Si toujours NaN, médiane globale par année
    df[c] = df.groupby("Annee")[c].transform(lambda s: s.fillna(s.median()))


# ── 9. Recalculer les cibles dérivées qui dépendent des features imputées ──
print("\n[9] Recalcul des cibles log1p sur features imputées…")
target_log_map = {
    "target_yield_cereals": "yield_cereals_kgha", "target_yield_oilcrops": "yield_oilcrops_kgha",
    "target_yield_pulses": "yield_pulses_kgha", "target_yield_roots": "yield_roots_kgha",
    "target_yield_fruits": "yield_fruits_kgha", "target_yield_vegetables": "yield_vegetables_kgha",
}
for tgt, src in target_log_map.items():
    if src in df.columns and tgt in df.columns:
        # Garde NaN si la source est NaN dans le pays
        df[tgt] = np.log1p(df[src].clip(lower=0))

for c in yield_cols:
    if not c.endswith("_kgha") and not c.startswith("yield_cereals"):
        tgt = f"target_{c}"
        if tgt in df.columns and c in df.columns:
            df[tgt] = np.log1p(df[c].clip(lower=0))


# ── 10. Stats finales ─────────────────────────────────────────────────────
print("\n[10] Stats finales :")
nan_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
print(f"   shape : {df.shape}")
print(f"   pays : {df['ISO'].nunique()}")
print(f"   NaN total : {nan_pct:.1f}% (vs ~25% avant)")

# Top cibles - taux de NaN
print("\n   Cibles - taux non-null :")
targets = [c for c in df.columns if c.startswith("target_")]
for t in sorted(targets)[:25]:
    n = df[t].notna().sum()
    print(f"     {t:35s} {n:6,d} ({n/len(df)*100:.0f}%)")


# ── 11. Sauvegarde ─────────────────────────────────────────────────────────
out_csv = f"{D}/dataset_final_v8_clean.csv"
df.to_csv(out_csv, index=False)
print(f"\n[OK] {out_csv}")

# Mapping cluster
country_profile.to_csv(f"{D}/country_clusters.csv", index=False)
print(f"[OK] {D}/country_clusters.csv")
