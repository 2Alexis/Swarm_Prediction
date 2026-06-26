"""
analyze_features.py — Diagnostic complet du dataset v3 :
  1. Quelles features corrèlent le plus avec chaque cible ?
  2. Quelles features sont inutiles (très peu de variance, trop de NaN) ?
  3. Quelles cibles manquent de features pertinentes ?
  4. Feature importance des modèles XGBoost
"""
import os, sys, io
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sns.set_theme(style="whitegrid")

D = "data/cleaned/dataset_final_v3.csv"
df = pd.read_csv(D)
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)

os.makedirs("reports/analysis", exist_ok=True)

TARGETS = [c for c in df.columns if c.startswith("target_")]
NUM = df.select_dtypes(include="number").columns.tolist()
FEATS = [c for c in NUM if not c.startswith("target_") and c not in ("Annee", "T_ref", "P_ref")]

print(f"Dataset: {df.shape} | {len(TARGETS)} cibles, {len(FEATS)} features numériques\n")

# ── 1. AUDIT QUALITÉ DES FEATURES ─────────────────────────────────────────
print("=" * 70)
print("1. AUDIT QUALITÉ — features problématiques")
print("=" * 70)

issues = []
for c in FEATS:
    s = df[c]
    n = len(s)
    nan_pct = s.isna().sum() / n * 100
    nunique = s.nunique()
    intra_std = df.groupby("ISO")[c].std().mean()
    issues.append({
        "feature": c,
        "nan_pct": nan_pct,
        "nunique": nunique,
        "intra_country_std": intra_std,
        "std": s.std() if pd.api.types.is_numeric_dtype(s) else np.nan,
    })
audit = pd.DataFrame(issues).sort_values("nan_pct", ascending=False)
print(f"\nFeatures avec >50% NaN ({(audit['nan_pct']>50).sum()}):")
print(audit[audit["nan_pct"] > 50][["feature", "nan_pct", "nunique"]].head(20).to_string(index=False))
print(f"\nFeatures statiques par pays (intra_std<0.01) : {(audit['intra_country_std']<0.01).sum()}")
print(audit[audit["intra_country_std"] < 0.01]["feature"].tolist())
audit.to_csv("reports/analysis/audit_features.csv", index=False)


# ── 2. CORRÉLATIONS FEATURE → TARGET ───────────────────────────────────────
print("\n" + "=" * 70)
print("2. CORRÉLATIONS (Pearson) avec chaque cible")
print("=" * 70)

corr_records = []
for tgt in TARGETS:
    sub = df.dropna(subset=[tgt])
    if len(sub) < 200:
        continue
    cors = sub[FEATS].corrwith(sub[tgt], method="pearson").dropna()
    cors_abs = cors.abs().sort_values(ascending=False)
    print(f"\n🎯 {tgt}  (n={len(sub)})")
    print("   Top 10 corrélations |r| :")
    for f, r in cors_abs.head(10).items():
        print(f"     {f:45s} r={cors[f]:+.3f}")
    # Top 20 dans le fichier
    for f, r in cors_abs.head(20).items():
        corr_records.append({"target": tgt, "feature": f, "r": cors[f], "abs_r": r})
pd.DataFrame(corr_records).to_csv("reports/analysis/top_correlations.csv", index=False)


# ── 3. FEATURE IMPORTANCE XGBoost ──────────────────────────────────────────
print("\n" + "=" * 70)
print("3. FEATURE IMPORTANCE des modèles XGBoost entraînés (v3)")
print("=" * 70)

models_dir = "models_v3"
fi_records = []
for tgt in TARGETS:
    p = f"{models_dir}/best_{tgt}.joblib"
    if not os.path.exists(p):
        continue
    pipe = joblib.load(p)
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        continue
    # Récupérer les noms de features depuis le pipeline
    # Le pipeline a déjà été fitté sur des colonnes spécifiques
    try:
        # On retrouve les features depuis la cible (logique cohérente avec train_v3)
        from train_v3 import build_X, get_blacklist  # même module
        d = df.dropna(subset=[tgt])
        _, feats = build_X(d, tgt)
        imp = model.feature_importances_
        if len(imp) != len(feats):
            continue
        ser = pd.Series(imp, index=feats).sort_values(ascending=False)
        print(f"\n🎯 {tgt}")
        print("   Top 10 importance :")
        for f, v in ser.head(10).items():
            print(f"     {f:45s} {v:.4f}")
        for f, v in ser.head(30).items():
            fi_records.append({"target": tgt, "feature": f, "importance": v})
    except Exception as e:
        print(f"   ✗ {tgt}: {e}")

if fi_records:
    fi_df = pd.DataFrame(fi_records)
    fi_df.to_csv("reports/analysis/feature_importance.csv", index=False)


# ── 4. CARTE DES "TROUS" — features qui pourraient aider mais absentes ────
print("\n" + "=" * 70)
print("4. ANALYSE DES CIBLES FAIBLES (R²<0.4)")
print("=" * 70)

WEAK = {
    "target_yield_fruits": "Fruits",
    "target_yield_pulses": "Légumineuses",
    "target_yield_vegetables": "Légumes",
    "target_yield_oilcrops": "Oléagineux",
    "target_pop_growth": "Croissance démographique",
    "target_disaster_damages_usd": "Dégâts économiques catastrophes",
}

for tgt, name in WEAK.items():
    if tgt not in df.columns:
        continue
    sub = df.dropna(subset=[tgt])
    print(f"\n🔻 {name} ({tgt})")
    print(f"   Données : {len(sub)} lignes, {sub['ISO'].nunique()} pays")

    # Corrélations existantes
    cors = sub[FEATS].corrwith(sub[tgt]).dropna().abs().sort_values(ascending=False)
    print(f"   Corrélation max actuelle : {cors.iloc[0]:.3f} ({cors.index[0]})")
    print(f"   Médiane des top 10 : {cors.head(10).median():.3f}")
    # Si la corrélation max est faible → on manque vraiment de features pertinentes
    if cors.iloc[0] < 0.5:
        print(f"   ⚠️  Aucune feature actuelle ne corrèle fortement ({cors.iloc[0]:.2f}<0.5)")
        print(f"      → Il faut chercher de nouvelles données spécifiques.")


# ── 5. HEATMAP CORRÉLATIONS — top 30 features × top targets ───────────────
print("\n" + "=" * 70)
print("5. Génération heatmaps")
print("=" * 70)

# Heatmap par groupe de features
GROUPS = {
    "Climat": [c for c in FEATS if any(k in c for k in ["temp_", "precip_", "T_anomaly", "P_anomaly", "T_annual", "P_annual", "enso", "soi", "nao", "amo", "pdo", "ao", "co2"])],
    "Sol": [c for c in FEATS if any(k in c for k in ["clay", "silt", "sand", "soil_pH", "organic_carbon", "Bilan_sols"])],
    "Agriculture": [c for c in FEATS if any(k in c for k in ["Engrais", "Pesticides", "Terres", "Bio_", "Irrigation", "Part_"])],
    "Géographie": [c for c in FEATS if any(k in c for k in ["latitude", "longitude", "elevation", "slope", "roughness", "dist_to_"])],
    "Catastrophes": [c for c in FEATS if "disaster" in c or "earthquake" in c or "volcan" in c],
    "Bio/Eco": [c for c in FEATS if any(k in c for k in ["feature_npp", "feature_fauna", "feature_vsi", "feature_photo", "wood_density"])],
    "Énergie": [c for c in FEATS if any(k in c for k in ["elec_", "Energy", "Renew", "Electricity"])],
}

key_targets = ["target_yield_cereals", "target_yield_fruits", "target_yield_pulses",
               "target_water_stress", "target_thermal_anomaly", "target_child_mortality",
               "target_pop_growth", "target_disaster_deaths"]

fig, axes = plt.subplots(len(GROUPS), 1, figsize=(14, 4*len(GROUPS)))
for ax, (gname, gfeats) in zip(axes, GROUPS.items()):
    if not gfeats:
        continue
    cmat = df[gfeats + key_targets].corr().loc[gfeats, key_targets]
    sns.heatmap(cmat, ax=ax, cmap="RdBu_r", center=0, vmin=-0.8, vmax=0.8,
                annot=True, fmt=".2f", cbar=True, annot_kws={"size": 7})
    ax.set_title(f"Corrélations — {gname}", weight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("reports/analysis/heatmap_by_group.png", dpi=120, bbox_inches="tight")
plt.close()
print("→ reports/analysis/heatmap_by_group.png")

print("\n[OK] Analyses dans reports/analysis/")
