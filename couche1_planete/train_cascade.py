"""
train_cascade.py — Entraînement en cascade Couche 1.

Pipeline :
  Pour chaque sous-couche dans l'ordre SL1.1 → SL1.6 :
    1. Sélectionner les features autorisées (groupes définis + cluster + cascade des SL précédentes)
    2. Pour chaque cible de la sous-couche :
       a. Anti-leak (drop sources brutes + leaks spécifiques)
       b. Train/test GroupShuffleSplit par pays
       c. Entraîner RandomForest léger (n_est=150, depth=12)
       d. R² test + sauvegarde modèle
       e. Générer OOF predictions GroupKFold(3) → cascade vers SL suivantes
    3. Inject OOF preds (preview SL_X__target_*) dans df pour SL suivantes

Outputs :
  - couche1_planete/models_cascade/best_{sublayer}_{target}.joblib
  - couche1_planete/reports/cascade_results.csv
"""
import os, sys, io, warnings, time
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

from couche1_planete.cascade_config import (
    SUBLAYERS, SUBLAYER_ORDER, TARGET_SOURCE, EXTRA_LEAKS
)

# ── Configuration ─────────────────────────────────────────────────────────
LAYER_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(LAYER_DIR, "models_cascade")
REPORTS_DIR = os.path.join(LAYER_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# RF léger comme demandé par l'utilisateur
RF_PARAMS = dict(n_estimators=150, max_depth=15, min_samples_leaf=4,
                  n_jobs=-1, random_state=42)

print("══════════════════════════════════════════════════════════════")
print("🌍 COUCHE 1 — CASCADE EN SOUS-COUCHES")
print("══════════════════════════════════════════════════════════════")
for sl in SUBLAYER_ORDER:
    cfg = SUBLAYERS[sl]
    n = len(cfg["targets"])
    cascade = cfg.get("use_cascade_from", [])
    arrow = " ← " + ", ".join(c.split("_")[1] for c in cascade) if cascade else ""
    print(f"  {cfg['label']:30s} {n} cibles{arrow}")
print()


# ── 1. Charger dataset V12 + fish (intégration directe) ───────────────────
print("[1] Chargement dataset…")
for path in [f"data/cleaned/dataset_final_v12_couche1.csv",
             f"data/cleaned/dataset_final_v11_couche1.csv",
             f"data/cleaned/dataset_final_v8_honest.csv"]:
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
        print(f"   ✓ {path}  ({df.shape})")
        break

df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
if "cluster" in df.columns:
    df["cluster"] = df["cluster"].astype(int)

# Intégrer fish (FAO 2024 cleaned)
fish_total = pd.read_csv("data/cleaned/fish_production_total.csv")
fish_src   = pd.read_csv("data/cleaned/fish_production_by_source.csv")
df = df.merge(fish_total, on=["ISO", "Annee"], how="left")
df = df.merge(fish_src,   on=["ISO", "Annee"], how="left")
# Combiner aquaculture
aqua_cols = [c for c in df.columns if "Aquaculture" in c and c.endswith("_t")]
if aqua_cols:
    df["fish_aquaculture_total_t"] = df[aqua_cols].sum(axis=1, min_count=1)

# Créer cibles fish
df["target_fish_total"]       = np.log1p(df["fish_total_t"].clip(lower=0))
df["target_fish_capture"]     = np.log1p(df.get("fish_Capture_t", pd.Series(np.nan)).clip(lower=0))
df["target_fish_aquaculture"] = np.log1p(df["fish_aquaculture_total_t"].clip(lower=0))

# Cibles dérivées si pas déjà créées
for c in ["disaster_deaths", "disaster_affected"]:
    if c in df.columns and f"target_{c}" not in df.columns:
        df[f"target_{c}"] = np.log1p(df[c].clip(lower=0))

print(f"   Dataset enrichi avec fish : {df.shape}")
print(f"   target_fish_total couverture : {df['target_fish_total'].notna().sum():,}")


# ── 2. Fonctions utilitaires ──────────────────────────────────────────────
def make_pipe():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("rf", RandomForestRegressor(**RF_PARAMS))])


def select_features_for_sublayer(df, sublayer_name, cascade_oof_cols=None):
    """Sélectionne les colonnes autorisées pour une sous-couche.
    Renvoie la liste des colonnes features (avant exclusion anti-leak)."""
    cfg = SUBLAYERS[sublayer_name]
    feat_groups = cfg["feature_groups"]
    cols = set()
    for prefix in feat_groups:
        cols |= {c for c in df.columns
                  if (c == prefix or c.startswith(prefix))
                  and not c.startswith("target_")
                  and df[c].dtype != object}
    # Ajouter les OOF prédictions de la cascade (déjà dans df si générées)
    if cascade_oof_cols:
        cols |= set(cascade_oof_cols)
    return sorted(cols)


def get_blacklist(target, base_features):
    """Anti-leak : drop source brute + leaks supplémentaires + tous les target_*."""
    bl = {"ISO", "Annee", "T_ref", "P_ref"}
    # Toutes les cibles
    target_cols = [c for c in base_features if c.startswith("target_")]
    bl |= set(target_cols)
    # Source brute
    src = TARGET_SOURCE.get(target)
    if src:
        bl |= {c for c in base_features
                if c == src or c.startswith(src + "_")}
    # Leaks supplémentaires
    for extra in EXTRA_LEAKS.get(target, []):
        bl |= {c for c in base_features
                if c == extra or c.startswith(extra + "_")}
    return bl


def train_target(df, target, feats):
    """Entraîne un RF sur une cible. Renvoie pipeline + R² + OOF preds."""
    d = df.dropna(subset=[target]).copy()
    if len(d) < 200 or d["ISO"].nunique() < 10:
        return None
    feats_use = [c for c in feats if c in d.columns and d[c].notna().sum() > 0]
    if not feats_use:
        return None
    X, y, groups = d[feats_use], d[target], d["ISO"]

    # Train/test split
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(splitter.split(d, groups=groups))
    pipe = make_pipe()
    pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred_te = pipe.predict(X.iloc[te_idx])
    r2 = r2_score(y.iloc[te_idx], pred_te)
    mae = mean_absolute_error(y.iloc[te_idx], pred_te)

    # OOF predictions pour cascade (GroupKFold 3)
    oof = np.full(len(d), np.nan)
    gkf = GroupKFold(n_splits=3)
    for tr, te in gkf.split(X, y, groups=groups):
        p = make_pipe()
        p.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = p.predict(X.iloc[te])
    # Aligner sur l'index global du df
    oof_series = pd.Series(np.nan, index=df.index)
    oof_series.loc[d.index] = oof

    return {
        "r2": r2, "mae": mae, "pipe": pipe, "features": feats_use,
        "n_obs": len(d), "n_test_countries": d.iloc[te_idx]["ISO"].nunique(),
        "oof": oof_series,
    }


# ── 3. CASCADE ────────────────────────────────────────────────────────────
results = []
oof_cols_by_sublayer = {}   # {sublayer: [list of OOF columns created]}

for sl_idx, sublayer_name in enumerate(SUBLAYER_ORDER):
    cfg = SUBLAYERS[sublayer_name]
    print(f"\n══════ {cfg['label']} — {cfg['description']} ══════")

    # Récupérer les OOF columns des sous-couches précédentes (cascade)
    cascade_from = cfg.get("use_cascade_from", [])
    cascade_oof_cols = []
    for prev_sl in cascade_from:
        cascade_oof_cols.extend(oof_cols_by_sublayer.get(prev_sl, []))
    if cascade_oof_cols:
        print(f"   → Utilise {len(cascade_oof_cols)} prédictions OOF des SL précédentes "
              f"({', '.join(c.split('_')[1] for c in cascade_from)})")

    # Features autorisées pour cette sous-couche
    base_features = select_features_for_sublayer(df, sublayer_name, cascade_oof_cols)
    print(f"   → {len(base_features)} features candidates")

    oof_created_this_sublayer = []

    for tgt, label in cfg["targets"].items():
        if tgt not in df.columns:
            print(f"   ⏭  {label} (cible absente)")
            continue

        # Blacklist anti-leak
        bl = get_blacklist(tgt, base_features)
        feats_clean = [c for c in base_features if c not in bl]

        t0 = time.time()
        res = train_target(df, tgt, feats_clean)
        if res is None:
            print(f"   ⏭  {label} (données insuffisantes)")
            continue
        dt = time.time() - t0
        n_cascade_used = sum(1 for f in res["features"] if f.startswith("oof_"))
        cascade_mark = f" [+{n_cascade_used} cascade]" if n_cascade_used else ""
        print(f"   🎯 {label:38s} R²={res['r2']:+.3f}  MAE={res['mae']:.2f}  "
              f"({len(res['features'])} feats{cascade_mark}) {dt:.0f}s")

        # Sauvegarder modèle
        joblib.dump({"pipe": res["pipe"], "features": res["features"],
                     "sublayer": sublayer_name},
                    os.path.join(MODELS_DIR, f"best_{tgt}.joblib"))

        # Injecter OOF dans df pour cascade
        oof_col = f"oof_{tgt}"
        df[oof_col] = res["oof"]
        oof_created_this_sublayer.append(oof_col)

        results.append({
            "Sous-couche": cfg["label"],
            "Cible": label,
            "Technique": tgt,
            "R² test": round(res["r2"], 4),
            "MAE": round(res["mae"], 3),
            "N obs": res["n_obs"],
            "N pays test": res["n_test_countries"],
            "N features": len(res["features"]),
            "Cascade OOF utilisées": n_cascade_used,
        })

    oof_cols_by_sublayer[sublayer_name] = oof_created_this_sublayer
    print(f"   ✓ {len(oof_created_this_sublayer)} OOF générées et propagées en cascade")


# ── 4. Sauvegarde résultats ───────────────────────────────────────────────
out = pd.DataFrame(results)
out_path = os.path.join(REPORTS_DIR, "cascade_results.csv")
out.to_csv(out_path, index=False)

print("\n══════════════════════════════════════════════════════════════")
print("📊 RÉSULTATS CASCADE COUCHE 1")
print("══════════════════════════════════════════════════════════════")
print(out.to_string(index=False))

# Bilan par sous-couche
print("\n\n  Bilan par sous-couche :")
bilan = out.groupby("Sous-couche").agg(
    n_cibles=("Technique", "count"),
    R2_moyen=("R² test", "mean"),
    R2_median=("R² test", "median"),
    R2_min=("R² test", "min"),
    R2_max=("R² test", "max"),
).round(3)
print(bilan.to_string())

print(f"\n→ {out_path}")
print(f"→ {len(os.listdir(MODELS_DIR))} modèles sauvegardés dans {MODELS_DIR}/")
