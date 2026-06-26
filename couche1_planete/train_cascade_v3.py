"""
train_cascade_v3.py — Cascade V3 BOOSTÉE avec dataset v13.

Améliorations vs v2 :
  - Dataset v13 (1070 colonnes, 81 cibles)
  - 8 sous-couches (atmosphère/hydrologie/sols/agri/élevage/pêche/énergie/géologie)
  - XGBoost 400 estim + features élargies + cascade OOF
  - Top-K = 150 features par cible
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
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from couche1_planete.cascade_config_v3 import (
    SUBLAYERS, SUBLAYER_ORDER, TARGET_SOURCE, EXTRA_LEAKS
)

LAYER_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(LAYER_DIR, "models_cascade_v3")
REPORTS_DIR = os.path.join(LAYER_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

TOP_K = 150
XGB_PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                   subsample=0.85, colsample_bytree=0.85,
                   random_state=42, n_jobs=-1, verbosity=0)

BASE_FEATURE_GROUPS = [
    "bio","temp_","precip_","nasa_t2m","nasa_prec","nasa_rh2m","nasa_ps",
    "nasa_ws10m","nasa_allsky","nasa_gwet","be_t_anom","be_t_baseline",
    "enso_","nao","amo","pdo","soi","ao_","co2_ppm_global",
    "latitude","longitude","elevation","slope","roughness",
    "dist_to_","tide_amplitude",
    "clay_pct","silt_pct","sand_pct","soil_pH","organic_carbon",
    "heat_stress","frost_risk","aridity","continentality",
    "growing_season","pet_annual",
    "cluster","area_km2","Population","Urban_pct",
    "feature_npp","feature_fauna","feature_photoperiod",
]


def make_xgb_pipe():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("model", XGBRegressor(**XGB_PARAMS))])

def make_rf_for_selection():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("rf", RandomForestRegressor(n_estimators=80, max_depth=12,
                                                    min_samples_leaf=4, n_jobs=-1, random_state=42))])

def select_features_for_sublayer(df, sublayer_name, cascade_oof_cols=None):
    cfg = SUBLAYERS[sublayer_name]
    all_groups = BASE_FEATURE_GROUPS + cfg["feature_groups"]
    cols = set()
    for prefix in all_groups:
        cols |= {c for c in df.columns
                  if (c == prefix or c.startswith(prefix))
                  and not c.startswith("target_")
                  and df[c].dtype != object}
    if cascade_oof_cols:
        cols |= set(cascade_oof_cols)
    return sorted(cols)


def get_blacklist(target, base_features):
    bl = {"ISO","Annee","T_ref","P_ref"}
    target_cols = [c for c in base_features if c.startswith("target_")]
    bl |= set(target_cols)
    src = TARGET_SOURCE.get(target)
    if src:
        bl |= {c for c in base_features if c == src or c.startswith(src + "_")}
    for extra in EXTRA_LEAKS.get(target, []):
        bl |= {c for c in base_features if c == extra or c.startswith(extra + "_")
                or c.startswith(extra)}
    return bl


def select_top_k(X, y, k=TOP_K):
    X = X.dropna(axis=1, how="all")
    if X.shape[1] <= k: return list(X.columns)
    pipe_rf = make_rf_for_selection()
    pipe_rf.fit(X, y)
    rf = pipe_rf.named_steps["rf"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(k).index.tolist()


def train_target(df, target, feats):
    d = df.dropna(subset=[target]).copy()
    if len(d) < 200 or d["ISO"].nunique() < 10:
        return None
    feats_use = [c for c in feats if c in d.columns and d[c].notna().sum() > 0]
    if not feats_use:
        return None

    X_full, y, groups = d[feats_use], d[target], d["ISO"]
    # FIX V3 : remplacer infinis par NaN (puis imputer)
    X_full = X_full.replace([np.inf, -np.inf], np.nan)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(splitter.split(d, groups=groups))

    sel = select_top_k(X_full.iloc[tr_idx], y.iloc[tr_idx], k=min(TOP_K, len(feats_use)))
    X = X_full[sel]

    pipe = make_xgb_pipe()
    pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred = pipe.predict(X.iloc[te_idx])
    r2 = r2_score(y.iloc[te_idx], pred)
    mae = mean_absolute_error(y.iloc[te_idx], pred)

    # OOF cascade
    oof = np.full(len(d), np.nan)
    gkf = GroupKFold(n_splits=3)
    for tr, te in gkf.split(X, y, groups=groups):
        p = make_xgb_pipe()
        p.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = p.predict(X.iloc[te])
    oof_series = pd.Series(np.nan, index=df.index)
    oof_series.loc[d.index] = oof

    return {
        "r2": r2, "mae": mae, "pipe": pipe, "features": sel,
        "n_obs": len(d), "n_test_countries": d.iloc[te_idx]["ISO"].nunique(),
        "oof": oof_series,
    }


# ── Chargement V13 ────────────────────────────────────────────────────────
print("══════════════════════════════════════════════════════════════")
print("🌍 COUCHE 1 — CASCADE V3 (V13 dataset, 8 sous-couches)")
print("══════════════════════════════════════════════════════════════\n")

path = "data/cleaned/shared/dataset_final_v13_couche1.csv"
print(f"[1] Chargement {path}…")
df = pd.read_csv(path, low_memory=False)
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
if "cluster" in df.columns:
    df["cluster"] = df["cluster"].astype(int)
print(f"   ✓ Shape: {df.shape}")


# ── CASCADE V3 ────────────────────────────────────────────────────────────
results = []
oof_cols_by_sublayer = {}

for sl_idx, sublayer_name in enumerate(SUBLAYER_ORDER):
    cfg = SUBLAYERS[sublayer_name]
    print(f"\n══════ {cfg['label']} ══════")

    cascade_from = cfg.get("use_cascade_from", [])
    cascade_oof_cols = []
    for prev_sl in cascade_from:
        cascade_oof_cols.extend(oof_cols_by_sublayer.get(prev_sl, []))
    if cascade_oof_cols:
        print(f"   → +{len(cascade_oof_cols)} OOF de SL précédentes")

    base_features = select_features_for_sublayer(df, sublayer_name, cascade_oof_cols)
    print(f"   → {len(base_features)} features candidates")

    oof_created = []
    for tgt, label in cfg["targets"].items():
        if tgt not in df.columns:
            print(f"   ⏭  {label} (cible absente)")
            continue
        bl = get_blacklist(tgt, base_features)
        feats_clean = [c for c in base_features if c not in bl]

        t0 = time.time()
        res = train_target(df, tgt, feats_clean)
        if res is None:
            print(f"   ⏭  {label} (données insuffisantes)")
            continue
        dt = time.time() - t0
        n_cascade = sum(1 for f in res["features"] if f.startswith("oof_"))
        cm = f" [+{n_cascade}cascade]" if n_cascade else ""
        print(f"   🎯 {label:35s} R²={res['r2']:+.3f} MAE={res['mae']:.2f}  "
              f"({len(res['features'])}f{cm}) {dt:.0f}s")

        joblib.dump({"pipe": res["pipe"], "features": res["features"],
                     "sublayer": sublayer_name},
                    os.path.join(MODELS_DIR, f"best_{tgt}.joblib"))
        oof_col = f"oof_{tgt}"
        df[oof_col] = res["oof"]
        oof_created.append(oof_col)

        results.append({
            "Sous-couche": cfg["label"], "Cible": label, "Technique": tgt,
            "R² cascade V3": round(res["r2"], 4),
            "MAE": round(res["mae"], 3),
            "N obs": res["n_obs"], "N features": len(res["features"]),
            "Cascade OOF": n_cascade,
        })

    oof_cols_by_sublayer[sublayer_name] = oof_created


# Sauvegarde
out = pd.DataFrame(results)
out_path = os.path.join(REPORTS_DIR, "cascade_v3_results.csv")
out.to_csv(out_path, index=False)

print("\n══════════════════════════════════════════════════════════════")
print("📊 BILAN CASCADE V3")
print("══════════════════════════════════════════════════════════════")
bilan = out.groupby("Sous-couche").agg(
    n_cibles=("Technique", "count"),
    R2_moyen=("R² cascade V3", "mean"),
    R2_max=("R² cascade V3", "max"),
).round(3)
print(bilan.to_string())
print(f"\n→ {out_path}  ({len(out)} cibles)")
print(f"→ {len(os.listdir(MODELS_DIR))} modèles dans {MODELS_DIR}/")

# Tableau global
print("\n📊 RÉSULTATS DÉTAILLÉS")
print(out.sort_values("R² cascade V3", ascending=False).to_string(index=False))
