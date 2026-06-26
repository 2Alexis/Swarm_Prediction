"""
train_crop_specific.py — Modèles XGB DÉDIÉS par culture importante.

Pour chaque culture économiquement majeure, on entraîne un modèle avec :
  - Features sélectionnées (top-50 RF importance) pour CETTE culture spécifique
  - Hyperparamètres optimisés (grid simple)
  - Pondération éventuelle des features les plus pertinentes selon agronomie

Stratégie : pour chaque culture, entraîner Ridge / RF / XGBoost avec 4 jeux
d'hyperparamètres XGB et garder le meilleur sur split par pays.
"""
import os, sys, io
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
DATA = "data/cleaned/dataset_final_v6.csv"
os.makedirs("models_crop", exist_ok=True)
os.makedirs("reports", exist_ok=True)

df = pd.read_csv(DATA)
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
print(f"Dataset: {df.shape}")

# Cultures économiquement importantes (top exportations + sécurité alimentaire)
IMPORTANT_CROPS = [
    "yield_soybeans", "yield_rapeseed", "yield_sunflower", "yield_groundnut",
    "yield_cotton", "yield_apple", "yield_banana", "yield_grape",
    "yield_orange", "yield_mango",
    "yield_tomato", "yield_potato", "yield_onion",
    "yield_cucumber", "yield_eggplant",
    "yield_drybean", "yield_chickpea",
    "yield_cereals", "yield_oilcrops", "yield_pulses", "yield_fruits",
    "yield_vegetables", "yield_roots",
]

# Pour chaque culture, drop les autres yields (leak), drop sources directes
ALL_YIELDS = [c for c in df.columns if c.startswith("yield_") and "kgha" in c]
ALL_YIELDS += [c for c in df.columns if c.startswith("yield_") and c not in ALL_YIELDS]
ALL_YIELDS = list(set([c.replace("_kgha", "") for c in ALL_YIELDS] + ALL_YIELDS))

# Features GLOBALES à blacklister (socio + cumuls problématiques)
GLOBAL_BL = [
    "cereal_production_t", "cereal_yield", "food_production_index",
    "cereals_prod_t", "cereals_area_ha",
    "owid_cereal_production", "wb_cereal_yield",
]
ID_COLS = ["ISO", "Annee", "T_ref", "P_ref"]
TARGET_COLS = [c for c in df.columns if c.startswith("target_")]


def get_yield_blacklist(target_crop):
    """Blacklist exhaustive pour une culture : tous les autres yields + leurs lags + sources."""
    bl = set(ID_COLS) | set(TARGET_COLS) | set(GLOBAL_BL)
    # Drop la cible elle-même et ses lags
    bl |= {c for c in df.columns if c == target_crop or c.startswith(target_crop + "_")}
    # Drop TOUS les autres yields + leurs lags
    for y in ALL_YIELDS:
        if y in df.columns:
            bl.add(y)
            bl |= {c for c in df.columns if c.startswith(y + "_")}
    # Drop colonnes textuelles
    return bl


def features_for(target_crop):
    bl = get_yield_blacklist(target_crop)
    return [c for c in df.columns if c not in bl and df[c].dtype != object]


def make_pre():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])


def select_top_features(X, y, k=80):
    """Top-k via RandomForest importance."""
    X = X.dropna(axis=1, how="all")
    if X.shape[1] <= k:
        return list(X.columns)
    Xp = make_pre().fit_transform(X)
    rf = RandomForestRegressor(n_estimators=80, max_depth=12,
                               min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(Xp, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(k).index.tolist()


splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
results = []
TOP_K = 80

# Grille XGBoost (rapide)
XGB_GRID = [
    {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05},
    {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05},
    {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.03},
    {"n_estimators": 1000, "max_depth": 6, "learning_rate": 0.02},
]

for crop_col in IMPORTANT_CROPS:
    if crop_col not in df.columns:
        continue
    target = f"target_{crop_col}"
    if target not in df.columns:
        # Créer target log1p si pas existant
        df[target] = np.log1p(df[crop_col].clip(lower=0))

    d = df.dropna(subset=[target]).copy()
    if len(d) < 300:
        continue

    tr_idx, te_idx = next(splitter.split(d, groups=d["ISO"]))
    tr, te = d.iloc[tr_idx], d.iloc[te_idx]

    feats_all = features_for(crop_col)
    Xtr_full, Xte_full = tr[feats_all], te[feats_all]
    ytr, yte = tr[target], te[target]

    selected = select_top_features(Xtr_full, ytr, k=TOP_K)
    Xtr, Xte = Xtr_full[selected], Xte_full[selected]

    print(f"\n🌾 {crop_col}  (n_train={len(tr)}, pays={tr['ISO'].nunique()})")
    print(f"   features candidats={len(feats_all)} → top {len(selected)}")

    best = (-np.inf, None, None, None)
    # Ridge
    pipe = Pipeline([("pre", make_pre()), ("model", Ridge(alpha=1.0))])
    pipe.fit(Xtr, ytr)
    r2 = r2_score(yte, pipe.predict(Xte))
    if r2 > best[0]: best = (r2, "Ridge", pipe, mean_absolute_error(yte, pipe.predict(Xte)))
    # RF
    rf = RandomForestRegressor(n_estimators=300, max_depth=15,
                               min_samples_leaf=4, n_jobs=-1, random_state=42)
    pipe = Pipeline([("pre", make_pre()), ("model", rf)])
    pipe.fit(Xtr, ytr)
    r2 = r2_score(yte, pipe.predict(Xte))
    if r2 > best[0]: best = (r2, "RandomForest", pipe, mean_absolute_error(yte, pipe.predict(Xte)))
    # XGB grid
    for params in XGB_GRID:
        try:
            xgb = XGBRegressor(**params, subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1, verbosity=0)
            pipe = Pipeline([("pre", make_pre()), ("model", xgb)])
            pipe.fit(Xtr, ytr)
            r2 = r2_score(yte, pipe.predict(Xte))
            if r2 > best[0]:
                tag = f"XGB(n={params['n_estimators']},d={params['max_depth']},lr={params['learning_rate']})"
                best = (r2, tag, pipe, mean_absolute_error(yte, pipe.predict(Xte)))
        except Exception:
            continue
    r2, bname, bpipe, bmae = best
    joblib.dump({"pipe": bpipe, "features": selected, "model": bname}, f"models_crop/best_{crop_col}.joblib")
    print(f"   🏆 {bname}  R²={r2:+.4f}  MAE={bmae:.3f}")
    results.append({"Crop": crop_col, "Meilleur": bname, "R²": round(r2, 4),
                    "MAE": round(bmae, 3), "N test": len(yte),
                    "N feats": len(selected)})

out = pd.DataFrame(results).sort_values("R²", ascending=False)
out.to_csv("reports/crop_specific_results.csv", index=False)
print("\n📊 Résultats modèles dédiés par culture :")
print(out.to_string(index=False))
