"""Refit 4 carcass targets V14 — garde FAO yields (vraies stats indépendantes)."""
import os, sys, io, warnings
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

CLN = "data/cleaned"
df = pd.read_csv(f"{CLN}/shared/dataset_final_v14_couche1.csv", low_memory=False)

TARGETS = {
    "target_pig_carcass":       ["livestock_pig_slaughtered","meat_pig_kg_pc","livestock_pig_carcass_kg"],
    "target_sheepgoat_carcass": ["livestock_sheepgoat_slaughtered","meat_sheepgoat_kg_pc","livestock_sheepgoat_carcass_kg"],
    "target_cattle_carcass":    ["livestock_cattle_slaughtered","meat_beef_kg_pc","glw4_","livestock_cattle_carcass_kg"],
    "target_chicken_carcass":   ["livestock_chicken_slaughtered","meat_poultry_kg_pc","livestock_chicken_carcass_g"],
}

results = []
for tgt, leaks in TARGETS.items():
    print(f"\n━━ {tgt} ━━")
    drop_cols = {"ISO","Annee","T_ref","P_ref"}
    drop_cols |= {c for c in df.columns if c.startswith("target_")}
    for l in leaks:
        drop_cols |= {c for c in df.columns if c == l or c.startswith(l)}
    feats = [c for c in df.columns if c not in drop_cols and df[c].dtype != object]
    d = df.dropna(subset=[tgt]).copy()
    feats = [c for c in feats if d[c].notna().sum() > 0]

    X_full, y, groups = d[feats], d[tgt], d["ISO"]
    X_full = X_full.replace([np.inf, -np.inf], np.nan)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(splitter.split(d, groups=groups))

    pipe_rf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),("sc", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=4,
                                      n_jobs=-1, random_state=42))])
    pipe_rf.fit(X_full.iloc[tr_idx], y.iloc[tr_idx])
    imp = pd.Series(pipe_rf.named_steps["rf"].feature_importances_,
                    index=X_full.columns).sort_values(ascending=False)
    sel = imp.head(150).index.tolist()
    X = X_full[sel]

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),("sc", StandardScaler()),
        ("model", XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.04,
                                subsample=0.85, colsample_bytree=0.85,
                                random_state=42, n_jobs=-1, verbosity=0))])
    pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred = pipe.predict(X.iloc[te_idx])
    r2 = r2_score(y.iloc[te_idx], pred)
    mae = mean_absolute_error(y.iloc[te_idx], pred)
    print(f"   R² = {r2:+.4f}   MAE = {mae:.3f}   (n={len(d)})")

    joblib.dump({"pipe": pipe, "features": sel, "sublayer": "v14_keep_fao_yields"},
                f"couche1_planete/models_cascade_v3/best_{tgt}.joblib")
    results.append({"target": tgt, "R2": round(r2,4), "MAE": round(mae,3), "n": len(d)})

print("\n📊 BILAN V14 (FAO yields conservés)")
res = pd.DataFrame(results)
print(res.to_string(index=False))
res.to_csv("couche1_planete/reports/carcass_v14_final.csv", index=False)
