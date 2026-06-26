"""3 chantiers d'amélioration :
(a) FAO Live Animals (stocks/heads) → nouvelles features pour carcasses
(b) Target encoding OOF par pays (cibles faibles)
(c) Poisson regressor pour target_seismic_activity
"""
import os, sys, io, zipfile, warnings, json
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
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import PoissonRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from build_dataset import custom_mappings, get_english_iso

CLN = "data/cleaned"
RAW = "data/raw"

def name_to_iso2(name):
    if pd.isna(name): return None
    s = str(name).strip().lower()
    if s in custom_mappings: return custom_mappings[s]
    return get_english_iso(name)


# ════════════════════════════════════════════════════════════════════
# (a) FAO Live Animals — extract Stocks + Slaughtered
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("(a) FAO LIVE ANIMALS — stocks & slaughtered")
print("="*70)

ZIP = f"{RAW}/elevage/fao_crops_livestock.zip"
TARGET_ITEMS = {
    "Cattle": "cattle", "Buffalo": "buffalo",
    "Chickens": "chicken", "Pigs": "pig",
    "Sheep": "sheep", "Goats": "goat",
    "Ducks": "duck", "Horses": "horse",
}
TARGET_ELEMENTS = {"Stocks": "stocks", "Producing Animals/Slaughtered": "slaughtered"}

with zipfile.ZipFile(ZIP) as z:
    name = [n for n in z.namelist() if n.endswith(".csv")][0]
    with z.open(name) as f:
        fao = pd.read_csv(f, encoding="latin-1", low_memory=False,
                          usecols=["Area","Item","Element","Year","Unit","Value"])

fao = fao[(fao["Item"].isin(TARGET_ITEMS)) & (fao["Element"].isin(TARGET_ELEMENTS))]
fao["ISO"] = fao["Area"].apply(name_to_iso2)
fao = fao.dropna(subset=["ISO","Value"])
fao["Annee"] = fao["Year"].astype(int)
fao["col"] = "fao_live_" + fao["Element"].map(TARGET_ELEMENTS) + "_" + fao["Item"].map(TARGET_ITEMS)
piv = fao.pivot_table(index=["ISO","Annee"], columns="col", values="Value", aggfunc="mean").reset_index()
print(f"   {piv.shape[0]} obs × {piv.shape[1]-2} features extraites")
print(f"   Features: {[c for c in piv.columns if c not in ['ISO','Annee']][:5]}...")


# ════════════════════════════════════════════════════════════════════
# BUILD V15 = V14 + FAO Live Animals
# ════════════════════════════════════════════════════════════════════
print("\n[BUILD] v15 = v14 + FAO Live Animals…")
df14 = pd.read_csv(f"{CLN}/shared/dataset_final_v14_couche1.csv", low_memory=False)
df14["ISO"] = df14["ISO"].astype(str)
df14["Annee"] = df14["Annee"].astype(int)
df15 = df14.merge(piv, on=["ISO","Annee"], how="left", suffixes=("","_DUP"))
df15 = df15.drop(columns=[c for c in df15.columns if c.endswith("_DUP")])
new_cols = [c for c in piv.columns if c not in ["ISO","Annee"]]
df15.to_csv(f"{CLN}/shared/dataset_final_v15_couche1.csv", index=False)
print(f"   V15 shape : {df15.shape}  (V14: {df14.shape})")
print(f"   {len(new_cols)} nouvelles features fao_live_*")


# ════════════════════════════════════════════════════════════════════
# (b) Target Encoding OOF — helper function
# ════════════════════════════════════════════════════════════════════
def add_oof_target_encoding(d, target, group_col="ISO", n_splits=5):
    """Add country-mean target as feature, computed OOF."""
    gkf = GroupKFold(n_splits=n_splits)
    enc = np.full(len(d), np.nan)
    for tr_idx, te_idx in gkf.split(d, groups=d[group_col]):
        means = d.iloc[tr_idx].groupby(group_col)[target].mean()
        enc[te_idx] = d.iloc[te_idx][group_col].map(means).values
    global_mean = d[target].mean()
    enc = np.where(np.isnan(enc), global_mean, enc)
    return enc


# ════════════════════════════════════════════════════════════════════
# Refit pipeline — XGB + optional OOF target encoding
# ════════════════════════════════════════════════════════════════════
def refit_xgb(df, tgt, extra_leaks, use_oof=False, label=""):
    drop_cols = {"ISO","Annee","T_ref","P_ref"}
    drop_cols |= {c for c in df.columns if c.startswith("target_")}
    for l in extra_leaks:
        drop_cols |= {c for c in df.columns if c == l or c.startswith(l)}
    feats = [c for c in df.columns if c not in drop_cols and df[c].dtype != object]
    d = df.dropna(subset=[tgt]).copy()
    feats = [c for c in feats if d[c].notna().sum() > 0]

    if use_oof:
        d["te_country_mean"] = add_oof_target_encoding(d, tgt)
        feats.append("te_country_mean")

    X_full, y, groups = d[feats], d[tgt], d["ISO"]
    X_full = X_full.replace([np.inf, -np.inf], np.nan)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(splitter.split(d, groups=groups))

    pipe_rf = Pipeline([("imp", SimpleImputer(strategy="median")),("sc", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=4,
                                      n_jobs=-1, random_state=42))])
    pipe_rf.fit(X_full.iloc[tr_idx], y.iloc[tr_idx])
    imp = pd.Series(pipe_rf.named_steps["rf"].feature_importances_,
                    index=X_full.columns).sort_values(ascending=False)
    sel = imp.head(150).index.tolist()
    if use_oof and "te_country_mean" not in sel:
        sel = ["te_country_mean"] + sel[:-1]
    X = X_full[sel]

    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),("sc", StandardScaler()),
        ("model", XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.04,
                                subsample=0.85, colsample_bytree=0.85,
                                random_state=42, n_jobs=-1, verbosity=0))])
    pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred = pipe.predict(X.iloc[te_idx])
    r2 = r2_score(y.iloc[te_idx], pred)
    mae = mean_absolute_error(y.iloc[te_idx], pred)
    return r2, mae, pipe, sel


# ════════════════════════════════════════════════════════════════════
# (a) REFIT CARCASS avec FAO Live Animals (yields blacklistés = honnête)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("(a) REFIT CARCASS — V15 (stocks/slaughtered) + yields blacklistés")
print("="*70)

CARCASS_LEAKS = {
    "target_pig_carcass":       ["livestock_pig_slaughtered","meat_pig_kg_pc","livestock_pig_carcass_kg",
                                  "fao_cl_yield_meat_of_pig_with_the_bone"],
    "target_sheepgoat_carcass": ["livestock_sheepgoat_slaughtered","meat_sheepgoat_kg_pc","livestock_sheepgoat_carcass_kg",
                                  "fao_cl_yield_sheep_and_goat_meat"],
    "target_cattle_carcass":    ["livestock_cattle_slaughtered","meat_beef_kg_pc","glw4_","livestock_cattle_carcass_kg",
                                  "fao_cl_yield_beef_and_buffalo_meat_pri","fao_cl_yield_meat_of_cattle_with_the_b"],
    "target_chicken_carcass":   ["livestock_chicken_slaughtered","meat_poultry_kg_pc","livestock_chicken_carcass_g",
                                  "fao_cl_yield_meat_of_chickens_fresh_or","fao_cl_yield_meat_poultry"],
}

results_a = []
for tgt, leaks in CARCASS_LEAKS.items():
    print(f"\n━━ {tgt} ━━")
    r2_no_oof, mae, pipe, sel = refit_xgb(df15, tgt, leaks, use_oof=False)
    r2_oof, mae_oof, pipe_oof, sel_oof = refit_xgb(df15, tgt, leaks, use_oof=True)
    print(f"   sans OOF : R²={r2_no_oof:+.4f}  MAE={mae:.3f}")
    print(f"   avec OOF : R²={r2_oof:+.4f}  MAE={mae_oof:.3f}")
    best_pipe, best_sel = (pipe_oof, sel_oof) if r2_oof > r2_no_oof else (pipe, sel)
    best_r2 = max(r2_no_oof, r2_oof)
    joblib.dump({"pipe": best_pipe, "features": best_sel, "sublayer": "v15_live_animals"},
                f"couche1_planete/models_cascade_v3/best_{tgt}.joblib")
    results_a.append({"target": tgt, "R2_v15_no_oof": round(r2_no_oof,3),
                       "R2_v15_oof": round(r2_oof,3), "best": round(best_r2,3)})

print("\n📊 (a) CARCASS V15 honnête")
print(pd.DataFrame(results_a).to_string(index=False))


# ════════════════════════════════════════════════════════════════════
# (b) TARGET ENCODING OOF — 10 cibles faibles
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("(b) TARGET ENCODING OOF — cibles faibles")
print("="*70)

WEAK_TARGETS = {
    "target_yield_eggplant":    [],
    "target_yield_rapeseed":    [],
    "target_yield_drypea":      [],
    "target_yield_strawberry":  [],
    "target_yield_apple":       [],
    "target_yield_drybean":     [],
    "target_yield_cotton":      [],
    "target_yield_groundnut":   [],
    "target_biodiversity_species": ["iucn_observations"],
    "target_powerplant_capacity_mw": ["powerplant_n_plants","powerplant_coal_mw","powerplant_gas_mw",
                                       "powerplant_oil_mw","powerplant_hydro_mw","powerplant_solar_mw","powerplant_wind_mw"],
}

results_b = []
for tgt, leaks in WEAK_TARGETS.items():
    if tgt not in df15.columns: continue
    r2_base, _, _, _ = refit_xgb(df15, tgt, leaks, use_oof=False)
    r2_oof, mae, pipe, sel = refit_xgb(df15, tgt, leaks, use_oof=True)
    gain = r2_oof - r2_base
    print(f"   {tgt:40s}  baseline={r2_base:+.3f}  OOF={r2_oof:+.3f}  Δ={gain:+.3f}")
    if r2_oof > r2_base:
        joblib.dump({"pipe": pipe, "features": sel, "sublayer": "v15_oof"},
                    f"couche1_planete/models_cascade_v3/best_{tgt}.joblib")
    results_b.append({"target": tgt, "R2_baseline": round(r2_base,3),
                       "R2_oof": round(r2_oof,3), "gain": round(gain,3)})

print("\n📊 (b) OOF target encoding")
print(pd.DataFrame(results_b).to_string(index=False))


# ════════════════════════════════════════════════════════════════════
# (c) POISSON REGRESSOR — target_seismic_activity
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("(c) POISSON REGRESSOR — target_seismic_activity")
print("="*70)

tgt = "target_seismic_activity"
leaks = ["eq_max_mag","eq_mean_mag","eq_mag_ge6","earthquake_count","earthquake_max_mag","eq_count"]
drop_cols = {"ISO","Annee","T_ref","P_ref"} | {c for c in df15.columns if c.startswith("target_")}
for l in leaks:
    drop_cols |= {c for c in df15.columns if c == l or c.startswith(l)}
feats = [c for c in df15.columns if c not in drop_cols and df15[c].dtype != object]
d = df15.dropna(subset=[tgt]).copy()
feats = [c for c in feats if d[c].notna().sum() > 0]
X_full = d[feats].replace([np.inf,-np.inf], np.nan)
y = d[tgt]
# target is log(count+1) → exponentiate back to count for Poisson
y_count = np.expm1(y).clip(lower=0).astype(int)
groups = d["ISO"]
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr_idx, te_idx = next(splitter.split(d, groups=groups))

# Feature select via RF
pipe_rf = Pipeline([("imp", SimpleImputer(strategy="median")),("sc", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=4,
                                  n_jobs=-1, random_state=42))])
pipe_rf.fit(X_full.iloc[tr_idx], y.iloc[tr_idx])
imp = pd.Series(pipe_rf.named_steps["rf"].feature_importances_,
                index=X_full.columns).sort_values(ascending=False)
sel = imp.head(80).index.tolist()
X = X_full[sel]

# Variant 1: XGB on log
xgb = Pipeline([("imp", SimpleImputer(strategy="median")),("sc", StandardScaler()),
    ("m", XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.04,
                        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbosity=0))])
xgb.fit(X.iloc[tr_idx], y.iloc[tr_idx])
r2_xgb = r2_score(y.iloc[te_idx], xgb.predict(X.iloc[te_idx]))

# Variant 2: XGB Poisson objective
xgb_p = Pipeline([("imp", SimpleImputer(strategy="median")),("sc", StandardScaler()),
    ("m", XGBRegressor(objective="count:poisson", n_estimators=600, max_depth=6, learning_rate=0.04,
                        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbosity=0))])
xgb_p.fit(X.iloc[tr_idx], y_count.iloc[tr_idx])
pred_p_count = xgb_p.predict(X.iloc[te_idx])
pred_p_log = np.log1p(pred_p_count)
r2_xgbp = r2_score(y.iloc[te_idx], pred_p_log)

# Variant 3: sklearn PoissonRegressor (linear)
pr = Pipeline([("imp", SimpleImputer(strategy="median")),("sc", StandardScaler()),
    ("m", PoissonRegressor(alpha=1.0, max_iter=500))])
pr.fit(X.iloc[tr_idx], y_count.iloc[tr_idx].clip(lower=0))
pred_pr_log = np.log1p(pr.predict(X.iloc[te_idx]).clip(min=0))
r2_pr = r2_score(y.iloc[te_idx], pred_pr_log)

print(f"   XGB log-target       : R² = {r2_xgb:+.4f}")
print(f"   XGB objective Poisson: R² = {r2_xgbp:+.4f}")
print(f"   sklearn Poisson lin  : R² = {r2_pr:+.4f}")

best_r2_c = max(r2_xgb, r2_xgbp, r2_pr)
best_model = {r2_xgb: ("xgb_log", xgb), r2_xgbp: ("xgb_poisson", xgb_p), r2_pr: ("poisson_lin", pr)}[best_r2_c]
print(f"   → meilleur : {best_model[0]} (R²={best_r2_c:+.4f})")
joblib.dump({"pipe": best_model[1], "features": sel, "sublayer": f"v15_{best_model[0]}"},
            f"couche1_planete/models_cascade_v3/best_{tgt}.joblib")


# ════════════════════════════════════════════════════════════════════
# SAVE COMBINED REPORT
# ════════════════════════════════════════════════════════════════════
all_res = {
    "carcass_v15": results_a,
    "oof_weak": results_b,
    "seismic": {"xgb_log": round(r2_xgb,4), "xgb_poisson": round(r2_xgbp,4),
                 "poisson_lin": round(r2_pr,4), "best": best_model[0]},
}
with open("couche1_planete/reports/3_chantiers_results.json","w") as f:
    json.dump(all_res, f, indent=2)
print("\n✓ Tout sauvegardé → couche1_planete/reports/3_chantiers_results.json")
