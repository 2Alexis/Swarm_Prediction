"""Fix : blacklist sources + skip Poisson linéaire."""
import os, sys, io, warnings, json
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
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

CLN = "data/cleaned"
df = pd.read_csv(f"{CLN}/shared/dataset_final_v15_couche1.csv", low_memory=False)
print(f"V15 shape: {df.shape}")

def add_oof_te(d, target, group="ISO", n=5):
    gkf = GroupKFold(n_splits=n)
    enc = np.full(len(d), np.nan)
    for tr, te in gkf.split(d, groups=d[group]):
        m = d.iloc[tr].groupby(group)[target].mean()
        enc[te] = d.iloc[te][group].map(m).values
    return np.where(np.isnan(enc), d[target].mean(), enc)

def refit(df, tgt, leaks, use_oof=False, model_kw=None):
    drop = {"ISO","Annee","T_ref","P_ref"} | {c for c in df.columns if c.startswith("target_")}
    for l in leaks:
        drop |= {c for c in df.columns if c == l or c.startswith(l)}
    feats = [c for c in df.columns if c not in drop and df[c].dtype != object]
    d = df.dropna(subset=[tgt]).copy()
    feats = [c for c in feats if d[c].notna().sum() > 0]
    if use_oof:
        d["te_country_mean"] = add_oof_te(d, tgt)
        feats.append("te_country_mean")
    X_full = d[feats].replace([np.inf,-np.inf], np.nan)
    y, groups = d[tgt], d["ISO"]
    sp = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(sp.split(d, groups=groups))

    rf = Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
        ("rf",RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=4,
            n_jobs=-1, random_state=42))])
    rf.fit(X_full.iloc[tr], y.iloc[tr])
    imp = pd.Series(rf.named_steps["rf"].feature_importances_,
        index=X_full.columns).sort_values(ascending=False)
    sel = imp.head(150).index.tolist()
    if use_oof and "te_country_mean" not in sel:
        sel = ["te_country_mean"] + sel[:-1]
    X = X_full[sel]

    kw = model_kw or dict(n_estimators=600, max_depth=6, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbosity=0)
    pipe = Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
        ("m",XGBRegressor(**kw))])
    pipe.fit(X.iloc[tr], y.iloc[tr])
    pred = pipe.predict(X.iloc[te])
    return r2_score(y.iloc[te], pred), mean_absolute_error(y.iloc[te], pred), pipe, sel


# ── (a) CARCASS V15 honnête ─────────────────────────────────────────
print("\n" + "="*70 + "\n(a) CARCASS V15 — yields FAO blacklistés (honnête)\n" + "="*70)
CARCASS = {
    "target_pig_carcass":       ["livestock_pig_slaughtered","meat_pig_kg_pc","livestock_pig_carcass_kg",
                                  "fao_cl_yield_meat_of_pig_with_the_bone"],
    "target_sheepgoat_carcass": ["livestock_sheepgoat_slaughtered","meat_sheepgoat_kg_pc","livestock_sheepgoat_carcass_kg",
                                  "fao_cl_yield_sheep_and_goat_meat","fao_cl_yield_meat_of_sheep_fresh_or_ch",
                                  "fao_cl_yield_meat_of_goat_fresh_or_chi"],
    "target_cattle_carcass":    ["livestock_cattle_slaughtered","meat_beef_kg_pc","glw4_","livestock_cattle_carcass_kg",
                                  "fao_cl_yield_beef_and_buffalo_meat_pri","fao_cl_yield_meat_of_cattle_with_the_b"],
    "target_chicken_carcass":   ["livestock_chicken_slaughtered","meat_poultry_kg_pc","livestock_chicken_carcass_g",
                                  "fao_cl_yield_meat_of_chickens_fresh_or","fao_cl_yield_meat_poultry"],
}
res_a = []
for tgt, leaks in CARCASS.items():
    r2a,_,_,_ = refit(df, tgt, leaks, use_oof=False)
    r2b,mae,pipe,sel = refit(df, tgt, leaks, use_oof=True)
    print(f"   {tgt:30s}  base={r2a:+.3f}   +OOF={r2b:+.3f}   Δ={r2b-r2a:+.3f}")
    best = max(r2a, r2b)
    if r2b > r2a:
        joblib.dump({"pipe":pipe,"features":sel,"sublayer":"v15_live_oof"},
                    f"couche1_planete/models_cascade_v3/best_{tgt}.joblib")
    res_a.append({"target":tgt,"R2_v15":round(r2a,3),"R2_v15_oof":round(r2b,3),"best":round(best,3)})

# ── (b) WEAK YIELDS — blacklist source yield_X ─────────────────────
print("\n" + "="*70 + "\n(b) OOF target encoding — cibles faibles\n" + "="*70)
WEAK = {
    "target_yield_eggplant":    ["yield_eggplant","spam_yield_eggp","suit_eggplant","suitability_eggplant"],
    "target_yield_rapeseed":    ["yield_rapeseed","spam_yield_rape","suit_rapeseed","suitability_rapeseed"],
    "target_yield_drypea":      ["yield_drypea","spam_yield_pea","suit_drypea","suitability_drypea"],
    "target_yield_strawberry":  ["yield_strawberry","spam_yield_stra","suit_strawberry","suitability_strawberry"],
    "target_yield_apple":       ["yield_apple","spam_yield_appl","suit_apple","suitability_apple"],
    "target_yield_drybean":     ["yield_drybean","spam_yield_bean","suit_drybean","suitability_drybean"],
    "target_yield_cotton":      ["yield_cotton","spam_yield_cott","suit_cotton","suitability_cotton"],
    "target_yield_groundnut":   ["yield_groundnut","spam_yield_grou","suit_groundnut","suitability_groundnut"],
    "target_biodiversity_species": ["iucn_observations","iucn_species_count"],
}
res_b = []
for tgt, leaks in WEAK.items():
    r2a,_,_,_ = refit(df, tgt, leaks, use_oof=False)
    r2b,mae,pipe,sel = refit(df, tgt, leaks, use_oof=True)
    print(f"   {tgt:32s}  base={r2a:+.3f}   +OOF={r2b:+.3f}   Δ={r2b-r2a:+.3f}")
    if r2b > r2a:
        joblib.dump({"pipe":pipe,"features":sel,"sublayer":"v15_oof"},
                    f"couche1_planete/models_cascade_v3/best_{tgt}.joblib")
    res_b.append({"target":tgt,"R2_base":round(r2a,3),"R2_oof":round(r2b,3),"gain":round(r2b-r2a,3)})

# ── (c) Poisson seismic ─────────────────────────────────────────────
print("\n" + "="*70 + "\n(c) Seismic — XGB log vs XGB Poisson\n" + "="*70)
tgt = "target_seismic_activity"
leaks = ["eq_max_mag","eq_mean_mag","eq_mag_ge6","earthquake_count","earthquake_max_mag","eq_count"]
r2_log, mae_log, pipe_log, sel = refit(df, tgt, leaks, use_oof=False)

# XGB Poisson on raw count
drop = {"ISO","Annee","T_ref","P_ref"} | {c for c in df.columns if c.startswith("target_")}
for l in leaks:
    drop |= {c for c in df.columns if c == l or c.startswith(l)}
feats = [c for c in df.columns if c not in drop and df[c].dtype != object]
d = df.dropna(subset=[tgt]).copy()
feats = [c for c in feats if d[c].notna().sum() > 0]
X_full = d[feats].replace([np.inf,-np.inf], np.nan)
y_log = d[tgt]
y_count = np.expm1(y_log).clip(lower=0).round().astype(int)
groups = d["ISO"]
sp = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr, te = next(sp.split(d, groups=groups))

X = X_full[sel]
xgb_p = Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
    ("m",XGBRegressor(objective="count:poisson", n_estimators=600, max_depth=6, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbosity=0))])
xgb_p.fit(X.iloc[tr], y_count.iloc[tr])
pred_count = xgb_p.predict(X.iloc[te])
pred_log = np.log1p(np.clip(pred_count, 0, 1e6))
r2_pois = r2_score(y_log.iloc[te], pred_log)

# Variant: XGB Poisson + OOF
d["te"] = add_oof_te(d, tgt)
feats_oof = sel + ["te"]
Xo = d[feats_oof].replace([np.inf,-np.inf], np.nan)
xgb_po = Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
    ("m",XGBRegressor(objective="count:poisson", n_estimators=600, max_depth=6, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbosity=0))])
xgb_po.fit(Xo.iloc[tr], y_count.iloc[tr])
pred_co = xgb_po.predict(Xo.iloc[te])
r2_pois_oof = r2_score(y_log.iloc[te], np.log1p(np.clip(pred_co, 0, 1e6)))

# Variant: XGB log + OOF
r2_log_oof, _, pipe_log_oof, sel_oof = refit(df, tgt, leaks, use_oof=True)

print(f"   XGB log target          : R²={r2_log:+.4f}")
print(f"   XGB log + OOF country   : R²={r2_log_oof:+.4f}")
print(f"   XGB Poisson on count    : R²={r2_pois:+.4f}")
print(f"   XGB Poisson + OOF       : R²={r2_pois_oof:+.4f}")

best = max(r2_log, r2_log_oof, r2_pois, r2_pois_oof)
print(f"   → meilleur : R²={best:+.4f}")

# Save best
options = {"xgb_log": (r2_log, pipe_log, sel),
            "xgb_log_oof": (r2_log_oof, pipe_log_oof, sel_oof),
            "xgb_poisson": (r2_pois, xgb_p, sel),
            "xgb_poisson_oof": (r2_pois_oof, xgb_po, feats_oof)}
best_name = max(options, key=lambda k: options[k][0])
_, best_pipe, best_sel = options[best_name]
joblib.dump({"pipe":best_pipe,"features":best_sel,"sublayer":f"v15_{best_name}"},
            f"couche1_planete/models_cascade_v3/best_{tgt}.joblib")
print(f"   → sauvegardé : {best_name}")

# ── BILAN ───────────────────────────────────────────────────────────
print("\n\n" + "="*70 + "\n📊 BILAN FINAL\n" + "="*70)
print("\n(a) CARCASS V15 honnête :")
print(pd.DataFrame(res_a).to_string(index=False))
print("\n(b) OOF target encoding cibles faibles :")
print(pd.DataFrame(res_b).to_string(index=False))
print("\n(c) SEISMIC :")
print(f"  XGB log         : {r2_log:+.4f}")
print(f"  XGB log + OOF   : {r2_log_oof:+.4f}")
print(f"  XGB Poisson     : {r2_pois:+.4f}")
print(f"  XGB Poisson+OOF : {r2_pois_oof:+.4f}")

all_res = {"carcass": res_a, "weak_oof": res_b,
            "seismic": {"xgb_log":round(r2_log,4),"xgb_log_oof":round(r2_log_oof,4),
                         "xgb_poisson":round(r2_pois,4),"xgb_poisson_oof":round(r2_pois_oof,4),
                         "best":best_name}}
with open("couche1_planete/reports/3_chantiers_results.json","w") as f:
    json.dump(all_res, f, indent=2)
print("\n✓ → couche1_planete/reports/3_chantiers_results.json")
