"""Retrain WAHIS HONNETE avec WHO Immunization features."""
import os, sys, io, warnings
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
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from build_dataset import custom_mappings, get_english_iso

def name_to_iso2(name):
    if pd.isna(name): return None
    s = str(name).strip().lower()
    code = custom_mappings.get(s)
    if code: return code
    return get_english_iso(name)


# 1. Charger v13 + merge WHO
df = pd.read_csv("data/cleaned/shared/dataset_final_v13_couche1.csv", low_memory=False)
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
imm = pd.read_csv("data/cleaned/demographie/who_immunization_coverage.csv")
df = df.merge(imm, on=["ISO","Annee"], how="left", suffixes=("","_DUP"))
df = df.drop(columns=[c for c in df.columns if c.endswith("_DUP")])
print(f"Shape after WHO merge: {df.shape}")
who_cols = [c for c in df.columns if c.startswith("who_") and c.endswith("_pct")]
print(f"WHO indicators ajoutés : {who_cols}")

# 2. WAHIS catégories
wahis = pd.read_csv("data/raw/misc/Données quantitatives 2026-06-24.csv",
                     low_memory=False, encoding="latin-1")
print(f"WAHIS cols: {list(wahis.columns)[:5]}")
# Renommer 1ère colonne (Année avec encoding bizarre) en "Annee_w"
wahis.columns = ["Annee_w" if i==0 else c for i,c in enumerate(wahis.columns)]
wahis["ISO"] = wahis["Pays"].apply(name_to_iso2)
wahis = wahis.dropna(subset=["ISO","Annee_w"])
wahis["Annee"] = pd.to_numeric(wahis["Annee_w"], errors="coerce").astype("Int64")
wahis = wahis.dropna(subset=["Annee"])
wahis["Annee"] = wahis["Annee"].astype(int)

def cat(m):
    if pd.isna(m): return "Other"
    m = str(m).lower()
    if any(k in m for k in ["bird","avian","poultry","newcastle"]): return "Avian"
    if any(k in m for k in ["foot","fmd","mouth"]): return "FMD"
    if any(k in m for k in ["swine","pig"]): return "Swine"
    if any(k in m for k in ["cattle","bovine","bse"]): return "Cattle"
    if any(k in m for k in ["lumpy","skin"]): return "Lumpy"
    return "Other"
wahis["Cat"] = wahis["Maladie"].apply(cat)
by_cat = wahis.groupby(["ISO","Annee","Cat"])["Outbreak_id"].nunique().unstack(fill_value=0).reset_index()
rename = {c: f"wahis_{c.lower()}_outbreaks" for c in by_cat.columns if c not in ["ISO","Annee"]}
by_cat = by_cat.rename(columns=rename)
df = df.merge(by_cat, on=["ISO","Annee"], how="left", suffixes=("","_DUP"))
df = df.drop(columns=[c for c in df.columns if c.endswith("_DUP")])

# 3. Train WAHIS HONNÊTE V2
print("\n🦠 WAHIS Honnête V2 (avec WHO Immunization)…\n")
tgt = "target_animal_disease_outbreaks"
LEAKS = ["wahis_outbreaks_total","wahis_diseases_unique","wahis_cases","wahis_deaths",
         "wahis_avian_outbreaks","wahis_fmd_outbreaks","wahis_swine_outbreaks",
         "wahis_cattle_outbreaks","wahis_lumpy_outbreaks","wahis_other_outbreaks"]
drop_cols = {"ISO","Annee","T_ref","P_ref"}
drop_cols |= {c for c in df.columns if c.startswith("target_")}
for l in LEAKS:
    drop_cols |= {c for c in df.columns if c == l or c.startswith(l)}
feats = [c for c in df.columns if c not in drop_cols and df[c].dtype != object]
d = df.dropna(subset=[tgt]).copy()
feats = [c for c in feats if d[c].notna().sum() > 0]

X_full, y, groups = d[feats], d[tgt], d["ISO"]
X_full = X_full.replace([np.inf, -np.inf], np.nan)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr_idx, te_idx = next(splitter.split(d, groups=groups))

# Feature selection
pipe_rf = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
    ("rf", RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=4,
                                    n_jobs=-1, random_state=42))])
pipe_rf.fit(X_full.iloc[tr_idx], y.iloc[tr_idx])
imp = pd.Series(pipe_rf.named_steps["rf"].feature_importances_,
                 index=X_full.columns).sort_values(ascending=False)
sel = imp.head(150).index.tolist()
X = X_full[sel]

# Train XGB optimisé
pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
    ("model", XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.04,
                            subsample=0.85, colsample_bytree=0.85,
                            random_state=42, n_jobs=-1, verbosity=0))])
pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
pred = pipe.predict(X.iloc[te_idx])
r2 = r2_score(y.iloc[te_idx], pred)
mae = mean_absolute_error(y.iloc[te_idx], pred)
print(f"   R² = {r2:+.4f}  MAE = {mae:.4f}  (n={len(d)})\n")

imp_f = pd.Series(pipe.named_steps["model"].feature_importances_,
                   index=sel).sort_values(ascending=False).head(15)
print(f"   Top 15 features :")
for f, v in imp_f.items():
    src = "🆕WHO" if f.startswith("who_") else ""
    print(f"     {f:50s} imp={v:.3f}  {src}")

# Combien de WHO dans top 50
top50 = imp.head(50).index.tolist()
who_in_top50 = [f for f in top50 if f.startswith("who_")]
print(f"\n   WHO immunization features dans TOP 50 : {len(who_in_top50)}")
for w in who_in_top50:
    print(f"     {w}")

joblib.dump({"pipe": pipe, "features": sel, "sublayer": "fix_honest_v2_who"},
            "couche1_planete/models_cascade_v3/best_target_animal_disease_outbreaks.joblib")
print(f"\n   ✓ Sauvegardé")
