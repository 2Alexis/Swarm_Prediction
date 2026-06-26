"""
train_v3.py — Entraîne tous les modèles sur dataset_final_v3.csv (avec datasets enrichis).
Mêmes règles anti-leak strictes que v2 + nouvelles cibles catastrophes.
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

DATA = "data/cleaned/dataset_final_v3.csv"
os.makedirs("models_v3", exist_ok=True)
os.makedirs("reports", exist_ok=True)

df = pd.read_csv(DATA)
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
print(f"Dataset v3: {df.shape}, {df['ISO'].nunique()} pays")

# Ajouter quelques cibles catastrophes (en log car distribution skewée)
for c in ["disaster_deaths", "disaster_affected", "disaster_damages_usd"]:
    if c in df.columns:
        df[f"target_{c}"] = np.log1p(df[c].clip(lower=0))

TARGETS = {
    "target_yield_cereals":      "Rendement céréales (FAO)",
    "target_yield_oilcrops":     "Rendement oléagineux (FAO)",
    "target_yield_pulses":       "Rendement légumineuses (FAO)",
    "target_yield_roots":        "Rendement racines/tubercules (FAO)",
    "target_yield_fruits":       "Rendement fruits (FAO)",
    "target_yield_vegetables":   "Rendement légumes (FAO)",
    "target_water_stress":       "Stress hydrique log(% prélèvement)",
    "target_soil_degradation":   "Dégradation du sol (kg/ha)",
    "target_thermal_anomaly":    "Anomalie thermique annuelle",
    "target_child_mortality":    "Mortalité infantile (WB)",
    "target_life_expectancy":    "Espérance de vie (WB)",
    "target_pop_growth":         "Croissance démographique (WB)",
    "target_disaster_deaths":    "Décès catastrophes log",
    "target_disaster_affected":  "Affectés catastrophes log",
    "target_disaster_damages_usd": "Dégâts économiques log USD",
}

TARGET_SOURCE = {
    "target_yield_cereals":      "yield_cereals_kgha",
    "target_yield_oilcrops":     "yield_oilcrops_kgha",
    "target_yield_pulses":       "yield_pulses_kgha",
    "target_yield_roots":        "yield_roots_kgha",
    "target_yield_fruits":       "yield_fruits_kgha",
    "target_yield_vegetables":   "yield_vegetables_kgha",
    "target_water_stress":       "Water_Withdrawal_pct",
    "target_soil_degradation":   "Bilan_sols_kgha",
    "target_thermal_anomaly":    "T_anomaly",
    "target_child_mortality":    "Child_Mort",
    "target_life_expectancy":    "Life_Exp",
    "target_pop_growth":         "Pop_Growth",
    "target_disaster_deaths":    "disaster_deaths",
    "target_disaster_affected":  "disaster_affected",
    "target_disaster_damages_usd": "disaster_damages_usd",
}

SOCIO_VARS = ["Child_Mort","Life_Exp","Pop_Growth","Birth_Rate","Death_Rate",
              "Net_Migration","HDI","GDP_pc","GDP_total_usd","Population",
              "Urban_pct","Inflation_CPI","Unemployment","Gini","Poverty_190",
              "Poverty_OWID","Debt_GDP","Trade_GDP","Hunger_Index",
              "Internet_pct","Mobile_subs","Electricity_pct","Energy_pc",
              "Health_GDP","Hospital_Beds","RD_GDP","Malaria","HIV",
              "Deaths_Communicable","Renew_Energy_pct","Energy_total",
              "Life_Exp_OWID","Fertility_Rate","Internet_OWID",
              "elec_generation_gwh","elec_renew_share"]
SOCIO_TARGETS = {"target_child_mortality","target_life_expectancy","target_pop_growth"}

DISASTER_VARS = ["disaster_deaths","disaster_affected","disaster_damages_usd","disaster_events"]
DISASTER_TARGETS = {"target_disaster_deaths","target_disaster_affected","target_disaster_damages_usd"}

ID_COLS = ["ISO","Annee","T_ref","P_ref"]
TARGET_COLS = [c for c in df.columns if c.startswith("target_")]
YIELD_COLS = [c for c in df.columns if c.startswith("yield_") and "kgha" in c]


def get_blacklist(target):
    bl = set(ID_COLS) | set(TARGET_COLS)
    src = TARGET_SOURCE.get(target)
    if src:
        bl |= {c for c in df.columns if c == src or c.startswith(src + "_")}
    if target.startswith("target_yield_"):
        bl |= set(YIELD_COLS) | {"cereals_prod_t", "cereals_area_ha"}
        # lags des yields autres aussi
        for y in YIELD_COLS:
            bl |= {c for c in df.columns if c.startswith(y)}
    if target in SOCIO_TARGETS:
        for v in SOCIO_VARS:
            bl |= {c for c in df.columns if c == v or c.startswith(v + "_")}
    if target in DISASTER_TARGETS:
        for v in DISASTER_VARS:
            bl |= {c for c in df.columns if c == v or c.startswith(v + "_")}
    return bl


def build_X(df_sub, target):
    bl = get_blacklist(target)
    feats = [c for c in df_sub.columns if c not in bl and df_sub[c].dtype != object]
    return df_sub[feats].copy(), feats


def make_pre():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])


splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
results = []

for tgt, label in TARGETS.items():
    if tgt not in df.columns:
        print(f"⏭  {tgt} absent du dataset")
        continue
    d = df.dropna(subset=[tgt]).copy()
    if len(d) < 200:
        print(f"⏭  {tgt} — pas assez de données ({len(d)})")
        continue
    tr_idx, te_idx = next(splitter.split(d, groups=d["ISO"]))
    tr, te = d.iloc[tr_idx], d.iloc[te_idx]
    Xtr, feats = build_X(tr, tgt)
    Xte, _ = build_X(te, tgt)
    ytr, yte = tr[tgt], te[tgt]

    print(f"\n🎯 {label} ({tgt})")
    print(f"   train={len(tr):,} pays={tr['ISO'].nunique()} | test={len(te):,} pays={te['ISO'].nunique()} | features={len(feats)}")

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=15,
                                              min_samples_leaf=4, n_jobs=-1, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1, verbosity=0),
    }
    best = (-np.inf, None, None, None)
    for name, m in models.items():
        pipe = Pipeline([("pre", make_pre()), ("model", m)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        r2 = r2_score(yte, pred)
        mae = mean_absolute_error(yte, pred)
        print(f"   [{name:13s}] R²={r2:+.4f}  MAE={mae:.3f}")
        if r2 > best[0]:
            best = (r2, name, pipe, mae)
    r2, bname, bpipe, bmae = best
    joblib.dump(bpipe, f"models_v3/best_{tgt}.joblib")
    print(f"   🏆 {bname}  R²={r2:+.4f}")
    results.append({"Cible": label, "Technique": tgt, "Meilleur": bname,
                    "R² (pays inconnus)": round(r2, 4),
                    "MAE": round(bmae, 3),
                    "N test": len(yte),
                    "N features": len(feats)})

out = pd.DataFrame(results)
out.to_csv("reports/tableau_resultats_v3.csv", index=False)
print("\n📊 Résultats v3 :")
print(out.to_string(index=False))
print("\n→ reports/tableau_resultats_v3.csv")
