"""
train_v4.py — Entraîne tous les modèles sur dataset_final_v4.csv (364 colonnes).
Mêmes règles anti-leak qu'avant + nouvelles cibles + jeu de features ENRICHI.
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
DATA = "data/cleaned/dataset_final_v4.csv"
os.makedirs("models_v4", exist_ok=True)
os.makedirs("reports", exist_ok=True)

df = pd.read_csv(DATA)
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
print(f"Dataset v4: {df.shape}, {df['ISO'].nunique()} pays")

# Cibles catastrophes (log)
for c in ["disaster_deaths", "disaster_affected", "disaster_damages_usd"]:
    if c in df.columns:
        df[f"target_{c}"] = np.log1p(df[c].clip(lower=0))
# Dégâts NORMALISÉS par PIB (vrai impact économique)
if "disaster_damages_usd" in df.columns and "GDP_total_usd" in df.columns:
    norm = df["disaster_damages_usd"] / df["GDP_total_usd"].clip(lower=1e6)
    df["target_disaster_damages_pct_gdp"] = np.log1p(norm.clip(lower=0, upper=10))

# Nouvelles cibles utiles ajoutées :
if "stunting_pct" in df.columns:
    df["target_stunting"] = df["stunting_pct"]
if "pm25_annual" in df.columns:
    df["target_pm25"] = df["pm25_annual"]
if "Fertility_Rate" in df.columns:
    df["target_fertility"] = df["Fertility_Rate"]
if "wb_food_production_index" in df.columns:
    df["target_food_production"] = df["wb_food_production_index"]

TARGETS = {
    "target_yield_cereals":         "Rendement céréales",
    "target_yield_oilcrops":        "Rendement oléagineux",
    "target_yield_pulses":          "Rendement légumineuses",
    "target_yield_roots":           "Rendement racines/tubercules",
    "target_yield_fruits":          "Rendement fruits",
    "target_yield_vegetables":      "Rendement légumes",
    "target_water_stress":          "Stress hydrique",
    "target_soil_degradation":     "Dégradation du sol",
    "target_thermal_anomaly":       "Anomalie thermique",
    "target_child_mortality":       "Mortalité infantile",
    "target_life_expectancy":       "Espérance de vie",
    "target_pop_growth":            "Croissance démographique",
    "target_disaster_deaths":       "Décès catastrophes",
    "target_disaster_affected":     "Affectés catastrophes",
    "target_disaster_damages_pct_gdp": "Dégâts catastrophes / PIB",
    "target_stunting":              "Retard de croissance enfants",
    "target_pm25":                  "PM2.5 (qualité air)",
    "target_fertility":             "Taux de fécondité",
    "target_food_production":       "Indice production alimentaire",
}

TARGET_SOURCE = {
    "target_yield_cereals":         "yield_cereals_kgha",
    "target_yield_oilcrops":        "yield_oilcrops_kgha",
    "target_yield_pulses":          "yield_pulses_kgha",
    "target_yield_roots":           "yield_roots_kgha",
    "target_yield_fruits":          "yield_fruits_kgha",
    "target_yield_vegetables":      "yield_vegetables_kgha",
    "target_water_stress":          "Water_Withdrawal_pct",
    "target_soil_degradation":     "Bilan_sols_kgha",
    "target_thermal_anomaly":       "T_anomaly",
    "target_child_mortality":       "Child_Mort",
    "target_life_expectancy":       "Life_Exp",
    "target_pop_growth":            "Pop_Growth",
    "target_disaster_deaths":       "disaster_deaths",
    "target_disaster_affected":     "disaster_affected",
    "target_disaster_damages_pct_gdp": "disaster_damages_usd",
    "target_stunting":              "stunting_pct",
    "target_pm25":                  "pm25_annual",
    "target_fertility":             "Fertility_Rate",
    "target_food_production":       "food_production_index",
}

# SOCIO élargi
SOCIO_VARS = ["Child_Mort","Life_Exp","Pop_Growth","Birth_Rate","Death_Rate",
              "Net_Migration","HDI","GDP_pc","GDP_total_usd","Population",
              "Urban_pct","Inflation_CPI","Unemployment","Gini","Poverty_190",
              "Poverty_OWID","Debt_GDP","Trade_GDP","Hunger_Index",
              "Internet_pct","Mobile_subs","Electricity_pct","Energy_pc",
              "Health_GDP","Hospital_Beds","RD_GDP","Malaria","HIV",
              "Deaths_Communicable","Renew_Energy_pct","Energy_total",
              "Life_Exp_OWID","Fertility_Rate","Internet_OWID",
              "elec_generation_gwh","elec_renew_share",
              # Nouveaux socio v4
              "stunting_pct","wasting_pct","overweight_pct","safe_water_pct","sanitation_pct",
              "physicians_per_1000","adult_mortality_male","adult_mortality_female",
              "infant_deaths_total","school_primary_enrollment","school_secondary_enrollment",
              "adult_literacy_pct","employ_agri_pct","employ_industry_pct","employ_services_pct",
              "energy_use_per_cap","broadband_per_100","dependency_ratio",
              "dependency_young","dependency_old","pop_65_plus_pct","pop_under14_pct",
              "urban_growth","schooling_years","meat_consumption_pc","extreme_poverty_pct",
              "pop_density_per_km2","gdp_per_arable_ha","urban_pop_abs",
              "elec_gen_per_capita","disaster_deaths_per_million","disaster_affected_per_capita",
              "malnutrition_compound","school_total_enrollment",
              "agri_value_pct_gdp","manuf_value_pct_gdp","services_value_pct_gdp",
              "pm25_annual","co_emissions","cereal_production_t","co2_per_capita_calc"]
SOCIO_TARGETS = {"target_child_mortality","target_life_expectancy","target_pop_growth",
                 "target_stunting","target_fertility","target_food_production","target_pm25"}

DISASTER_VARS = ["disaster_deaths","disaster_affected","disaster_damages_usd","disaster_events"]
DISASTER_TARGETS = {"target_disaster_deaths","target_disaster_affected",
                    "target_disaster_damages_pct_gdp"}

ID_COLS = ["ISO","Annee","T_ref","P_ref"]
TARGET_COLS = [c for c in df.columns if c.startswith("target_")]
YIELD_COLS = [c for c in df.columns if c.startswith("yield_") and "kgha" in c]


# Sources brutes ADDITIONNELLES à drop par cible (leaks subtils)
EXTRA_LEAKS = {
    # WB cereal_yield = essentiellement la même valeur que yield_cereals_kgha
    "target_yield_cereals":   ["cereal_yield", "cereal_production_t", "food_production_index"],
    "target_yield_oilcrops":  ["food_production_index"],
    "target_yield_pulses":    ["food_production_index"],
    "target_yield_roots":     ["food_production_index"],
    "target_yield_fruits":    ["food_production_index"],
    "target_yield_vegetables":["food_production_index"],
    "target_water_stress":    ["freshwater_withdraw_total", "freshwater_internal_per_cap",
                               "water_stress_ratio_raw"],
    "target_food_production": ["cereal_production_t", "cereal_yield",
                               "wb_food_production_index", "agri_value_pct_gdp"],
    "target_stunting":        ["wasting_pct", "overweight_pct", "malnutrition_compound",
                               "infant_deaths_total", "Hunger_Index"],
    "target_fertility":       ["Birth_Rate", "Death_Rate", "Pop_Growth"],
}

def get_blacklist(target):
    bl = set(ID_COLS) | set(TARGET_COLS)
    src = TARGET_SOURCE.get(target)
    if src:
        bl |= {c for c in df.columns if c == src or c.startswith(src + "_")}
    # Extra leaks par cible
    for extra in EXTRA_LEAKS.get(target, []):
        bl |= {c for c in df.columns if c == extra or c.startswith(extra + "_")}
    if target.startswith("target_yield_"):
        bl |= set(YIELD_COLS) | {"cereals_prod_t", "cereals_area_ha",
                                  "cereal_production_t", "food_production_index",
                                  "cereal_yield"}
        for y in YIELD_COLS + ["cereal_production_t", "cereal_yield", "food_production_index"]:
            bl |= {c for c in df.columns if c.startswith(y + "_")}
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
        continue
    d = df.dropna(subset=[tgt]).copy()
    if len(d) < 200:
        print(f"⏭  {tgt} ({len(d)} obs)")
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
        "XGBoost": XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1, verbosity=0),
    }
    best = (-np.inf, None, None, None)
    for name, m in models.items():
        try:
            pipe = Pipeline([("pre", make_pre()), ("model", m)])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            r2 = r2_score(yte, pred)
            mae = mean_absolute_error(yte, pred)
            print(f"   [{name:13s}] R²={r2:+.4f}  MAE={mae:.3f}")
            if r2 > best[0]:
                best = (r2, name, pipe, mae)
        except Exception as e:
            print(f"   ✗ {name}: {e}")
    if best[2] is None: continue
    r2, bname, bpipe, bmae = best
    joblib.dump(bpipe, f"models_v4/best_{tgt}.joblib")
    print(f"   🏆 {bname}  R²={r2:+.4f}")
    results.append({"Cible": label, "Technique": tgt, "Meilleur": bname,
                    "R² (pays inconnus)": round(r2, 4),
                    "MAE": round(bmae, 3),
                    "N test": len(yte),
                    "N features": len(feats)})

out = pd.DataFrame(results)
out.to_csv("reports/tableau_resultats_v4.csv", index=False)
print("\n📊 Résultats v4 :")
print(out.to_string(index=False))
print("\n→ reports/tableau_resultats_v4.csv")
