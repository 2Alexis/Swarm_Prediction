"""
train_stacking.py — Stacking meta-learner.

Niveau 1 (base) : modèles V6 déjà entraînés pour cibles à R² ≥ 0.6
  - target_forest_share (R²=0.86)
  - target_thermal_anomaly (R²=0.79)
  - target_child_mortality (R²=0.79)
  - target_fertility (R²=0.78)
  - target_birth_rate (R²=0.75)
  - target_yield_roots (R²=0.72)
  - target_life_expectancy (R²=0.70)
  - target_soil_degradation (R²=0.66)

Niveau 2 (meta) : entraîne pour les cibles FAIBLES avec :
  - Features V6 originales (top-50 par cible)
  - + Prédictions out-of-fold des modèles forts

Cibles cibles du stacking :
  - target_pop_growth
  - target_net_migration
  - target_yield_pulses / fruits / oilcrops agrégés
  - target_yield_orange, target_yield_mango, target_yield_avocado, target_yield_chickpea
  - target_disaster_damages_pct_gdp
  - target_stunting
  - target_pm25
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
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
DATA = "data/cleaned/dataset_final_v6.csv"
os.makedirs("models_stack", exist_ok=True)

df = pd.read_csv(DATA)
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
print(f"Dataset: {df.shape}\n")

# ── Cibles base (utilisées comme features) ──────────────────────────────
BASE_TARGETS = [
    "target_forest_share",
    "target_thermal_anomaly",
    "target_child_mortality",
    "target_fertility",
    "target_birth_rate",
    "target_life_expectancy",
    "target_soil_degradation",
    "target_yield_roots",
    "target_yield_cereals",
    "target_water_stress",
    "target_soil_moisture_root",
]

# Vérif présence cibles dans df (créées par train_v6.py)
for c in ["target_disaster_deaths", "target_disaster_affected"]:
    if "disaster_deaths" in df.columns and "target_disaster_deaths" not in df.columns:
        df["target_disaster_deaths"] = np.log1p(df["disaster_deaths"].clip(lower=0))
    if "disaster_affected" in df.columns and "target_disaster_affected" not in df.columns:
        df["target_disaster_affected"] = np.log1p(df["disaster_affected"].clip(lower=0))
if "stunting_pct" in df.columns and "target_stunting" not in df.columns:
    df["target_stunting"] = df["stunting_pct"]
if "Fertility_Rate" in df.columns and "target_fertility" not in df.columns:
    df["target_fertility"] = df["Fertility_Rate"]


# ── Out-of-fold predictions des modèles base (avec GroupKFold sur ISO) ────
print("[1] Génération OOF predictions des modèles base…")

ID_COLS = ["ISO", "Annee", "T_ref", "P_ref"]
TARGET_COLS = [c for c in df.columns if c.startswith("target_")]
YIELD_COLS = [c for c in df.columns if c.startswith("yield_") and "kgha" in c]

# Blacklist basique (mêmes règles que train_v6)
SOCIO_VARS = ["Child_Mort","Life_Exp","Pop_Growth","Birth_Rate","Death_Rate",
              "Net_Migration","HDI","GDP_pc","GDP_total_usd","Population",
              "Urban_pct","Inflation_CPI","Unemployment","Gini","Poverty_190",
              "Poverty_OWID","Debt_GDP","Trade_GDP","Hunger_Index",
              "Internet_pct","Mobile_subs","Electricity_pct","Energy_pc",
              "Health_GDP","Hospital_Beds","RD_GDP","Malaria","HIV",
              "Deaths_Communicable","Renew_Energy_pct","Energy_total",
              "Life_Exp_OWID","Fertility_Rate","Internet_OWID",
              "elec_generation_gwh","elec_renew_share",
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
              "pm25_annual","co_emissions","cereal_production_t","co2_per_capita_calc",
              "owid_births_total","owid_deaths_total","owid_crude_birth_rate",
              "owid_crude_death_rate","owid_pop_density","owid_agri_land_share",
              "births_per_1000_minus_deaths",
              "owid_co2_pc","owid_methane","owid_n2o","owid_ghg_by_sector",
              "ghi_owid","agri_output_usd"]

TARGET_SOURCE = {
    "target_forest_share":       "forest_share_pct",
    "target_thermal_anomaly":    "T_anomaly",
    "target_child_mortality":    "Child_Mort",
    "target_fertility":          "Fertility_Rate",
    "target_birth_rate":         "Birth_Rate",
    "target_life_expectancy":    "Life_Exp",
    "target_soil_degradation":   "Bilan_sols_kgha",
    "target_yield_roots":        "yield_roots_kgha",
    "target_yield_cereals":      "yield_cereals_kgha",
    "target_water_stress":       "Water_Withdrawal_pct",
    "target_soil_moisture_root": "nasa_gwetroot",
    "target_pop_growth":         "Pop_Growth",
    "target_net_migration":      "Net_Migration",
    "target_yield_pulses":       "yield_pulses_kgha",
    "target_yield_fruits":       "yield_fruits_kgha",
    "target_yield_oilcrops":     "yield_oilcrops_kgha",
    "target_yield_orange":       "yield_orange",
    "target_yield_mango":        "yield_mango",
    "target_yield_avocado":      "yield_avocado",
    "target_yield_chickpea":     "yield_chickpea",
    "target_stunting":           "stunting_pct",
    "target_pm25":               "pm25_annual",
    "target_disaster_deaths":    "disaster_deaths",
}

SOCIO_TARGETS = {"target_child_mortality","target_life_expectancy","target_pop_growth",
                 "target_stunting","target_fertility","target_pm25",
                 "target_birth_rate","target_death_rate","target_net_migration"}


def get_bl(target):
    bl = set(ID_COLS) | set(TARGET_COLS)
    src = TARGET_SOURCE.get(target)
    if src:
        bl |= {c for c in df.columns if c == src or c.startswith(src + "_") or c == src + "_ref"}
    if target.startswith("target_yield_"):
        bl |= set(YIELD_COLS) | {"cereals_prod_t", "cereals_area_ha",
                                  "cereal_production_t", "food_production_index", "cereal_yield"}
        for y in YIELD_COLS + ["cereal_production_t", "cereal_yield"]:
            bl |= {c for c in df.columns if c.startswith(y + "_")}
        # tous les yields specifiques aussi
        for c in df.columns:
            if c.startswith("yield_") and c != src:
                bl.add(c)
    if target in SOCIO_TARGETS:
        for v in SOCIO_VARS:
            bl |= {c for c in df.columns if c == v or c.startswith(v + "_")}
    return bl


def make_pre():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])


# Génère OOF predictions via GroupKFold (sur ISO) — pas de leak temporel
gkf = GroupKFold(n_splits=5)

def oof_predictions(target):
    d = df.dropna(subset=[target]).copy()
    if len(d) < 200:
        return None
    bl = get_bl(target)
    feats = [c for c in d.columns if c not in bl and d[c].dtype != object]
    feats = [c for c in feats if d[c].notna().sum() > 0]
    X, y = d[feats], d[target]
    oof = pd.Series(index=d.index, dtype=float)
    for tr, te in gkf.split(X, y, groups=d["ISO"]):
        pipe = Pipeline([("pre", make_pre()),
                         ("model", XGBRegressor(n_estimators=300, max_depth=6,
                                                 learning_rate=0.05, n_jobs=-1,
                                                 random_state=42, verbosity=0))])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof.iloc[te] = pipe.predict(X.iloc[te])
    return oof  # série de prédictions OOF indexée comme d


# Générer OOF pour BASE_TARGETS et stocker dans df
oof_cols = []
for bt in BASE_TARGETS:
    if bt not in df.columns: continue
    print(f"   OOF base : {bt}")
    oof = oof_predictions(bt)
    if oof is None: continue
    col = f"oof_{bt}"
    df[col] = np.nan
    df.loc[oof.index, col] = oof.values
    oof_cols.append(col)

print(f"\n   OOF cols créées : {oof_cols}\n")


# ── Cibles cibles du stacking ─────────────────────────────────────────────
WEAK_TARGETS = {
    "target_pop_growth":              "Croissance démographique",
    "target_net_migration":           "Migration nette",
    "target_death_rate":              "Taux de mortalité brut",
    "target_yield_pulses":            "Rendement légumineuses (agrégé)",
    "target_yield_fruits":            "Rendement fruits (agrégé)",
    "target_yield_oilcrops":          "Rendement oléagineux (agrégé)",
    "target_yield_vegetables":        "Rendement légumes (agrégé)",
    "target_yield_orange":            "Rendement Orange",
    "target_yield_mango":             "Rendement Mangue",
    "target_yield_avocado":           "Rendement Avocat",
    "target_yield_chickpea":          "Rendement Pois chiche",
    "target_yield_cabbage":           "Rendement Chou",
    "target_yield_lemon":             "Rendement Citron",
    "target_yield_grape":             "Rendement Raisin",
    "target_disaster_deaths":         "Décès catastrophes",
    "target_disaster_affected":       "Affectés catastrophes",
    "target_stunting":                "Retard de croissance",
    "target_tree_cover_loss":         "Perte couvert arboré",
}


def select_top(X, y, k=80):
    X = X.dropna(axis=1, how="all")
    if X.shape[1] <= k: return list(X.columns)
    Xp = make_pre().fit_transform(X)
    rf = RandomForestRegressor(n_estimators=80, max_depth=12,
                                min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(Xp, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(k).index.tolist()


splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
results = []

print("[2] Train cibles faibles avec STACKING…\n")
for tgt, label in WEAK_TARGETS.items():
    if tgt not in df.columns: continue
    d = df.dropna(subset=[tgt]).copy()
    if len(d) < 300: continue
    tr_idx, te_idx = next(splitter.split(d, groups=d["ISO"]))
    tr, te = d.iloc[tr_idx], d.iloc[te_idx]
    bl = get_bl(tgt)
    feats_base = [c for c in d.columns if c not in bl and d[c].dtype != object]
    # OOF cols : on ajoute SAUF si la cible base = cible target
    oof_to_use = [c for c in oof_cols if c.replace("oof_", "") != tgt]
    # Sécurité : pour cibles socio, exclure les OOF socio aussi
    if tgt in SOCIO_TARGETS:
        oof_to_use = [c for c in oof_to_use if c.replace("oof_", "") not in SOCIO_TARGETS]
    feats_with_oof = feats_base + oof_to_use

    Xtr_full = tr[feats_with_oof]
    Xte_full = te[feats_with_oof]
    ytr, yte = tr[tgt], te[tgt]

    selected = select_top(Xtr_full, ytr, k=80)
    Xtr, Xte = Xtr_full[selected], Xte_full[selected]

    n_oof_kept = sum(1 for f in selected if f.startswith("oof_"))
    print(f"🎯 {label} ({tgt})")
    print(f"   n_train={len(tr)} | feats base={len(feats_base)} +{len(oof_to_use)} OOF -> top {len(selected)}  (dont {n_oof_kept} OOF)")

    best = (-np.inf, None, None, None)
    # XGBoost optimisé
    for params in [
        {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05},
        {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.03},
    ]:
        try:
            pipe = Pipeline([("pre", make_pre()),
                             ("model", XGBRegressor(**params, subsample=0.8, colsample_bytree=0.8,
                                                    n_jobs=-1, random_state=42, verbosity=0))])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            r2 = r2_score(yte, pred)
            mae = mean_absolute_error(yte, pred)
            tag = f"XGB(n={params['n_estimators']},d={params['max_depth']})"
            if r2 > best[0]:
                best = (r2, tag, pipe, mae)
        except Exception: continue
    # RF aussi
    rf = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=4,
                               n_jobs=-1, random_state=42)
    pipe = Pipeline([("pre", make_pre()), ("model", rf)])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    r2 = r2_score(yte, pred)
    if r2 > best[0]: best = (r2, "RandomForest", pipe, mean_absolute_error(yte, pred))

    r2, bname, bpipe, bmae = best
    joblib.dump({"pipe": bpipe, "features": selected}, f"models_stack/best_{tgt}.joblib")
    print(f"   🏆 {bname}  R²={r2:+.4f}  MAE={bmae:.3f}\n")
    results.append({"Cible": label, "Technique": tgt, "Meilleur": bname,
                    "R² (stacking)": round(r2, 4),
                    "MAE": round(bmae, 3),
                    "N OOF utilisées": n_oof_kept})

out = pd.DataFrame(results)
out.to_csv("reports/stacking_results.csv", index=False)
print("\n📊 Résultats stacking :")
print(out.to_string(index=False))
