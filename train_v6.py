"""
train_v6.py — Entraîne sur dataset_final_v6.csv avec FEATURE SELECTION par cible.

Pipeline pour chaque cible :
  1. Anti-leak strict (blacklist sources + lags + circulaires)
  2. Imputation médiane + scaling
  3. Feature selection : top-N par mutual_info_regression (rapide, non-linéaire)
     ou par RandomForest importance pour cibles avec >2000 obs
  4. Train Ridge / RandomForest / XGBoost sur les top-N features
  5. Garder le meilleur
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
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
DATA = "data/cleaned/dataset_final_v6.csv"
os.makedirs("models_v6", exist_ok=True)
os.makedirs("reports", exist_ok=True)

TOP_N_FEATURES = 100  # Feature selection top-N

df = pd.read_csv(DATA)
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
print(f"Dataset v6: {df.shape}, {df['ISO'].nunique()} pays\n")

# Cibles catastrophes
for c in ["disaster_deaths", "disaster_affected"]:
    if c in df.columns:
        df[f"target_{c}"] = np.log1p(df[c].clip(lower=0))
if "stunting_pct" in df.columns: df["target_stunting"] = df["stunting_pct"]
if "pm25_annual" in df.columns:  df["target_pm25"] = df["pm25_annual"]
if "Fertility_Rate" in df.columns: df["target_fertility"] = df["Fertility_Rate"]

# Cibles RENDEMENT SPÉCIFIQUE
SPECIFIC_CROPS_LIST = [c.replace("target_", "") for c in df.columns
                        if c.startswith("target_yield_") and c not in (
                          "target_yield_cereals","target_yield_oilcrops","target_yield_pulses",
                          "target_yield_roots","target_yield_fruits","target_yield_vegetables")]

TARGETS = {
    # Rendements familles
    "target_yield_cereals":      "Rendement céréales",
    "target_yield_oilcrops":     "Rendement oléagineux (agrégé)",
    "target_yield_pulses":       "Rendement légumineuses (agrégé)",
    "target_yield_roots":        "Rendement racines/tubercules",
    "target_yield_fruits":       "Rendement fruits (agrégé)",
    "target_yield_vegetables":   "Rendement légumes (agrégé)",
    # Cibles environnementales
    "target_water_stress":       "Stress hydrique",
    "target_soil_degradation":   "Dégradation du sol",
    "target_thermal_anomaly":    "Anomalie thermique",
    "target_soil_moisture_root": "Humidité sol racinaire",
    "target_forest_share":       "% forêt national",
    "target_tree_cover_loss":    "Perte couvert arboré (log)",
    # Démographie
    "target_child_mortality":    "Mortalité infantile",
    "target_life_expectancy":    "Espérance de vie",
    "target_pop_growth":         "Croissance démographique",
    "target_birth_rate":         "Taux de natalité",
    "target_death_rate":         "Taux de mortalité brut",
    "target_net_migration":      "Migration nette",
    "target_fertility":          "Taux de fécondité",
    # Santé/Société
    "target_stunting":           "Retard de croissance enfants",
    "target_pm25":               "PM2.5 air",
    # Catastrophes
    "target_disaster_deaths":    "Décès catastrophes",
    "target_disaster_affected":  "Affectés catastrophes",
}

# Ajout des cibles spécifiques par culture (avec libellé clair)
SPECIFIC_LABELS = {
    "yield_soybeans": "Soja", "yield_rapeseed": "Colza", "yield_sunflower": "Tournesol",
    "yield_groundnut": "Arachide", "yield_olives": "Olives", "yield_sesame": "Sésame",
    "yield_coconut": "Coco", "yield_cotton": "Coton",
    "yield_apple": "Pomme", "yield_banana": "Banane", "yield_orange": "Orange",
    "yield_mango": "Mangue", "yield_pineapple": "Ananas", "yield_grape": "Raisin",
    "yield_avocado": "Avocat", "yield_lemon": "Citron", "yield_peach": "Pêche",
    "yield_pear": "Poire", "yield_strawberry": "Fraise", "yield_watermelon": "Pastèque",
    "yield_dates": "Datte", "yield_apricot": "Abricot", "yield_cherry": "Cerise",
    "yield_plum": "Prune",
    "yield_tomato": "Tomate", "yield_potato": "Pomme de terre",
    "yield_onion": "Oignon", "yield_cabbage": "Chou", "yield_carrot": "Carotte",
    "yield_cucumber": "Concombre", "yield_eggplant": "Aubergine",
    "yield_cauliflower": "Chou-fleur", "yield_lettuce": "Laitue",
    "yield_chickpea": "Pois chiche", "yield_drybean": "Haricot sec", "yield_drypea": "Pois sec",
}
for crop_col in SPECIFIC_LABELS:
    tgt = f"target_{crop_col}"
    if tgt in df.columns:
        TARGETS[tgt] = f"Rendement {SPECIFIC_LABELS[crop_col]}"

# Sources brutes
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
    "target_birth_rate":         "Birth_Rate",
    "target_death_rate":         "Death_Rate",
    "target_net_migration":      "Net_Migration",
    "target_disaster_deaths":    "disaster_deaths",
    "target_disaster_affected":  "disaster_affected",
    "target_stunting":           "stunting_pct",
    "target_pm25":               "pm25_annual",
    "target_fertility":          "Fertility_Rate",
    "target_soil_moisture_root": "nasa_gwetroot",
    "target_forest_share":       "forest_share_pct",
    "target_tree_cover_loss":    "tree_cover_loss_ha",
}
# Sources brutes pour cibles spécifiques (= la colonne correspondante)
for crop_col in SPECIFIC_LABELS:
    TARGET_SOURCE[f"target_{crop_col}"] = crop_col

# Extra leaks
EXTRA_LEAKS = {
    "target_yield_cereals":   ["cereal_yield", "cereal_production_t", "food_production_index",
                                "owid_cereal_production"],
    "target_yield_oilcrops":  ["food_production_index"],
    "target_yield_pulses":    ["food_production_index"],
    "target_yield_roots":     ["food_production_index"],
    "target_yield_fruits":    ["food_production_index"],
    "target_yield_vegetables":["food_production_index"],
    "target_water_stress":    ["freshwater_withdraw_total", "freshwater_internal_per_cap",
                                "water_stress_ratio_raw", "owid_water_stress"],
    "target_stunting":        ["wasting_pct", "overweight_pct", "malnutrition_compound",
                                "infant_deaths_total", "Hunger_Index", "ghi_owid"],
    "target_fertility":       ["Birth_Rate", "Death_Rate", "Pop_Growth",
                                "owid_crude_birth_rate", "owid_crude_death_rate"],
    "target_birth_rate":      ["Death_Rate", "Net_Migration", "Pop_Growth", "Fertility_Rate",
                                "owid_crude_birth_rate", "owid_crude_death_rate",
                                "owid_births_total", "owid_deaths_total",
                                "births_per_1000_minus_deaths"],
    "target_death_rate":      ["Birth_Rate", "Net_Migration", "Pop_Growth", "Fertility_Rate",
                                "owid_crude_birth_rate", "owid_crude_death_rate",
                                "owid_births_total", "owid_deaths_total",
                                "births_per_1000_minus_deaths",
                                "adult_mortality_male", "adult_mortality_female",
                                "infant_deaths_total"],
    "target_net_migration":   ["Birth_Rate", "Death_Rate", "Pop_Growth", "Fertility_Rate",
                                "owid_crude_birth_rate", "owid_crude_death_rate",
                                "owid_births_total", "owid_deaths_total",
                                "refugees_origin", "refugees_destination",
                                "refugees_origin_owid", "idps_conflict"],
    "target_soil_moisture_root": ["nasa_gwettop", "nasa_gwetprof",
                                  "soil_moisture_deficit", "combined_drought_index",
                                  "heat_drought_stress", "soil_moisture_top_root_ratio"],
    "target_forest_share":    ["forest_area_km2", "forest_change",
                                "tree_cover_loss_ha", "tree_cover_loss_cumul5y",
                                "deforestation_annual", "deforestation_pct_annual",
                                "forest_per_capita_km2"],
    "target_tree_cover_loss": ["forest_change", "forest_share_pct", "forest_area_km2",
                                "deforestation_annual", "deforestation_pct_annual",
                                "tree_cover_loss_cumul5y"],
    "target_pm25":            ["who_pm25_mean", "who_ambient_air_deaths_rate"],
    "target_thermal_anomaly": ["owid_temp_anomaly"],  # OWID = même valeur source FAO
}

# Pour chaque cible spécifique : blacklister TOUS les autres yields (croissés)
SPECIFIC_YIELDS = list(SPECIFIC_LABELS.keys())
for c in SPECIFIC_YIELDS:
    EXTRA_LEAKS[f"target_{c}"] = (SPECIFIC_YIELDS +
                                  ["yield_cereals_kgha","yield_oilcrops_kgha","yield_pulses_kgha",
                                   "yield_roots_kgha","yield_fruits_kgha","yield_vegetables_kgha",
                                   "yield_citrus_kgha","yield_treenuts_kgha","yield_fibres_kgha",
                                   "cereal_yield","cereal_production_t","food_production_index"])

# Variables SOCIO
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
              # nouveau v6
              "owid_co2_pc","owid_methane","owid_n2o","owid_ghg_by_sector",
              "ghi_owid","agri_output_usd"]

SOCIO_TARGETS = {"target_child_mortality","target_life_expectancy","target_pop_growth",
                 "target_stunting","target_fertility","target_pm25",
                 "target_birth_rate","target_death_rate","target_net_migration"}

DISASTER_VARS = ["disaster_deaths","disaster_affected","disaster_damages_usd","disaster_events"]
DISASTER_TARGETS = {"target_disaster_deaths","target_disaster_affected"}

ID_COLS = ["ISO","Annee","T_ref","P_ref"]
TARGET_COLS = [c for c in df.columns if c.startswith("target_")]
YIELD_COLS = [c for c in df.columns if c.startswith("yield_") and "kgha" in c]


def get_blacklist(target):
    bl = set(ID_COLS) | set(TARGET_COLS)
    src = TARGET_SOURCE.get(target)
    if src:
        bl |= {c for c in df.columns if c == src or c.startswith(src + "_") or c == src + "_ref"}
    for extra in EXTRA_LEAKS.get(target, []):
        bl |= {c for c in df.columns if c == extra or c.startswith(extra + "_") or c == extra + "_ref"}
    if target.startswith("target_yield_"):
        # Sauf si cible spécifique, dans ce cas EXTRA_LEAKS déjà géré
        if target not in [f"target_{c}" for c in SPECIFIC_LABELS]:
            bl |= set(YIELD_COLS) | {"cereals_prod_t", "cereals_area_ha",
                                     "cereal_production_t", "food_production_index",
                                     "cereal_yield"}
            for y in YIELD_COLS + ["cereal_production_t", "cereal_yield"]:
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


def select_features(X, y, k=TOP_N_FEATURES):
    """Sélection top-k via RandomForest importance (rapide, robuste)."""
    # Drop colonnes 100% NaN (sinon SimpleImputer les supprime silencieusement)
    X = X.dropna(axis=1, how="all")
    if X.shape[1] <= k:
        return list(X.columns)
    pre = make_pre()
    Xp = pre.fit_transform(X)
    rf = RandomForestRegressor(n_estimators=80, max_depth=12,
                               min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(Xp, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(k).index.tolist()


splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
results = []
feature_imps = {}

for tgt, label in TARGETS.items():
    if tgt not in df.columns:
        continue
    d = df.dropna(subset=[tgt]).copy()
    if len(d) < 200:
        print(f"⏭  {tgt} ({len(d)} obs)")
        continue
    tr_idx, te_idx = next(splitter.split(d, groups=d["ISO"]))
    tr, te = d.iloc[tr_idx], d.iloc[te_idx]
    Xtr_full, feats_all = build_X(tr, tgt)
    Xte_full, _ = build_X(te, tgt)
    ytr, yte = tr[tgt], te[tgt]

    # FEATURE SELECTION sur train uniquement
    n_target_feat = min(TOP_N_FEATURES, max(20, int(len(tr) * 0.05)))  # adapt to data size
    selected = select_features(Xtr_full, ytr, k=n_target_feat)
    feature_imps[tgt] = selected[:20]
    Xtr = Xtr_full[selected]
    Xte = Xte_full[selected]

    print(f"\n🎯 {label} ({tgt})")
    print(f"   train={len(tr):,} | test={len(te):,} | feats_total={len(feats_all)} → SELECTION top {len(selected)}")

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=15,
                                              min_samples_leaf=4, n_jobs=-1, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
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
    joblib.dump({"pipe": bpipe, "features": selected}, f"models_v6/best_{tgt}.joblib")
    print(f"   🏆 {bname}  R²={r2:+.4f}")
    results.append({"Cible": label, "Technique": tgt, "Meilleur": bname,
                    "R² (pays inconnus)": round(r2, 4),
                    "MAE": round(bmae, 3),
                    "N test": len(yte),
                    "N feats sélectionnées": len(selected)})

out = pd.DataFrame(results)
out.to_csv("reports/tableau_resultats_v6.csv", index=False)
print("\n📊 Résultats v6 :")
print(out.to_string(index=False))
print("\n→ reports/tableau_resultats_v6.csv")

# Top features par cible
imp_rows = []
for tgt, feats in feature_imps.items():
    for rank, f in enumerate(feats, 1):
        imp_rows.append({"target": tgt, "rank": rank, "feature": f})
pd.DataFrame(imp_rows).to_csv("reports/top_features_per_target_v6.csv", index=False)
print("→ reports/top_features_per_target_v6.csv")
