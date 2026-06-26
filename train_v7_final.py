"""
train_v7_final.py — Pipeline FINAL :
  1. Entraînement direct sur v7 (avec Berkeley Earth)
  2. Stacking : OOF des modèles forts en features pour les faibles
  3. Anti-leak strict (EXTRA_LEAKS hérités de V6 + nouveau Berkeley)
  4. Feature selection top-100 par cible

Sortie : reports/tableau_resultats_v7_final.csv
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
DATA = "data/cleaned/dataset_final_v7.csv"
os.makedirs("models_v7", exist_ok=True)
os.makedirs("reports", exist_ok=True)
TOP_K = 100

df = pd.read_csv(DATA)
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
print(f"Dataset v7: {df.shape}\n")

# Création cibles dérivées
for c in ["disaster_deaths", "disaster_affected"]:
    if c in df.columns and f"target_{c}" not in df.columns:
        df[f"target_{c}"] = np.log1p(df[c].clip(lower=0))
if "stunting_pct" in df.columns and "target_stunting" not in df.columns:
    df["target_stunting"] = df["stunting_pct"]
if "pm25_annual" in df.columns and "target_pm25" not in df.columns:
    df["target_pm25"] = df["pm25_annual"]
if "Fertility_Rate" in df.columns and "target_fertility" not in df.columns:
    df["target_fertility"] = df["Fertility_Rate"]

# ── Configuration ──────────────────────────────────────────────────────
TARGETS = {
    "target_yield_cereals":      "Rendement céréales",
    "target_yield_oilcrops":     "Rendement oléagineux (agrégé)",
    "target_yield_pulses":       "Rendement légumineuses (agrégé)",
    "target_yield_roots":        "Rendement racines/tubercules",
    "target_yield_fruits":       "Rendement fruits (agrégé)",
    "target_yield_vegetables":   "Rendement légumes (agrégé)",
    "target_water_stress":       "Stress hydrique",
    "target_soil_degradation":   "Dégradation du sol",
    "target_thermal_anomaly":    "Anomalie thermique",
    "target_soil_moisture_root": "Humidité sol racinaire",
    "target_forest_share":       "% forêt national",
    "target_tree_cover_loss":    "Perte couvert arboré",
    "target_child_mortality":    "Mortalité infantile",
    "target_life_expectancy":    "Espérance de vie",
    "target_pop_growth":         "Croissance démographique",
    "target_birth_rate":         "Taux de natalité",
    "target_death_rate":         "Taux de mortalité brut",
    "target_net_migration":      "Migration nette",
    "target_fertility":          "Taux de fécondité",
    "target_stunting":           "Retard de croissance enfants",
    "target_pm25":               "PM2.5",
    "target_disaster_deaths":    "Décès catastrophes",
    "target_disaster_affected":  "Affectés catastrophes",
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

EXTRA_LEAKS = {
    "target_yield_cereals":   ["cereal_yield", "cereal_production_t", "food_production_index", "owid_cereal_production"],
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
                               "tree_cover_loss_cumul5y", "forest_per_capita_km2"],
    "target_pm25":            ["who_pm25_mean", "who_ambient_air_deaths_rate"],
    "target_thermal_anomaly": ["owid_temp_anomaly", "be_t_anom_annual",
                               "be_t_anom_vs_preindustrial", "be_t_baseline_1850_1900"],
    # Disasters : ne pas leak entre elles
    "target_disaster_deaths": ["disaster_affected", "disaster_damages_usd", "disaster_events",
                               "disaster_deaths_drought","disaster_deaths_flood","disaster_deaths_storm",
                               "disaster_affected_per_capita","disaster_deaths_per_million",
                               "disaster_events_cumul5y"],
    "target_disaster_affected": ["disaster_deaths", "disaster_damages_usd", "disaster_events",
                                 "disaster_affected_drought","disaster_affected_flood","disaster_affected_storm",
                                 "disaster_affected_per_capita","disaster_deaths_per_million",
                                 "disaster_events_cumul5y"],
}

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

SOCIO_TARGETS = {"target_child_mortality","target_life_expectancy","target_pop_growth",
                 "target_stunting","target_fertility","target_pm25",
                 "target_birth_rate","target_death_rate","target_net_migration"}

DISASTER_VARS = ["disaster_deaths","disaster_affected","disaster_damages_usd","disaster_events"]
DISASTER_TARGETS = {"target_disaster_deaths","target_disaster_affected"}

ID_COLS = ["ISO","Annee","T_ref","P_ref"]
TARGET_COLS = [c for c in df.columns if c.startswith("target_")]
YIELD_COLS = [c for c in df.columns if c.startswith("yield_") and "kgha" in c]


def get_blacklist(target, exclude_oof=None):
    bl = set(ID_COLS) | set(TARGET_COLS)
    src = TARGET_SOURCE.get(target)
    if src:
        bl |= {c for c in df.columns if c == src or c.startswith(src + "_") or c == src + "_ref"}
    for extra in EXTRA_LEAKS.get(target, []):
        bl |= {c for c in df.columns if c == extra or c.startswith(extra + "_") or c == extra + "_ref"}
    if target.startswith("target_yield_"):
        bl |= set(YIELD_COLS) | {"cereals_prod_t", "cereals_area_ha",
                                  "cereal_production_t", "food_production_index", "cereal_yield"}
        for y in YIELD_COLS + ["cereal_production_t", "cereal_yield"]:
            bl |= {c for c in df.columns if c.startswith(y + "_")}
    if target in SOCIO_TARGETS:
        for v in SOCIO_VARS:
            bl |= {c for c in df.columns if c == v or c.startswith(v + "_")}
    if target in DISASTER_TARGETS:
        for v in DISASTER_VARS:
            bl |= {c for c in df.columns if c == v or c.startswith(v + "_")}
    if exclude_oof:
        for o in exclude_oof:
            bl.add(o)
    return bl


def build_X(d, target, exclude_oof=None):
    bl = get_blacklist(target, exclude_oof=exclude_oof)
    feats = [c for c in d.columns if c not in bl and d[c].dtype != object]
    return d[feats].copy(), feats


def make_pre():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])


def select_top(X, y, k=TOP_K):
    X = X.dropna(axis=1, how="all")
    if X.shape[1] <= k: return list(X.columns)
    Xp = make_pre().fit_transform(X)
    rf = RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=5,
                                n_jobs=-1, random_state=42)
    rf.fit(Xp, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(k).index.tolist()


def fit_best(Xtr, ytr, Xte, yte):
    best = (-np.inf, None, None, None)
    grid = [
        ("Ridge", Ridge(alpha=1.0)),
        ("RandomForest", RandomForestRegressor(n_estimators=300, max_depth=15,
                                                min_samples_leaf=4, n_jobs=-1, random_state=42)),
        ("XGB-400-5", XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                    random_state=42, verbosity=0)),
        ("XGB-600-6", XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                    random_state=42, verbosity=0)),
        ("XGB-800-4", XGBRegressor(n_estimators=800, max_depth=4, learning_rate=0.03,
                                    subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                    random_state=42, verbosity=0)),
    ]
    for name, m in grid:
        try:
            pipe = Pipeline([("pre", make_pre()), ("model", m)])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            r2 = r2_score(yte, pred)
            if r2 > best[0]:
                best = (r2, name, pipe, mean_absolute_error(yte, pred))
        except Exception:
            continue
    return best


# ── PASS 1 : entraînement direct ──────────────────────────────────────
print("════════════════════════════════════════════════════════════════")
print("PASS 1 — Entraînement direct (sans stacking)")
print("════════════════════════════════════════════════════════════════")

splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
results_pass1 = []
oof_predictions = {}  # OOF des cibles fortes pour stacking pass 2

# Cibles fortes pour OOF
STRONG_TARGETS = [
    "target_forest_share", "target_thermal_anomaly", "target_child_mortality",
    "target_fertility", "target_birth_rate", "target_life_expectancy",
    "target_soil_degradation", "target_yield_roots", "target_yield_cereals",
    "target_water_stress", "target_soil_moisture_root",
]
gkf = GroupKFold(n_splits=5)

for tgt, label in TARGETS.items():
    if tgt not in df.columns: continue
    d = df.dropna(subset=[tgt]).copy()
    if len(d) < 200: continue

    tr_idx, te_idx = next(splitter.split(d, groups=d["ISO"]))
    tr, te = d.iloc[tr_idx], d.iloc[te_idx]
    Xtr_full, feats_all = build_X(tr, tgt)
    Xte_full, _ = build_X(te, tgt)
    ytr, yte = tr[tgt], te[tgt]

    selected = select_top(Xtr_full, ytr, k=TOP_K)
    Xtr, Xte = Xtr_full[selected], Xte_full[selected]
    print(f"\n🎯 {label} ({tgt})")
    print(f"   n={len(d)} | feats_all={len(feats_all)} → top {len(selected)}")
    r2, bname, bpipe, bmae = fit_best(Xtr, ytr, Xte, yte)
    if bpipe is None: continue
    joblib.dump({"pipe": bpipe, "features": selected}, f"models_v7/best_{tgt}.joblib")
    print(f"   🏆 {bname}  R²={r2:+.4f}  MAE={bmae:.3f}")
    results_pass1.append({"Cible": label, "Technique": tgt, "Modèle": bname,
                          "R² (sans stacking)": round(r2, 4),
                          "MAE": round(bmae, 3), "N test": len(yte),
                          "N feats": len(selected)})

    # Si cible forte → générer OOF pour stacking
    if tgt in STRONG_TARGETS:
        oof = pd.Series(index=d.index, dtype=float)
        Xall = d[selected]
        yall = d[tgt]
        for tr_o, te_o in gkf.split(Xall, yall, groups=d["ISO"]):
            pipe2 = Pipeline([("pre", make_pre()),
                              ("model", XGBRegressor(n_estimators=400, max_depth=6,
                                                      learning_rate=0.05, n_jobs=-1,
                                                      random_state=42, verbosity=0))])
            pipe2.fit(Xall.iloc[tr_o], yall.iloc[tr_o])
            oof.iloc[te_o] = pipe2.predict(Xall.iloc[te_o])
        oof_predictions[tgt] = oof

# Injection des OOF dans df
print(f"\n→ {len(oof_predictions)} OOF strong targets générés")
for tgt, oof in oof_predictions.items():
    col = f"oof_{tgt}"
    df[col] = np.nan
    df.loc[oof.index, col] = oof.values
oof_cols = list(f"oof_{t}" for t in oof_predictions)


# ── PASS 2 : STACKING sur les cibles faibles ──────────────────────────
print("\n════════════════════════════════════════════════════════════════")
print("PASS 2 — STACKING (OOF des forts → features pour les faibles)")
print("════════════════════════════════════════════════════════════════")

WEAK_TARGETS = ["target_pop_growth", "target_net_migration", "target_death_rate",
                "target_yield_pulses", "target_yield_fruits", "target_yield_oilcrops",
                "target_yield_vegetables", "target_tree_cover_loss",
                "target_disaster_deaths", "target_disaster_affected",
                "target_stunting", "target_pm25"]

results_pass2 = []
for tgt in WEAK_TARGETS:
    if tgt not in df.columns: continue
    label = TARGETS.get(tgt, tgt)
    # OOF à exclure : oof_tgt si même cible. Pour cibles socio, exclure OOF socio.
    exclude_oof = set()
    if tgt in SOCIO_TARGETS:
        for o in oof_cols:
            t2 = o.replace("oof_", "")
            if t2 in SOCIO_TARGETS:
                exclude_oof.add(o)
    # Pour disasters : exclure OOF disasters (s'il y en avait)
    if tgt in DISASTER_TARGETS:
        for o in oof_cols:
            t2 = o.replace("oof_", "")
            if t2 in DISASTER_TARGETS:
                exclude_oof.add(o)
    # Pour yields : ne pas exclure les yields strong → ça aide

    d = df.dropna(subset=[tgt]).copy()
    if len(d) < 200: continue
    tr_idx, te_idx = next(splitter.split(d, groups=d["ISO"]))
    tr, te = d.iloc[tr_idx], d.iloc[te_idx]
    Xtr_full, feats_all = build_X(tr, tgt, exclude_oof=exclude_oof)
    Xte_full, _ = build_X(te, tgt, exclude_oof=exclude_oof)
    ytr, yte = tr[tgt], te[tgt]
    selected = select_top(Xtr_full, ytr, k=TOP_K)
    Xtr, Xte = Xtr_full[selected], Xte_full[selected]
    n_oof_used = sum(1 for f in selected if f.startswith("oof_"))
    print(f"\n🎯 {label} ({tgt})")
    print(f"   feats_all={len(feats_all)} → top {len(selected)} (OOF kept: {n_oof_used})")
    r2, bname, bpipe, bmae = fit_best(Xtr, ytr, Xte, yte)
    if bpipe is None: continue
    joblib.dump({"pipe": bpipe, "features": selected}, f"models_v7/stacked_{tgt}.joblib")
    print(f"   🏆 {bname}  R²={r2:+.4f}  MAE={bmae:.3f}")
    # Comparaison vs pass1
    prev = next((r for r in results_pass1 if r["Technique"] == tgt), None)
    if prev:
        delta = r2 - prev["R² (sans stacking)"]
        sign = "+" if delta > 0 else ""
        print(f"   Δ vs pass1 = {sign}{delta:.3f}")
    results_pass2.append({"Cible": label, "Technique": tgt, "Modèle": bname,
                          "R² (stacking)": round(r2, 4), "MAE": round(bmae, 3),
                          "N OOF utilisées": n_oof_used,
                          "R² pass1": prev["R² (sans stacking)"] if prev else None,
                          "Δ stacking": round(r2 - prev["R² (sans stacking)"], 4) if prev else None})


# ── Sauvegarde rapports ──────────────────────────────────────────────
out1 = pd.DataFrame(results_pass1)
out2 = pd.DataFrame(results_pass2)
out1.to_csv("reports/v7_pass1_direct.csv", index=False)
out2.to_csv("reports/v7_pass2_stacking.csv", index=False)

# Tableau combiné final : prendre le meilleur des 2 passes
final = []
for r in results_pass1:
    tgt = r["Technique"]
    pass2_r = next((x for x in results_pass2 if x["Technique"] == tgt), None)
    if pass2_r and pass2_r["R² (stacking)"] > r["R² (sans stacking)"]:
        final.append({"Cible": r["Cible"], "Technique": tgt, "Modèle": pass2_r["Modèle"],
                      "R² FINAL": pass2_r["R² (stacking)"], "MAE": pass2_r["MAE"],
                      "Méthode": "STACKING"})
    else:
        final.append({"Cible": r["Cible"], "Technique": tgt, "Modèle": r["Modèle"],
                      "R² FINAL": r["R² (sans stacking)"], "MAE": r["MAE"],
                      "Méthode": "DIRECT"})

dffinal = pd.DataFrame(final).sort_values("R² FINAL", ascending=False)
dffinal.to_csv("reports/tableau_resultats_v7_final.csv", index=False)
print("\n════════════════════════════════════════════════════════════════")
print("📊 RÉSULTATS FINAUX V7 :")
print("════════════════════════════════════════════════════════════════")
print(dffinal.to_string(index=False))
print("\n→ reports/tableau_resultats_v7_final.csv")
