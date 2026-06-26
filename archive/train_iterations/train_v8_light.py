"""
train_v8_light.py — Version LÉGÈRE de v8 :
  - GroupShuffleSplit unique (comme v7)
  - RF + XGB + LGB sans tuning (params raisonnables)
  - Ensemble : moyenne pondérée par R² test
  - Stacking direct sur cibles faibles
  - Pas d'Optuna, pas de CV5 → ~10 min total
"""
import os, sys, io, warnings, time
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
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA = "data/cleaned/dataset_final_v7.csv"
os.makedirs("models_v8", exist_ok=True)
os.makedirs("reports", exist_ok=True)
TOP_K = 80

df = pd.read_csv(DATA)
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
print(f"Dataset: {df.shape}\n", flush=True)

for c in ["disaster_deaths", "disaster_affected"]:
    if c in df.columns and f"target_{c}" not in df.columns:
        df[f"target_{c}"] = np.log1p(df[c].clip(lower=0))
if "stunting_pct" in df.columns and "target_stunting" not in df.columns:
    df["target_stunting"] = df["stunting_pct"]
if "Fertility_Rate" in df.columns and "target_fertility" not in df.columns:
    df["target_fertility"] = df["Fertility_Rate"]


TARGETS = {
    "target_yield_cereals":      "Rendement céréales",
    "target_yield_oilcrops":     "Rendement oléagineux",
    "target_yield_pulses":       "Rendement légumineuses",
    "target_yield_roots":        "Rendement racines",
    "target_yield_fruits":       "Rendement fruits",
    "target_yield_vegetables":   "Rendement légumes",
    "target_water_stress":       "Stress hydrique",
    "target_soil_degradation":   "Dégradation sol",
    "target_thermal_anomaly":    "Anomalie thermique",
    "target_soil_moisture_root": "Humidité sol racinaire",
    "target_forest_share":       "% forêt",
    "target_tree_cover_loss":    "Perte couvert arboré",
    "target_child_mortality":    "Mortalité infantile",
    "target_life_expectancy":    "Espérance de vie",
    "target_pop_growth":         "Croissance démo",
    "target_birth_rate":         "Natalité",
    "target_death_rate":         "Mortalité brut",
    "target_net_migration":      "Migration nette",
    "target_fertility":          "Fécondité",
    "target_stunting":           "Stunting",
    "target_disaster_deaths":    "Décès catastrophes",
    "target_disaster_affected":  "Affectés catastrophes",
}

TARGET_SOURCE = {
    "target_yield_cereals":"yield_cereals_kgha","target_yield_oilcrops":"yield_oilcrops_kgha",
    "target_yield_pulses":"yield_pulses_kgha","target_yield_roots":"yield_roots_kgha",
    "target_yield_fruits":"yield_fruits_kgha","target_yield_vegetables":"yield_vegetables_kgha",
    "target_water_stress":"Water_Withdrawal_pct","target_soil_degradation":"Bilan_sols_kgha",
    "target_thermal_anomaly":"T_anomaly","target_child_mortality":"Child_Mort",
    "target_life_expectancy":"Life_Exp","target_pop_growth":"Pop_Growth",
    "target_birth_rate":"Birth_Rate","target_death_rate":"Death_Rate",
    "target_net_migration":"Net_Migration","target_disaster_deaths":"disaster_deaths",
    "target_disaster_affected":"disaster_affected","target_stunting":"stunting_pct",
    "target_fertility":"Fertility_Rate","target_soil_moisture_root":"nasa_gwetroot",
    "target_forest_share":"forest_share_pct","target_tree_cover_loss":"tree_cover_loss_ha",
}

EXTRA_LEAKS = {
    "target_yield_cereals":["cereal_yield","cereal_production_t","food_production_index","owid_cereal_production"],
    "target_yield_oilcrops":["food_production_index"],"target_yield_pulses":["food_production_index"],
    "target_yield_roots":["food_production_index"],"target_yield_fruits":["food_production_index"],
    "target_yield_vegetables":["food_production_index"],
    "target_water_stress":["freshwater_withdraw_total","freshwater_internal_per_cap","water_stress_ratio_raw","owid_water_stress"],
    "target_stunting":["wasting_pct","overweight_pct","malnutrition_compound","infant_deaths_total","Hunger_Index","ghi_owid"],
    "target_fertility":["Birth_Rate","Death_Rate","Pop_Growth","owid_crude_birth_rate","owid_crude_death_rate"],
    "target_birth_rate":["Death_Rate","Net_Migration","Pop_Growth","Fertility_Rate","owid_crude_birth_rate","owid_crude_death_rate","owid_births_total","owid_deaths_total","births_per_1000_minus_deaths"],
    "target_death_rate":["Birth_Rate","Net_Migration","Pop_Growth","Fertility_Rate","owid_crude_birth_rate","owid_crude_death_rate","owid_births_total","owid_deaths_total","births_per_1000_minus_deaths","adult_mortality_male","adult_mortality_female","infant_deaths_total"],
    "target_net_migration":["Birth_Rate","Death_Rate","Pop_Growth","Fertility_Rate","owid_crude_birth_rate","owid_crude_death_rate","owid_births_total","owid_deaths_total","refugees_origin","refugees_destination","refugees_origin_owid","idps_conflict"],
    "target_soil_moisture_root":["nasa_gwettop","nasa_gwetprof","soil_moisture_deficit","combined_drought_index","heat_drought_stress","soil_moisture_top_root_ratio"],
    "target_forest_share":["forest_area_km2","forest_change","tree_cover_loss_ha","tree_cover_loss_cumul5y","deforestation_annual","deforestation_pct_annual","forest_per_capita_km2"],
    "target_tree_cover_loss":["forest_change","forest_share_pct","forest_area_km2","deforestation_annual","deforestation_pct_annual","tree_cover_loss_cumul5y","forest_per_capita_km2"],
    "target_thermal_anomaly":["owid_temp_anomaly","be_t_anom_annual","be_t_anom_vs_preindustrial","be_t_baseline_1850_1900"],
    "target_disaster_deaths":["disaster_affected","disaster_damages_usd","disaster_events","disaster_affected_per_capita","disaster_deaths_per_million","disaster_events_cumul5y"],
    "target_disaster_affected":["disaster_deaths","disaster_damages_usd","disaster_events","disaster_affected_per_capita","disaster_deaths_per_million","disaster_events_cumul5y"],
}

SOCIO_VARS = ["Child_Mort","Life_Exp","Pop_Growth","Birth_Rate","Death_Rate","Net_Migration","HDI","GDP_pc","GDP_total_usd","Population","Urban_pct","Inflation_CPI","Unemployment","Gini","Poverty_190","Poverty_OWID","Debt_GDP","Trade_GDP","Hunger_Index","Internet_pct","Mobile_subs","Electricity_pct","Energy_pc","Health_GDP","Hospital_Beds","RD_GDP","Malaria","HIV","Deaths_Communicable","Renew_Energy_pct","Energy_total","Life_Exp_OWID","Fertility_Rate","Internet_OWID","elec_generation_gwh","elec_renew_share","stunting_pct","wasting_pct","overweight_pct","safe_water_pct","sanitation_pct","physicians_per_1000","adult_mortality_male","adult_mortality_female","infant_deaths_total","school_primary_enrollment","school_secondary_enrollment","adult_literacy_pct","employ_agri_pct","employ_industry_pct","employ_services_pct","energy_use_per_cap","broadband_per_100","dependency_ratio","dependency_young","dependency_old","pop_65_plus_pct","pop_under14_pct","urban_growth","schooling_years","meat_consumption_pc","extreme_poverty_pct","pop_density_per_km2","gdp_per_arable_ha","urban_pop_abs","elec_gen_per_capita","disaster_deaths_per_million","disaster_affected_per_capita","malnutrition_compound","school_total_enrollment","agri_value_pct_gdp","manuf_value_pct_gdp","services_value_pct_gdp","pm25_annual","co_emissions","cereal_production_t","co2_per_capita_calc","owid_births_total","owid_deaths_total","owid_crude_birth_rate","owid_crude_death_rate","owid_pop_density","owid_agri_land_share","births_per_1000_minus_deaths","owid_co2_pc","owid_methane","owid_n2o","owid_ghg_by_sector","ghi_owid","agri_output_usd"]
SOCIO_TARGETS = {"target_child_mortality","target_life_expectancy","target_pop_growth","target_stunting","target_fertility","target_birth_rate","target_death_rate","target_net_migration"}
DISASTER_VARS = ["disaster_deaths","disaster_affected","disaster_damages_usd","disaster_events"]
DISASTER_TARGETS = {"target_disaster_deaths","target_disaster_affected"}
ID_COLS = ["ISO","Annee","T_ref","P_ref"]
TARGET_COLS = [c for c in df.columns if c.startswith("target_")]
YIELD_COLS = [c for c in df.columns if c.startswith("yield_") and "kgha" in c]


def get_bl(target, exclude_oof=None):
    bl = set(ID_COLS) | set(TARGET_COLS)
    src = TARGET_SOURCE.get(target)
    if src:
        bl |= {c for c in df.columns if c == src or c.startswith(src + "_") or c == src + "_ref"}
    for extra in EXTRA_LEAKS.get(target, []):
        bl |= {c for c in df.columns if c == extra or c.startswith(extra + "_") or c == extra + "_ref"}
    if target.startswith("target_yield_"):
        bl |= set(YIELD_COLS) | {"cereals_prod_t","cereals_area_ha","cereal_production_t","food_production_index","cereal_yield"}
        for y in YIELD_COLS + ["cereal_production_t","cereal_yield"]:
            bl |= {c for c in df.columns if c.startswith(y + "_")}
    if target in SOCIO_TARGETS:
        for v in SOCIO_VARS:
            bl |= {c for c in df.columns if c == v or c.startswith(v + "_")}
    if target in DISASTER_TARGETS:
        for v in DISASTER_VARS:
            bl |= {c for c in df.columns if c == v or c.startswith(v + "_")}
    if exclude_oof:
        for o in exclude_oof: bl.add(o)
    return bl


def make_pre():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])


def select_top(X, y, k=TOP_K):
    X = X.dropna(axis=1, how="all")
    if X.shape[1] <= k: return list(X.columns)
    Xp = make_pre().fit_transform(X)
    rf = RandomForestRegressor(n_estimators=80, max_depth=10, min_samples_leaf=5,
                                n_jobs=-1, random_state=42)
    rf.fit(Xp, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(k).index.tolist()


def train_eval(d, target, exclude_oof=None):
    bl = get_bl(target, exclude_oof=exclude_oof)
    feats = [c for c in d.columns if c not in bl and d[c].dtype != object]
    feats = [c for c in feats if d[c].notna().sum() > 0]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(splitter.split(d, groups=d["ISO"]))
    tr, te = d.iloc[tr_idx], d.iloc[te_idx]

    Xtr_full, Xte_full = tr[feats], te[feats]
    ytr, yte = tr[target], te[target]
    sel = select_top(Xtr_full, ytr, k=TOP_K)
    Xtr, Xte = Xtr_full[sel], Xte_full[sel]

    # Trois modèles + ensemble
    results = {}
    preds = {}

    pipe_rf = Pipeline([("pre", make_pre()),
                         ("model", RandomForestRegressor(n_estimators=250, max_depth=15,
                                                          min_samples_leaf=4, n_jobs=-1,
                                                          random_state=42))])
    pipe_rf.fit(Xtr, ytr)
    p = pipe_rf.predict(Xte)
    results["RF"] = (r2_score(yte, p), mean_absolute_error(yte, p))
    preds["RF"] = p

    pipe_xgb = Pipeline([("pre", make_pre()),
                          ("model", XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                                  subsample=0.85, colsample_bytree=0.85,
                                                  random_state=42, n_jobs=-1, verbosity=0))])
    pipe_xgb.fit(Xtr, ytr)
    p = pipe_xgb.predict(Xte)
    results["XGB"] = (r2_score(yte, p), mean_absolute_error(yte, p))
    preds["XGB"] = p

    pipe_lgb = Pipeline([("pre", make_pre()),
                          ("model", lgb.LGBMRegressor(n_estimators=500, max_depth=6, num_leaves=50,
                                                       learning_rate=0.05, subsample=0.85,
                                                       colsample_bytree=0.85, min_child_samples=10,
                                                       random_state=42, n_jobs=-1, verbose=-1))])
    pipe_lgb.fit(Xtr, ytr)
    p = pipe_lgb.predict(Xte)
    results["LGB"] = (r2_score(yte, p), mean_absolute_error(yte, p))
    preds["LGB"] = p

    # Ensemble : poids par max(0, R²)
    weights = np.array([max(results[m][0], 0.01) for m in ["RF", "XGB", "LGB"]])
    weights /= weights.sum()
    ens_p = weights[0]*preds["RF"] + weights[1]*preds["XGB"] + weights[2]*preds["LGB"]
    results["Ensemble"] = (r2_score(yte, ens_p), mean_absolute_error(yte, ens_p))

    best = max(results.items(), key=lambda kv: kv[1][0])
    # Pour stocker OOF — réentraîner sur tout en GroupKFold 3
    return best, results, sel


def oof_predict(d, target, sel, model_factory):
    """3-fold GroupKFold OOF pour stacking."""
    gkf = GroupKFold(n_splits=3)
    oof = np.zeros(len(d))
    X, y = d[sel], d[target]
    for tr, te in gkf.split(X, y, groups=d["ISO"]):
        pipe = Pipeline([("pre", make_pre()), ("model", model_factory())])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = pipe.predict(X.iloc[te])
    return oof


STRONG_R2 = ["target_forest_share", "target_thermal_anomaly", "target_fertility",
             "target_child_mortality", "target_birth_rate", "target_yield_roots",
             "target_life_expectancy", "target_water_stress", "target_soil_degradation",
             "target_stunting", "target_yield_oilcrops", "target_yield_cereals"]

results = []
oof_store = {}

print("══ PASS 1 — RF + XGB + LGB + Ensemble ══", flush=True)
for tgt, label in TARGETS.items():
    if tgt not in df.columns: continue
    t0 = time.time()
    d = df.dropna(subset=[tgt]).copy()
    if len(d) < 200: continue

    (bname, (br2, bmae)), all_res, sel = train_eval(d, tgt)
    dt = time.time() - t0
    print(f"\n🎯 {label}  ({tgt})  n={len(d)}  feats_top={len(sel)}  [{dt:.0f}s]", flush=True)
    for name, (r2, mae) in all_res.items():
        mark = " ★" if name == bname else ""
        print(f"   [{name:8s}] R²={r2:+.4f}  MAE={mae:.3f}{mark}", flush=True)
    print(f"   🏆 {bname}  R²={br2:+.4f}", flush=True)

    if tgt in STRONG_R2 and br2 > 0.5:
        # OOF via XGB pour stacking
        try:
            oof = oof_predict(d, tgt, sel,
                              lambda: XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                                    random_state=42, n_jobs=-1, verbosity=0))
            oof_series = pd.Series(np.nan, index=df.index)
            oof_series.loc[d.index] = oof
            oof_store[tgt] = oof_series
        except Exception as e:
            print(f"   OOF err: {e}", flush=True)

    results.append({"Cible": label, "Technique": tgt, "Meilleur": bname,
                    "R² test": round(br2, 4), "MAE": round(bmae, 3),
                    "R² RF": round(all_res["RF"][0], 4),
                    "R² XGB": round(all_res["XGB"][0], 4),
                    "R² LGB": round(all_res["LGB"][0], 4),
                    "R² Ens": round(all_res["Ensemble"][0], 4),
                    "N obs": len(d), "Méthode": "Direct"})

# Inject OOF
for tgt, s in oof_store.items():
    df[f"oof_{tgt}"] = s
print(f"\n→ {len(oof_store)} OOF strong générés", flush=True)


print("\n══ PASS 2 — Stacking sur cibles < 0.5 ══", flush=True)
WEAK = [r["Technique"] for r in results if r["R² test"] < 0.5]
print(f"Cibles: {WEAK}\n", flush=True)

for tgt in WEAK:
    if tgt not in df.columns: continue
    label = TARGETS[tgt]
    exclude_oof = set()
    if tgt in SOCIO_TARGETS:
        for t2 in oof_store:
            if t2 in SOCIO_TARGETS: exclude_oof.add(f"oof_{t2}")
    if tgt in DISASTER_TARGETS:
        for t2 in oof_store:
            if t2 in DISASTER_TARGETS: exclude_oof.add(f"oof_{t2}")

    d = df.dropna(subset=[tgt]).copy()
    if len(d) < 200: continue
    t0 = time.time()
    (bname, (br2, bmae)), all_res, sel = train_eval(d, tgt, exclude_oof=exclude_oof)
    n_oof = sum(1 for f in sel if f.startswith("oof_"))
    dt = time.time() - t0
    print(f"\n🎯 STACK {label}  (OOF utilisées: {n_oof})  [{dt:.0f}s]", flush=True)
    for name, (r2, mae) in all_res.items():
        mark = " ★" if name == bname else ""
        print(f"   [{name:8s}] R²={r2:+.4f}{mark}", flush=True)

    prev = next((r for r in results if r["Technique"] == tgt), None)
    if prev and br2 > prev["R² test"]:
        delta = br2 - prev["R² test"]
        print(f"   ⬆️  gain stacking +{delta:.3f}", flush=True)
        prev["Meilleur"] = bname
        prev["R² test"] = round(br2, 4)
        prev["MAE"] = round(bmae, 3)
        prev["Méthode"] = "STACKING"
    else:
        print(f"   ➡️  pas de gain", flush=True)


out = pd.DataFrame(results).sort_values("R² test", ascending=False)
out.to_csv("reports/tableau_resultats_v8_LIGHT.csv", index=False)
print("\n══════════════════════════════════════════════════════════════════")
print("📊 V8 LIGHT — RÉSULTATS FINAUX")
print("══════════════════════════════════════════════════════════════════")
print(out.to_string(index=False))
print("\n→ reports/tableau_resultats_v8_LIGHT.csv")
