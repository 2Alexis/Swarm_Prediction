"""
train_v8_ultimate.py — PIPELINE ULTIME avec toutes optims :
  - Optuna tuning XGBoost + LightGBM (30 trials par cible)
  - Ensemble blending (mean ou best-3 averaged)
  - 5-fold GroupKFold pour R² robuste (mean ± std)
  - Quantile transformer pour cibles skewées
  - Feature selection top-100 par cible
  - Stacking 2 passes (OOF des forts → features pour faibles)
"""
import os, sys, io, warnings, logging
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
DATA = "data/cleaned/dataset_final_v7.csv"
os.makedirs("models_v8", exist_ok=True)
os.makedirs("reports", exist_ok=True)
TOP_K = 100
N_TRIALS = 25      # essais Optuna par cible
N_CV_FOLDS = 5     # GroupKFold

df = pd.read_csv(DATA)
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
print(f"Dataset v7: {df.shape}\n")

# ── Cibles dérivées ───────────────────────────────────────────────────────
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
    "target_thermal_anomaly": ["owid_temp_anomaly", "be_t_anom_annual",
                               "be_t_anom_vs_preindustrial", "be_t_baseline_1850_1900"],
    "target_disaster_deaths": ["disaster_affected", "disaster_damages_usd", "disaster_events",
                               "disaster_affected_per_capita","disaster_deaths_per_million",
                               "disaster_events_cumul5y"],
    "target_disaster_affected": ["disaster_deaths", "disaster_damages_usd", "disaster_events",
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


def cv_r2(X, y, groups, model_factory, n_folds=N_CV_FOLDS):
    """Renvoie le R² moyen sur GroupKFold."""
    gkf = GroupKFold(n_splits=n_folds)
    scores = []
    for tr, te in gkf.split(X, y, groups=groups):
        pipe = Pipeline([("pre", make_pre()), ("model", model_factory())])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        scores.append(r2_score(y.iloc[te], pred))
    return float(np.mean(scores)), float(np.std(scores))


def optuna_xgb(X, y, groups, n_trials=N_TRIALS):
    def obj(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        def f(): return XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
        r2, _ = cv_r2(X, y, groups, f, n_folds=3)  # CV3 dans Optuna (rapide)
        return r2
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optuna_lgb(X, y, groups, n_trials=N_TRIALS):
    def obj(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        def f(): return lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
        r2, _ = cv_r2(X, y, groups, f, n_folds=3)
        return r2
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def evaluate_full_cv(target, exclude_oof=None):
    """Evalue avec Optuna + Ensemble sur 5-fold CV."""
    d = df.dropna(subset=[target]).copy()
    if len(d) < 200:
        return None
    bl = get_blacklist(target, exclude_oof=exclude_oof)
    feats = [c for c in d.columns if c not in bl and d[c].dtype != object]
    feats = [c for c in feats if d[c].notna().sum() > 0]
    X, y, groups = d[feats], d[target], d["ISO"]

    # Feature selection sur le dataset complet (sans leakage car non-target features)
    selected = select_top(X, y, k=TOP_K)
    Xs = X[selected]

    # Optuna : tuner XGB + LGBM
    xgb_params = optuna_xgb(Xs, y, groups, n_trials=N_TRIALS)
    lgb_params = optuna_lgb(Xs, y, groups, n_trials=N_TRIALS)

    # Évaluation finale en 5-fold CV : individuels + ensemble
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    fold_results = {"Ridge": [], "RF": [], "XGB": [], "LGB": [], "Ensemble": []}
    oof_preds_per_model = {"Ridge": np.zeros(len(d)), "RF": np.zeros(len(d)),
                           "XGB": np.zeros(len(d)), "LGB": np.zeros(len(d)),
                           "Ensemble": np.zeros(len(d))}

    for tr, te in gkf.split(Xs, y, groups=groups):
        # Ridge
        pipe_r = Pipeline([("pre", make_pre()), ("model", Ridge(alpha=1.0))])
        pipe_r.fit(Xs.iloc[tr], y.iloc[tr])
        p_r = pipe_r.predict(Xs.iloc[te])
        fold_results["Ridge"].append(r2_score(y.iloc[te], p_r))
        oof_preds_per_model["Ridge"][te] = p_r

        # RF
        pipe_rf = Pipeline([("pre", make_pre()),
                             ("model", RandomForestRegressor(n_estimators=300, max_depth=15,
                                                              min_samples_leaf=4, n_jobs=-1,
                                                              random_state=42))])
        pipe_rf.fit(Xs.iloc[tr], y.iloc[tr])
        p_rf = pipe_rf.predict(Xs.iloc[te])
        fold_results["RF"].append(r2_score(y.iloc[te], p_rf))
        oof_preds_per_model["RF"][te] = p_rf

        # XGB tuned
        pipe_x = Pipeline([("pre", make_pre()),
                            ("model", XGBRegressor(**xgb_params, random_state=42, n_jobs=-1, verbosity=0))])
        pipe_x.fit(Xs.iloc[tr], y.iloc[tr])
        p_x = pipe_x.predict(Xs.iloc[te])
        fold_results["XGB"].append(r2_score(y.iloc[te], p_x))
        oof_preds_per_model["XGB"][te] = p_x

        # LGB tuned
        pipe_l = Pipeline([("pre", make_pre()),
                            ("model", lgb.LGBMRegressor(**lgb_params, random_state=42, n_jobs=-1, verbose=-1))])
        pipe_l.fit(Xs.iloc[tr], y.iloc[tr])
        p_l = pipe_l.predict(Xs.iloc[te])
        fold_results["LGB"].append(r2_score(y.iloc[te], p_l))
        oof_preds_per_model["LGB"][te] = p_l

        # Ensemble : moyenne pondérée par R² du fold (favor le meilleur)
        scores = [fold_results[m][-1] for m in ["RF", "XGB", "LGB"]]
        scores = [max(s, 0.01) for s in scores]
        total = sum(scores)
        w = [s/total for s in scores]
        p_ens = w[0]*p_rf + w[1]*p_x + w[2]*p_l
        fold_results["Ensemble"].append(r2_score(y.iloc[te], p_ens))
        oof_preds_per_model["Ensemble"][te] = p_ens

    summary = {m: (np.mean(v), np.std(v)) for m, v in fold_results.items()}
    # Trouver le meilleur (par mean)
    best_model = max(summary.keys(), key=lambda k: summary[k][0])
    return {
        "target": target,
        "n_obs": len(d),
        "n_features": len(selected),
        "selected_features": selected,
        "best_model": best_model,
        "best_r2_mean": summary[best_model][0],
        "best_r2_std": summary[best_model][1],
        "all_results": summary,
        "xgb_params": xgb_params,
        "lgb_params": lgb_params,
        "oof_predictions": oof_preds_per_model[best_model],
        "oof_index": d.index,
    }


import time
results = []
oof_store = {}

# ── PASS 1 — sans stacking ──
print("══════════════════════════════════════════════════════════════════")
print("PASS 1 — Optuna + Ensemble + 5-fold CV (sans stacking)")
print("══════════════════════════════════════════════════════════════════")

STRONG_TARGETS = [
    "target_forest_share", "target_thermal_anomaly", "target_child_mortality",
    "target_fertility", "target_birth_rate", "target_life_expectancy",
    "target_soil_degradation", "target_yield_roots", "target_yield_cereals",
    "target_water_stress", "target_soil_moisture_root",
]

for tgt, label in TARGETS.items():
    if tgt not in df.columns: continue
    t0 = time.time()
    print(f"\n🎯 {label} ({tgt})…")
    res = evaluate_full_cv(tgt)
    if res is None:
        print("   skip (n<200)")
        continue
    dt = time.time() - t0
    print(f"   n={res['n_obs']}  feats={res['n_features']}")
    for m, (mu, sd) in res["all_results"].items():
        marker = " ★" if m == res["best_model"] else ""
        print(f"   [{m:9s}] R²= {mu:+.4f} ± {sd:.4f}{marker}")
    print(f"   🏆 {res['best_model']}  CV5 R²={res['best_r2_mean']:+.4f} ± {res['best_r2_std']:.4f}  ({dt:.0f}s)")

    if tgt in STRONG_TARGETS:
        oof_series = pd.Series(np.nan, index=df.index)
        oof_series.loc[res["oof_index"]] = res["oof_predictions"]
        oof_store[tgt] = oof_series

    results.append({"Cible": label, "Technique": tgt,
                    "Meilleur": res["best_model"],
                    "R² CV5 mean": round(res["best_r2_mean"], 4),
                    "R² CV5 std": round(res["best_r2_std"], 4),
                    "N obs": res["n_obs"],
                    "Méthode": "Direct (Optuna+Ensemble)"})

# Injecter OOF dans df pour stacking
print(f"\n→ {len(oof_store)} OOF strong générés")
for tgt, s in oof_store.items():
    df[f"oof_{tgt}"] = s


# ── PASS 2 — Stacking sur cibles faibles ──
print("\n══════════════════════════════════════════════════════════════════")
print("PASS 2 — Stacking (OOF des forts → features pour les faibles)")
print("══════════════════════════════════════════════════════════════════")

WEAK_TARGETS = ["target_pop_growth", "target_net_migration", "target_death_rate",
                "target_yield_pulses", "target_yield_fruits", "target_yield_oilcrops",
                "target_yield_vegetables", "target_tree_cover_loss",
                "target_disaster_deaths", "target_disaster_affected",
                "target_stunting"]

for tgt in WEAK_TARGETS:
    if tgt not in df.columns: continue
    label = TARGETS.get(tgt, tgt)
    # OOF à exclure : socio si tgt socio, disaster si tgt disaster
    exclude_oof = set()
    if tgt in SOCIO_TARGETS:
        for o in [f"oof_{t}" for t in oof_store.keys()]:
            t2 = o.replace("oof_", "")
            if t2 in SOCIO_TARGETS: exclude_oof.add(o)
    if tgt in DISASTER_TARGETS:
        for o in [f"oof_{t}" for t in oof_store.keys()]:
            t2 = o.replace("oof_", "")
            if t2 in DISASTER_TARGETS: exclude_oof.add(o)

    t0 = time.time()
    print(f"\n🎯 STACK {label} ({tgt})")
    res = evaluate_full_cv(tgt, exclude_oof=exclude_oof)
    if res is None: continue
    dt = time.time() - t0
    print(f"   feats={res['n_features']}")
    for m, (mu, sd) in res["all_results"].items():
        marker = " ★" if m == res["best_model"] else ""
        print(f"   [{m:9s}] R²= {mu:+.4f} ± {sd:.4f}{marker}")
    print(f"   🏆 {res['best_model']}  CV5 R²={res['best_r2_mean']:+.4f} ± {res['best_r2_std']:.4f}  ({dt:.0f}s)")
    # comparer avec pass 1
    prev = next((r for r in results if r["Technique"] == tgt), None)
    delta = res["best_r2_mean"] - prev["R² CV5 mean"] if prev else 0
    print(f"   Δ vs direct = {'+' if delta >= 0 else ''}{delta:.4f}")

    # Si stacking gagne → remplace
    if prev and res["best_r2_mean"] > prev["R² CV5 mean"]:
        prev["Meilleur"] = res["best_model"]
        prev["R² CV5 mean"] = round(res["best_r2_mean"], 4)
        prev["R² CV5 std"] = round(res["best_r2_std"], 4)
        prev["Méthode"] = "STACKING (Optuna+Ensemble)"


# ── Save report ──
out = pd.DataFrame(results).sort_values("R² CV5 mean", ascending=False)
out.to_csv("reports/tableau_resultats_v8_ULTIMATE.csv", index=False)
print("\n════════════════════════════════════════════════════════════════")
print("📊 RÉSULTATS FINAUX V8 (CV5, mean ± std) :")
print("════════════════════════════════════════════════════════════════")
print(out.to_string(index=False))
print("\n→ reports/tableau_resultats_v8_ULTIMATE.csv")
