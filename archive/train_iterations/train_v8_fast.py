"""
train_v8_fast.py — Version ACCÉLÉRÉE de l'ultimate :

Stratégie :
  - Cibles FORTES (déjà R²≥0.6 en v7) : ensemble RF+XGB+LGB sans Optuna (grid raisonnable)
  - Cibles FAIBLES (R²<0.6 en v7) : Optuna 15 trials sur XGB+LGB (gain potentiel)
  - Évaluation : CV3 GroupKFold pour mean ± std (plus stable qu'un split)
  - Ensemble : moyenne pondérée par R² des 3 modèles
  - Stacking optionnel sur les très faibles

Temps cible : ~30 min total
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
TOP_K = 80
N_CV = 3  # CV3 pour rapidité

df = pd.read_csv(DATA)
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
print(f"Dataset: {df.shape}\n", flush=True)

# Cibles dérivées
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
                 "target_stunting","target_fertility",
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
    return bl


def make_pre():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])

def select_top(X, y, k=TOP_K):
    X = X.dropna(axis=1, how="all")
    if X.shape[1] <= k: return list(X.columns)
    Xp = make_pre().fit_transform(X)
    rf = RandomForestRegressor(n_estimators=60, max_depth=10, min_samples_leaf=5,
                                n_jobs=-1, random_state=42)
    rf.fit(Xp, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(k).index.tolist()


def cv_score(X, y, groups, model_factory, n_folds=N_CV):
    gkf = GroupKFold(n_splits=n_folds)
    scores, maes = [], []
    for tr, te in gkf.split(X, y, groups=groups):
        pipe = Pipeline([("pre", make_pre()), ("model", model_factory())])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        scores.append(r2_score(y.iloc[te], pred))
        maes.append(mean_absolute_error(y.iloc[te], pred))
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(maes))


def cv_predict(X, y, groups, factory_dict, n_folds=N_CV):
    """Renvoie OOF preds par modèle + scores ensemble."""
    gkf = GroupKFold(n_splits=n_folds)
    oof = {name: np.zeros(len(y)) for name in factory_dict}
    fold_r2s = {name: [] for name in factory_dict}
    ensemble_scores = []
    for tr, te in gkf.split(X, y, groups=groups):
        preds = {}
        for name, fct in factory_dict.items():
            pipe = Pipeline([("pre", make_pre()), ("model", fct())])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            p = pipe.predict(X.iloc[te])
            preds[name] = p
            oof[name][te] = p
            fold_r2s[name].append(r2_score(y.iloc[te], p))
        # Ensemble pondéré par R² du fold
        r2s = [max(fold_r2s[m][-1], 0.01) for m in factory_dict]
        w = np.array(r2s) / sum(r2s)
        ens = sum(w[i] * preds[m] for i, m in enumerate(factory_dict))
        ensemble_scores.append(r2_score(y.iloc[te], ens))
    return oof, {m: (np.mean(fold_r2s[m]), np.std(fold_r2s[m])) for m in factory_dict}, (np.mean(ensemble_scores), np.std(ensemble_scores))


def optuna_tune(X, y, groups, model_type="xgb", n_trials=15, timeout=180):
    """Optuna rapide : 15 trials max, timeout 3 min."""
    def obj(trial):
        if model_type == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5, log=True),
            }
            f = lambda: XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "num_leaves": trial.suggest_int("num_leaves", 15, 100),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            }
            f = lambda: lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
        r2, _, _ = cv_score(X, y, groups, f, n_folds=3)
        return r2
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(obj, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study.best_params


# Cibles fortes / faibles
STRONG_R2 = {
    "target_forest_share": 0.86, "target_thermal_anomaly": 0.80,
    "target_fertility": 0.79, "target_child_mortality": 0.77,
    "target_birth_rate": 0.75, "target_yield_roots": 0.74,
    "target_life_expectancy": 0.70, "target_water_stress": 0.70,
    "target_soil_degradation": 0.66, "target_stunting": 0.65,
    "target_yield_oilcrops": 0.60, "target_yield_cereals": 0.60,
}
WEAK_TARGETS = [t for t in TARGETS if t not in STRONG_R2 and t in df.columns]
print(f"Fortes ({len(STRONG_R2)}) : ensemble simple")
print(f"Faibles ({len(WEAK_TARGETS)}) : Optuna XGB+LGB + ensemble\n", flush=True)

results = []
oof_store = {}

# ── PASS 1 — Toutes les cibles avec ensemble + Optuna pour faibles ──
print("══ PASS 1 — Ensemble RF+XGB+LGB ══", flush=True)
for tgt, label in TARGETS.items():
    if tgt not in df.columns: continue
    t0 = time.time()
    d = df.dropna(subset=[tgt]).copy()
    if len(d) < 200: continue

    bl = get_blacklist(tgt)
    feats = [c for c in d.columns if c not in bl and d[c].dtype != object]
    feats = [c for c in feats if d[c].notna().sum() > 0]
    X, y, groups = d[feats], d[tgt], d["ISO"]
    selected = select_top(X, y, k=TOP_K)
    Xs = X[selected]

    # Tuning pour cibles faibles
    if tgt in WEAK_TARGETS:
        print(f"\n🎯 {label} ({tgt}) — Optuna…", flush=True)
        try:
            xgb_p = optuna_tune(Xs, y, groups, "xgb", n_trials=12, timeout=120)
            lgb_p = optuna_tune(Xs, y, groups, "lgb", n_trials=12, timeout=120)
        except Exception as e:
            print(f"   Optuna err: {e}", flush=True)
            xgb_p = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05}
            lgb_p = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "num_leaves": 50}
    else:
        print(f"\n🎯 {label} ({tgt}) — grid simple…", flush=True)
        xgb_p = {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05,
                 "subsample": 0.8, "colsample_bytree": 0.8}
        lgb_p = {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.05,
                 "num_leaves": 50, "subsample": 0.8, "colsample_bytree": 0.8}

    factories = {
        "RF": lambda: RandomForestRegressor(n_estimators=250, max_depth=15,
                                              min_samples_leaf=4, n_jobs=-1, random_state=42),
        "XGB": lambda: XGBRegressor(**xgb_p, random_state=42, n_jobs=-1, verbosity=0),
        "LGB": lambda: lgb.LGBMRegressor(**lgb_p, random_state=42, n_jobs=-1, verbose=-1),
    }
    oof, indiv, (ens_mu, ens_sd) = cv_predict(Xs, y, groups, factories, n_folds=N_CV)

    # Meilleur
    best_name = "Ensemble"
    best_mu = ens_mu
    best_sd = ens_sd
    for name, (mu, sd) in indiv.items():
        if mu > best_mu:
            best_mu, best_sd, best_name = mu, sd, name

    dt = time.time() - t0
    for name, (mu, sd) in indiv.items():
        marker = " ★" if name == best_name else ""
        print(f"   [{name:8s}] R²= {mu:+.4f} ± {sd:.4f}{marker}", flush=True)
    marker = " ★" if best_name == "Ensemble" else ""
    print(f"   [Ensemble] R²= {ens_mu:+.4f} ± {ens_sd:.4f}{marker}", flush=True)
    print(f"   🏆 {best_name}  R²={best_mu:+.4f}  ({dt:.0f}s)", flush=True)

    # Store OOF des forts pour stacking
    if tgt in STRONG_R2 and best_mu > 0.6:
        oof_series = pd.Series(np.nan, index=df.index)
        oof_series.iloc[d.index] = oof.get(best_name) if best_name != "Ensemble" else (
            # Recalcule ensemble OOF
            (oof["RF"] + oof["XGB"] + oof["LGB"]) / 3
        )
        # plus simple : on prend oof du best
        if best_name in oof:
            oof_series = pd.Series(np.nan, index=df.index)
            oof_series.loc[d.index] = oof[best_name]
        else:
            oof_series = pd.Series(np.nan, index=df.index)
            oof_series.loc[d.index] = (oof["RF"] + oof["XGB"] + oof["LGB"]) / 3
        oof_store[tgt] = oof_series

    results.append({"Cible": label, "Technique": tgt,
                    "Meilleur": best_name,
                    "R² CV3 mean": round(best_mu, 4),
                    "R² CV3 std": round(best_sd, 4),
                    "N obs": len(d),
                    "Méthode": "Optuna+Ensemble" if tgt in WEAK_TARGETS else "Grid+Ensemble"})


# Inject OOF dans df
for tgt, s in oof_store.items():
    df[f"oof_{tgt}"] = s
print(f"\n→ {len(oof_store)} OOF strong générés pour stacking", flush=True)


# ── PASS 2 — Stacking sur cibles faibles encore <0.5 ──
print("\n══ PASS 2 — Stacking sur cibles très faibles ══", flush=True)
RESTACK = [r["Technique"] for r in results if r["R² CV3 mean"] < 0.5]
print(f"Cibles à re-stacker : {RESTACK}", flush=True)

for tgt in RESTACK:
    if tgt not in df.columns: continue
    label = TARGETS[tgt]
    exclude_oof = set()
    if tgt in SOCIO_TARGETS:
        for t2 in oof_store:
            if t2 in SOCIO_TARGETS: exclude_oof.add(f"oof_{t2}")
    if tgt in DISASTER_TARGETS:
        for t2 in oof_store:
            if t2 in DISASTER_TARGETS: exclude_oof.add(f"oof_{t2}")

    t0 = time.time()
    d = df.dropna(subset=[tgt]).copy()
    if len(d) < 200: continue
    bl = get_blacklist(tgt)
    bl |= exclude_oof
    feats = [c for c in d.columns if c not in bl and d[c].dtype != object]
    feats = [c for c in feats if d[c].notna().sum() > 0]
    X, y, groups = d[feats], d[tgt], d["ISO"]
    selected = select_top(X, y, k=TOP_K)
    Xs = X[selected]
    n_oof = sum(1 for f in selected if f.startswith("oof_"))

    try:
        xgb_p = optuna_tune(Xs, y, groups, "xgb", n_trials=10, timeout=90)
        lgb_p = optuna_tune(Xs, y, groups, "lgb", n_trials=10, timeout=90)
    except Exception:
        xgb_p = {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05}
        lgb_p = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "num_leaves": 40}

    factories = {
        "RF": lambda: RandomForestRegressor(n_estimators=250, max_depth=15,
                                              min_samples_leaf=4, n_jobs=-1, random_state=42),
        "XGB": lambda: XGBRegressor(**xgb_p, random_state=42, n_jobs=-1, verbosity=0),
        "LGB": lambda: lgb.LGBMRegressor(**lgb_p, random_state=42, n_jobs=-1, verbose=-1),
    }
    oof, indiv, (ens_mu, ens_sd) = cv_predict(Xs, y, groups, factories, n_folds=N_CV)
    best_name = "Ensemble"
    best_mu, best_sd = ens_mu, ens_sd
    for name, (mu, sd) in indiv.items():
        if mu > best_mu: best_mu, best_sd, best_name = mu, sd, name

    dt = time.time() - t0
    print(f"\n🎯 STACK {label}  (OOF kept: {n_oof})", flush=True)
    for name, (mu, sd) in indiv.items():
        marker = " ★" if name == best_name else ""
        print(f"   [{name:8s}] R²= {mu:+.4f} ± {sd:.4f}{marker}", flush=True)
    print(f"   [Ensemble] R²= {ens_mu:+.4f}  ({dt:.0f}s)", flush=True)
    # update si gain
    prev = next((r for r in results if r["Technique"] == tgt), None)
    if prev and best_mu > prev["R² CV3 mean"]:
        delta = best_mu - prev["R² CV3 mean"]
        print(f"   ⬆️  gain stacking +{delta:.3f}", flush=True)
        prev["Meilleur"] = best_name
        prev["R² CV3 mean"] = round(best_mu, 4)
        prev["R² CV3 std"] = round(best_sd, 4)
        prev["Méthode"] = "STACKING+Optuna+Ensemble"
    else:
        print(f"   ➡️  stacking n'aide pas", flush=True)


# Save
out = pd.DataFrame(results).sort_values("R² CV3 mean", ascending=False)
out.to_csv("reports/tableau_resultats_v8_ULTIMATE.csv", index=False)
print("\n══════════════════════════════════════════════════════════════════")
print("📊 RÉSULTATS V8 ULTIMATE (CV3 mean ± std)")
print("══════════════════════════════════════════════════════════════════")
print(out.to_string(index=False))
print("\n→ reports/tableau_resultats_v8_ULTIMATE.csv")
