"""
train_v8_honest.py — Entraînement sur dataset_final_v8_honest.csv

Stratégies comparées :
  A. Modèle GLOBAL (baseline) sur dataset honnête
  B. Modèle PAR CLUSTER (gain attendu sur cibles avec spécificités régionales)
  C. Modèle global + feature cluster (compromis)

Sortie : reports/v8_honest_results.csv
"""
import os, sys, io, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA = "data/cleaned/dataset_final_v8_honest.csv"
os.makedirs("models_v8", exist_ok=True)
os.makedirs("models_cluster", exist_ok=True)
TOP_K = 80

df = pd.read_csv(DATA, low_memory=False)
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
df["cluster"] = df["cluster"].astype(int)
print(f"Dataset v8 honest : {df.shape}\n")

TARGETS = {
    "target_yield_cereals":"Rendement céréales","target_yield_oilcrops":"Rendement oléagineux",
    "target_yield_pulses":"Rendement légumineuses","target_yield_roots":"Rendement racines",
    "target_yield_fruits":"Rendement fruits","target_yield_vegetables":"Rendement légumes",
    "target_water_stress":"Stress hydrique","target_soil_degradation":"Dégradation sol",
    "target_thermal_anomaly":"Anomalie thermique","target_soil_moisture_root":"Humidité sol",
    "target_forest_share":"% forêt","target_tree_cover_loss":"Perte couvert arboré",
    "target_child_mortality":"Mortalité infantile","target_life_expectancy":"Espérance de vie",
    "target_pop_growth":"Croissance démo","target_birth_rate":"Natalité",
    "target_death_rate":"Mortalité brut","target_net_migration":"Migration nette",
    "target_fertility":"Fécondité","target_stunting":"Stunting",
    "target_disaster_deaths":"Décès catastrophes","target_disaster_affected":"Affectés catastrophes",
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
    "target_birth_rate":["Death_Rate","Net_Migration","Pop_Growth","Fertility_Rate","owid_crude_birth_rate","owid_crude_death_rate","owid_births_total","owid_deaths_total","births_per_1000_minus_deaths","target_birth_rate_owid"],
    "target_death_rate":["Birth_Rate","Net_Migration","Pop_Growth","Fertility_Rate","owid_crude_birth_rate","owid_crude_death_rate","owid_births_total","owid_deaths_total","births_per_1000_minus_deaths","adult_mortality_male","adult_mortality_female","infant_deaths_total","target_death_rate_owid"],
    "target_net_migration":["Birth_Rate","Death_Rate","Pop_Growth","Fertility_Rate","owid_crude_birth_rate","owid_crude_death_rate","owid_births_total","owid_deaths_total","refugees_origin","refugees_destination","refugees_origin_owid","idps_conflict"],
    "target_soil_moisture_root":["nasa_gwettop","nasa_gwetprof","soil_moisture_deficit","combined_drought_index","heat_drought_stress","soil_moisture_top_root_ratio"],
    "target_forest_share":["forest_area_km2","forest_change","tree_cover_loss_ha","tree_cover_loss_cumul5y","deforestation_annual","deforestation_pct_annual","forest_per_capita_km2"],
    "target_tree_cover_loss":["forest_change","forest_share_pct","forest_area_km2","deforestation_annual","deforestation_pct_annual","tree_cover_loss_cumul5y","forest_per_capita_km2"],
    "target_thermal_anomaly":["owid_temp_anomaly","be_t_anom_annual","be_t_anom_vs_preindustrial","be_t_baseline_1850_1900"],
    "target_disaster_deaths":["disaster_affected","disaster_damages_usd","disaster_events","disaster_affected_per_capita","disaster_deaths_per_million","disaster_events_cumul5y"],
    "target_disaster_affected":["disaster_deaths","disaster_damages_usd","disaster_events","disaster_affected_per_capita","disaster_deaths_per_million","disaster_events_cumul5y"],
}

SOCIO_VARS = ["Child_Mort","Life_Exp","Pop_Growth","Birth_Rate","Death_Rate","Net_Migration",
              "HDI","GDP_pc","GDP_total_usd","Population","Urban_pct","Inflation_CPI",
              "Unemployment","Gini","Poverty_190","Poverty_OWID","Debt_GDP","Trade_GDP",
              "Hunger_Index","Internet_pct","Mobile_subs","Electricity_pct","Energy_pc",
              "Health_GDP","Hospital_Beds","RD_GDP","Malaria","HIV","Deaths_Communicable",
              "Renew_Energy_pct","Energy_total","Life_Exp_OWID","Fertility_Rate","Internet_OWID",
              "stunting_pct","wasting_pct","overweight_pct","safe_water_pct","sanitation_pct",
              "physicians_per_1000","adult_mortality_male","adult_mortality_female",
              "infant_deaths_total","schooling_years","meat_consumption_pc","extreme_poverty_pct",
              "owid_births_total","owid_deaths_total","owid_crude_birth_rate","owid_crude_death_rate"]
SOCIO_TARGETS = {"target_child_mortality","target_life_expectancy","target_pop_growth",
                 "target_stunting","target_fertility","target_birth_rate",
                 "target_death_rate","target_net_migration"}
DISASTER_VARS = ["disaster_deaths","disaster_affected","disaster_damages_usd","disaster_events"]
DISASTER_TARGETS = {"target_disaster_deaths","target_disaster_affected"}

ID_COLS = ["ISO","Annee","T_ref","P_ref","cluster"]
TARGET_COLS = [c for c in df.columns if c.startswith("target_")]
YIELD_COLS = [c for c in df.columns if c.startswith("yield_") and "kgha" in c]


def get_bl(target):
    bl = set(ID_COLS) | set(TARGET_COLS)
    # Drop les flags _cultivated (info de leak)
    bl |= {c for c in df.columns if c.endswith("_cultivated")}
    src = TARGET_SOURCE.get(target)
    if src:
        bl |= {c for c in df.columns if c == src or c.startswith(src + "_") or c == src + "_ref"}
    for extra in EXTRA_LEAKS.get(target, []):
        bl |= {c for c in df.columns if c == extra or c.startswith(extra + "_")}
    if target.startswith("target_yield_"):
        bl |= set(YIELD_COLS) | {"cereals_prod_t","cereals_area_ha","cereal_production_t","food_production_index","cereal_yield"}
        for y in YIELD_COLS + ["cereal_production_t","cereal_yield"]:
            bl |= {c for c in df.columns if c.startswith(y + "_")}
        # Drop autres cultures spécifiques
        for c in df.columns:
            if c.startswith("yield_") and not c.endswith("_kgha"):
                bl.add(c)
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
    rf = RandomForestRegressor(n_estimators=80, max_depth=10, min_samples_leaf=5,
                               n_jobs=-1, random_state=42)
    rf.fit(Xp, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(k).index.tolist()


def train_global(target):
    """Modèle global standard."""
    d = df.dropna(subset=[target]).copy()
    if len(d) < 200: return None
    bl = get_bl(target)
    feats = [c for c in d.columns if c not in bl and d[c].dtype != object]
    feats = [c for c in feats if d[c].notna().sum() > 0]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(splitter.split(d, groups=d["ISO"]))
    tr, te = d.iloc[tr_idx], d.iloc[te_idx]
    Xtr_full, Xte_full = tr[feats], te[feats]
    ytr, yte = tr[target], te[target]
    sel = select_top(Xtr_full, ytr)
    Xtr, Xte = Xtr_full[sel], Xte_full[sel]

    pipe = Pipeline([("pre", make_pre()),
                     ("model", XGBRegressor(n_estimators=500, max_depth=6,
                                            learning_rate=0.05, subsample=0.85,
                                            colsample_bytree=0.85,
                                            random_state=42, n_jobs=-1, verbosity=0))])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    return {"r2": r2_score(yte, pred), "mae": mean_absolute_error(yte, pred),
            "n_obs": len(d), "pipe": pipe, "features": sel}


def train_with_cluster_feature(target):
    """Modèle global avec le numéro de cluster comme feature."""
    d = df.dropna(subset=[target]).copy()
    if len(d) < 200: return None
    bl = get_bl(target) - {"cluster"}  # Remettre cluster comme feature
    feats = [c for c in d.columns if c not in bl and d[c].dtype != object]
    feats = [c for c in feats if d[c].notna().sum() > 0]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(splitter.split(d, groups=d["ISO"]))
    tr, te = d.iloc[tr_idx], d.iloc[te_idx]
    Xtr_full, Xte_full = tr[feats], te[feats]
    ytr, yte = tr[target], te[target]
    sel = select_top(Xtr_full, ytr)
    if "cluster" not in sel: sel = ["cluster"] + sel[:-1]
    Xtr, Xte = Xtr_full[sel], Xte_full[sel]

    pipe = Pipeline([("pre", make_pre()),
                     ("model", XGBRegressor(n_estimators=500, max_depth=6,
                                            learning_rate=0.05, subsample=0.85,
                                            colsample_bytree=0.85,
                                            random_state=42, n_jobs=-1, verbosity=0))])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    return {"r2": r2_score(yte, pred), "mae": mean_absolute_error(yte, pred),
            "pipe": pipe, "features": sel}


def train_by_cluster(target):
    """Un modèle par cluster, prédictions pondérées."""
    weighted_r2 = 0
    total_n = 0
    per_cluster_r2 = {}
    for c in sorted(df["cluster"].unique()):
        d_c = df[df["cluster"] == c].dropna(subset=[target])
        if len(d_c) < 100 or d_c["ISO"].nunique() < 4:
            per_cluster_r2[c] = None
            continue
        bl = get_bl(target)
        feats = [col for col in d_c.columns if col not in bl and d_c[col].dtype != object]
        feats = [col for col in feats if d_c[col].notna().sum() > 0]
        if not feats: continue
        try:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
            tr_idx, te_idx = next(splitter.split(d_c, groups=d_c["ISO"]))
        except Exception:
            continue
        tr, te = d_c.iloc[tr_idx], d_c.iloc[te_idx]
        if len(te) < 30: continue
        Xtr, Xte = tr[feats], te[feats]
        ytr, yte = tr[target], te[target]
        sel = select_top(Xtr, ytr, k=min(40, len(feats)))
        Xtr, Xte = Xtr[sel], Xte[sel]
        try:
            pipe = Pipeline([("pre", make_pre()),
                             ("model", XGBRegressor(n_estimators=300, max_depth=5,
                                                    learning_rate=0.05,
                                                    random_state=42, n_jobs=-1, verbosity=0))])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            r2 = r2_score(yte, pred)
            per_cluster_r2[c] = r2
            joblib.dump({"pipe": pipe, "features": sel},
                        f"models_cluster/{target}_c{c}.joblib")
            weighted_r2 += r2 * len(te)
            total_n += len(te)
        except Exception:
            continue
    avg_r2 = weighted_r2 / total_n if total_n > 0 else np.nan
    return {"r2": avg_r2, "per_cluster": per_cluster_r2}


results = []
print("══ Comparaison des 3 stratégies ══\n")
print(f"{'Cible':<28s} {'Global':>8s} {'Global+C':>10s} {'PerCluster':>11s}")
print("-" * 60)

for tgt, label in TARGETS.items():
    if tgt not in df.columns: continue

    res_g = train_global(tgt)
    if res_g is None:
        continue
    if res_g["r2"] is not None:
        joblib.dump({"pipe": res_g["pipe"], "features": res_g["features"]},
                    f"models_v8/best_{tgt}.joblib")

    res_gc = train_with_cluster_feature(tgt)
    res_c = train_by_cluster(tgt)

    g  = res_g["r2"]
    gc = res_gc["r2"] if res_gc else np.nan
    c  = res_c["r2"] if res_c and pd.notna(res_c["r2"]) else np.nan

    print(f"{label[:27]:<28s} {g:>+7.3f} {gc:>+9.3f} {c:>+10.3f}")

    results.append({
        "Cible": label, "Technique": tgt,
        "R² Global": round(g, 4),
        "R² Global+Cluster": round(gc, 4) if pd.notna(gc) else None,
        "R² PerCluster": round(c, 4) if pd.notna(c) else None,
        "MAE Global": round(res_g["mae"], 3),
        "N obs": res_g["n_obs"],
    })

out = pd.DataFrame(results)
out.to_csv("reports/v8_honest_results.csv", index=False)
print("\n══════════════════════════════════════════════════════════")
print("📊 RÉSULTATS V8 HONNÊTE")
print("══════════════════════════════════════════════════════════")
out_sorted = out.sort_values("R² Global", ascending=False)
print(out_sorted.to_string(index=False))
print(f"\n→ reports/v8_honest_results.csv")
