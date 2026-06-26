"""
train_by_cluster.py — Entraîne un modèle PAR CLUSTER climatique pour chaque cible.

Stratégie :
  - 8 clusters climatiques (issus de impute_and_cluster.py)
  - Pour chaque cible × cluster : un modèle XGBoost dédié
  - Split par pays AU SEIN du cluster (test sur pays inconnus du cluster)
  - Comparaison vs modèle global pour démontrer le gain

Sortie : reports/cluster_results.csv + models_cluster/best_{target}_c{N}.joblib
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

DATA = "data/cleaned/dataset_final_v8_clean.csv"
os.makedirs("models_cluster", exist_ok=True)
os.makedirs("reports", exist_ok=True)
TOP_K = 60  # Plus petit car moins de données par cluster

df = pd.read_csv(DATA, low_memory=False)
df = df.dropna(subset=["ISO", "cluster"]).copy()
df["ISO"] = df["ISO"].astype(str)
df["cluster"] = df["cluster"].astype(int)
print(f"Dataset v8 clean : {df.shape}, {df['ISO'].nunique()} pays, {df['cluster'].nunique()} clusters\n")

# Mêmes configs anti-leak qu'avant
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
    "target_yield_cereals":["cereal_yield","cereal_production_t","food_production_index"],
    "target_yield_oilcrops":["food_production_index"],"target_yield_pulses":["food_production_index"],
    "target_yield_roots":["food_production_index"],"target_yield_fruits":["food_production_index"],
    "target_yield_vegetables":["food_production_index"],
    "target_water_stress":["freshwater_withdraw_total","freshwater_internal_per_cap","water_stress_ratio_raw"],
    "target_stunting":["wasting_pct","overweight_pct","malnutrition_compound","infant_deaths_total","Hunger_Index","ghi_owid"],
    "target_fertility":["Birth_Rate","Death_Rate","Pop_Growth","owid_crude_birth_rate","owid_crude_death_rate"],
    "target_birth_rate":["Death_Rate","Net_Migration","Pop_Growth","Fertility_Rate","owid_crude_birth_rate","owid_crude_death_rate","owid_births_total","owid_deaths_total"],
    "target_death_rate":["Birth_Rate","Net_Migration","Pop_Growth","Fertility_Rate","owid_crude_birth_rate","owid_crude_death_rate","owid_births_total","owid_deaths_total","adult_mortality_male","adult_mortality_female","infant_deaths_total"],
    "target_net_migration":["Birth_Rate","Death_Rate","Pop_Growth","Fertility_Rate","owid_crude_birth_rate","owid_crude_death_rate"],
    "target_soil_moisture_root":["nasa_gwettop","nasa_gwetprof","soil_moisture_deficit","combined_drought_index"],
    "target_forest_share":["forest_area_km2","forest_change","tree_cover_loss_ha","deforestation_annual","forest_per_capita_km2"],
    "target_tree_cover_loss":["forest_change","forest_share_pct","forest_area_km2","deforestation_annual","deforestation_pct_annual"],
    "target_thermal_anomaly":["owid_temp_anomaly","be_t_anom_annual","be_t_anom_vs_preindustrial"],
    "target_disaster_deaths":["disaster_affected","disaster_damages_usd","disaster_events"],
    "target_disaster_affected":["disaster_deaths","disaster_damages_usd","disaster_events"],
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
    # Drop les flags _cultivated (info redondante)
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
        # Pour cluster : drop autres cultures spécifiques
        for c in df.columns:
            if c.startswith("yield_") and not c.endswith("_kgha") and c != target.replace("target_",""):
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
    rf = RandomForestRegressor(n_estimators=60, max_depth=8, min_samples_leaf=3,
                               n_jobs=-1, random_state=42)
    rf.fit(Xp, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(k).index.tolist()


def train_on(d, target):
    """Entraîne RF + XGB sur un sous-ensemble (d) avec split par pays."""
    if len(d) < 100: return None
    if d[target].notna().sum() < 50: return None
    if d["ISO"].nunique() < 4: return None  # besoin de >= 4 pays pour split

    bl = get_bl(target)
    feats = [c for c in d.columns if c not in bl and d[c].dtype != object]
    feats = [c for c in feats if d[c].notna().sum() > 0]
    if not feats: return None

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    try:
        tr_idx, te_idx = next(splitter.split(d, groups=d["ISO"]))
    except Exception:
        return None
    tr, te = d.iloc[tr_idx], d.iloc[te_idx]
    if len(te) < 30 or te[target].notna().sum() < 20: return None

    Xtr_full, Xte_full = tr[feats], te[feats]
    ytr, yte = tr[target], te[target]
    sel = select_top(Xtr_full, ytr, k=min(TOP_K, len(feats)))
    Xtr, Xte = Xtr_full[sel], Xte_full[sel]

    # XGB
    pipe = Pipeline([("pre", make_pre()),
                     ("model", XGBRegressor(n_estimators=300, max_depth=5,
                                            learning_rate=0.05, subsample=0.85,
                                            colsample_bytree=0.85,
                                            random_state=42, n_jobs=-1, verbosity=0))])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    return {
        "r2": r2_score(yte, pred),
        "mae": mean_absolute_error(yte, pred),
        "n_train": len(tr), "n_test": len(te),
        "n_countries_train": tr["ISO"].nunique(),
        "n_countries_test": te["ISO"].nunique(),
        "pipe": pipe, "features": sel,
    }


# ── 1. Modèle GLOBAL (baseline) ────────────────────────────────────────────
print("══ PASS 1 — Modèle GLOBAL (baseline) ══")
global_results = {}
for tgt, label in TARGETS.items():
    if tgt not in df.columns: continue
    res = train_on(df, tgt)
    if res is None:
        print(f"⏭  {label} (insuffisant)")
        continue
    global_results[tgt] = res
    print(f"🌍 {label:30s}  R²={res['r2']:+.4f}  MAE={res['mae']:.3f}")


# ── 2. Modèles PAR CLUSTER ────────────────────────────────────────────────
print("\n══ PASS 2 — Modèles par cluster ══")
cluster_results = []
for tgt, label in TARGETS.items():
    if tgt not in df.columns: continue
    cluster_r2s = {}
    weighted_r2 = 0
    total_n = 0
    for c in sorted(df["cluster"].unique()):
        d_c = df[df["cluster"] == c]
        res = train_on(d_c, tgt)
        if res is None:
            cluster_r2s[c] = "N/A"
            continue
        cluster_r2s[c] = res["r2"]
        # Sauvegarde modèle cluster
        joblib.dump({"pipe": res["pipe"], "features": res["features"]},
                    f"models_cluster/{tgt}_c{c}.joblib")
        weighted_r2 += res["r2"] * res["n_test"]
        total_n += res["n_test"]
    avg_r2 = weighted_r2 / total_n if total_n > 0 else np.nan
    glob_r2 = global_results.get(tgt, {}).get("r2", np.nan)
    cluster_results.append({
        "Cible": label, "Technique": tgt,
        "R² Global": round(glob_r2, 4) if pd.notna(glob_r2) else "N/A",
        "R² Cluster (pondéré)": round(avg_r2, 4) if pd.notna(avg_r2) else "N/A",
        "Gain": round(avg_r2 - glob_r2, 4) if pd.notna(avg_r2) and pd.notna(glob_r2) else "N/A",
        "R² par cluster": " | ".join([f"{c}:{r:.2f}" if isinstance(r, float) else f"{c}:{r}"
                                       for c, r in cluster_r2s.items()]),
    })
    if pd.notna(avg_r2):
        sign = "↑" if avg_r2 > glob_r2 else "↓"
        delta = abs(avg_r2 - glob_r2)
        print(f"📊 {label:30s}  Global={glob_r2:+.3f}  Cluster={avg_r2:+.3f}  {sign}{delta:.3f}")


# ── Sauvegarde ────────────────────────────────────────────────────────────
out = pd.DataFrame(cluster_results)
out.to_csv("reports/cluster_results.csv", index=False)
print("\n📊 RÉSULTATS COMPARATIFS")
print("═" * 90)
print(out[["Cible", "R² Global", "R² Cluster (pondéré)", "Gain"]].to_string(index=False))
print(f"\n→ reports/cluster_results.csv ({len(out)} cibles)")
print(f"→ models_cluster/ ({sum(1 for _ in os.listdir('models_cluster'))} modèles sauvegardés)")
