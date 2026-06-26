"""
config_shared.py — Configuration partagée entre toutes les couches du projet.

Contient :
  - Anti-leak global (SOCIO_VARS, DISASTER_VARS)
  - Fonctions communes (blacklist, preprocessing, feature selection)
  - Datasets de référence
"""
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

DATA = "data/cleaned/dataset_final_v8_honest.csv"
TOP_K = 80

# ── Variables socio-éco (blacklistées pour cibles démo strictes env→démo) ─
SOCIO_VARS = [
    "Child_Mort","Life_Exp","Pop_Growth","Birth_Rate","Death_Rate","Net_Migration",
    "HDI","GDP_pc","GDP_total_usd","Population","Urban_pct","Inflation_CPI",
    "Unemployment","Gini","Poverty_190","Poverty_OWID","Debt_GDP","Trade_GDP",
    "Hunger_Index","Internet_pct","Mobile_subs","Electricity_pct","Energy_pc",
    "Health_GDP","Hospital_Beds","RD_GDP","Malaria","HIV","Deaths_Communicable",
    "Renew_Energy_pct","Energy_total","Life_Exp_OWID","Fertility_Rate","Internet_OWID",
    "stunting_pct","wasting_pct","overweight_pct","safe_water_pct","sanitation_pct",
    "physicians_per_1000","adult_mortality_male","adult_mortality_female",
    "infant_deaths_total","schooling_years","meat_consumption_pc","extreme_poverty_pct",
    "owid_births_total","owid_deaths_total","owid_crude_birth_rate","owid_crude_death_rate",
]

DISASTER_VARS = ["disaster_deaths","disaster_affected","disaster_damages_usd","disaster_events"]

ID_COLS = ["ISO","Annee","T_ref","P_ref"]


def load_dataset(path=DATA):
    """Charge le dataset V8 honnête."""
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["ISO"]).copy()
    df["ISO"] = df["ISO"].astype(str)
    df["Annee"] = df["Annee"].astype(int)
    if "cluster" in df.columns:
        df["cluster"] = df["cluster"].astype(int)
    return df


def make_preprocessor():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])


def make_xgb(n_estimators=500, max_depth=6, learning_rate=0.05):
    return XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, n_jobs=-1, verbosity=0,
    )


def select_top_features(X, y, k=TOP_K):
    """Top-k features par importance RandomForest."""
    X = X.dropna(axis=1, how="all")
    if X.shape[1] <= k:
        return list(X.columns)
    Xp = make_preprocessor().fit_transform(X)
    rf = RandomForestRegressor(n_estimators=80, max_depth=10, min_samples_leaf=5,
                                n_jobs=-1, random_state=42)
    rf.fit(Xp, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(k).index.tolist()


def build_blacklist(df, target, target_source=None, extra_leaks=None,
                    socio_targets=None, disaster_targets=None,
                    yield_targets=None, keep_cluster=False):
    """Construit la blacklist exhaustive pour une cible donnée.

    target_source : nom de la colonne source brute (ex: 'Birth_Rate' pour 'target_birth_rate')
    extra_leaks   : liste de colonnes additionnelles à drop
    socio_targets : si la cible est dedans → drop tout SOCIO_VARS
    disaster_targets : si dedans → drop DISASTER_VARS
    yield_targets : set de cibles agricoles → drop tous les yields
    keep_cluster  : si True, garde 'cluster' comme feature
    """
    bl = set(ID_COLS)
    if not keep_cluster:
        bl.add("cluster")
    # Toutes les cibles target_* sont blacklistées
    target_cols = [c for c in df.columns if c.startswith("target_")]
    bl |= set(target_cols)

    if target_source:
        bl |= {c for c in df.columns
               if c == target_source or c.startswith(target_source + "_") or c == target_source + "_ref"}

    if extra_leaks:
        for extra in extra_leaks:
            bl |= {c for c in df.columns if c == extra or c.startswith(extra + "_")}

    if yield_targets and target in yield_targets:
        yield_cols = [c for c in df.columns if c.startswith("yield_") and "kgha" in c]
        bl |= set(yield_cols)
        bl |= {"cereals_prod_t","cereals_area_ha","cereal_production_t",
               "food_production_index","cereal_yield"}
        for y in yield_cols + ["cereal_production_t","cereal_yield"]:
            bl |= {c for c in df.columns if c.startswith(y + "_")}
        # Cultures spécifiques aussi (anti-leak strict)
        for c in df.columns:
            if c.startswith("yield_") and not c.endswith("_kgha"):
                bl.add(c)

    if socio_targets and target in socio_targets:
        for v in SOCIO_VARS:
            bl |= {c for c in df.columns if c == v or c.startswith(v + "_")}

    if disaster_targets and target in disaster_targets:
        for v in DISASTER_VARS:
            bl |= {c for c in df.columns if c == v or c.startswith(v + "_")}

    return bl


def split_train_test(d, test_size=0.25, random_state=42):
    """GroupShuffleSplit par ISO."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, te_idx = next(splitter.split(d, groups=d["ISO"]))
    return d.iloc[tr_idx], d.iloc[te_idx]
