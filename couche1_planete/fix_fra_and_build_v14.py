"""Fix FRA wide → long + Build v14 + retrain cibles faibles."""
import os, sys, io, glob, warnings, time
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

CLN = "data/cleaned"
RAW = "data/raw"


# ════════════════════════════════════════════════════════════════════════
print("[1] Fix FRA 2025 — wide → long…")
# ════════════════════════════════════════════════════════════════════════
fra_dir = f"{RAW}/bulk-download_fra_2025"
KEY_FILES = {
    "FRA_Years_variables/1a_forestArea_2026-06-25.csv":          "fra_forest_area_kha",
    "FRA_Years_variables/1a_landArea_2026-06-25.csv":            "fra_land_area_kha",
    "FRA_Years_variables/1a_otherWoodedLand_2026-06-25.csv":     "fra_other_wooded_kha",
    "FRA_Years_variables/1b_primary_2026-06-25.csv":              "fra_primary_forest_kha",
    "FRA_Years_variables/1b_naturallyRegeneratingForest_2026-06-25.csv": "fra_natural_regen_kha",
    "FRA_Years_variables/1b_plantedForest_2026-06-25.csv":        "fra_planted_forest_kha",
    "FRA_Years_variables/2c_agb_total_2026-06-25.csv":            "fra_biomass_agb_tot",
    "FRA_Years_variables/2d_carbon_agb_total_2026-06-25.csv":     "fra_carbon_agb_tot",
    "FRA_Years_variables/3b_protected_2026-06-25.csv":            "fra_forest_protected",
}

all_fra = []
for relpath, name in KEY_FILES.items():
    p = f"{fra_dir}/{relpath}"
    if not os.path.exists(p): continue
    try:
        df = pd.read_csv(p, low_memory=False)
        # Format wide : iso3, iso2, ..., 1990, 2000, 2010, 2015, 2020, 2025
        year_cols = [c for c in df.columns if str(c).strip().isdigit() and 1900 <= int(c) <= 2030]
        if not year_cols: continue
        # ISO2 direct
        iso_col = "iso2" if "iso2" in df.columns else "iso3"
        if iso_col == "iso3":
            import pycountry
            def iso3to2(c):
                try: return pycountry.countries.get(alpha_3=c).alpha_2
                except: return None
            df["ISO"] = df["iso3"].apply(iso3to2)
        else:
            df["ISO"] = df["iso2"]
        df = df.dropna(subset=["ISO"])
        long = df.melt(id_vars=["ISO"], value_vars=year_cols, var_name="Annee", value_name="value")
        long["Annee"] = long["Annee"].astype(int)
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long.dropna(subset=["value"])
        agg = long.groupby(["ISO","Annee"])["value"].mean().reset_index().rename(columns={"value": name})
        all_fra.append(agg)
        print(f"   + {name}: {len(agg)} obs, {agg['ISO'].nunique()} pays")
    except Exception as e:
        print(f"   ⚠️ {name}: {str(e)[:80]}")

if all_fra:
    fra_merged = all_fra[0]
    for d in all_fra[1:]:
        fra_merged = fra_merged.merge(d, on=["ISO","Annee"], how="outer")
    fra_merged.to_csv(f"{CLN}/sol_ecologie/fao_fra2025_forest_indicators.csv", index=False)
    print(f"   ✓ → fao_fra2025_forest_indicators.csv ({len(fra_merged)} lignes × {fra_merged.shape[1]-2} indic)")


# ════════════════════════════════════════════════════════════════════════
print("\n[2] Build v14 = v13 + tous nouveaux datasets…")
# ════════════════════════════════════════════════════════════════════════
df = pd.read_csv(f"{CLN}/shared/dataset_final_v13_couche1.csv", low_memory=False)
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
if "cluster" in df.columns:
    df["cluster"] = df["cluster"].astype(int)
print(f"   départ : {df.shape}")

def merge_safe(df, path, label):
    if not os.path.exists(path): return df, 0
    try:
        sub = pd.read_csv(path, low_memory=False)
        if "ISO" not in sub.columns: return df, 0
        keys = ["ISO","Annee"] if "Annee" in sub.columns else ["ISO"]
        if "ISO" in sub.columns:
            sub["ISO"] = sub["ISO"].astype(str)
        if "Annee" in sub.columns:
            sub["Annee"] = pd.to_numeric(sub["Annee"], errors="coerce").astype("Int64")
            sub = sub.dropna(subset=["Annee"])
            sub["Annee"] = sub["Annee"].astype(int)
        new_cols = [c for c in sub.columns if c in keys or c not in df.columns]
        sub = sub[new_cols]
        df = df.merge(sub, on=keys, how="left", suffixes=("","_DUP"))
        df = df.drop(columns=[c for c in df.columns if c.endswith("_DUP")])
        added = len([c for c in sub.columns if c not in keys])
        print(f"   + {label}: +{added} cols")
        return df, added
    except Exception as e:
        print(f"   ⚠️ {label}: {str(e)[:80]}")
        return df, 0

# Merge tous les nouveaux
NEW_FILES = [
    (f"{CLN}/sol_ecologie/fao_fra2025_forest_indicators.csv", "FRA 2025 forest"),
    (f"{CLN}/sol_ecologie/gbif_occurrences_by_country.csv", "GBIF occurrences"),
    (f"{CLN}/geologie/earthquakes_M5plus_by_country_year.csv", "USGS NEIC M5+ historique"),
    (f"{CLN}/agriculture/fao_crops_livestock_yields_top30.csv", "FAO Crops+Livestock yields"),
    (f"{CLN}/agriculture/fao_fertilizers_by_product.csv", "FAO Fertilizers by Product"),
    (f"{CLN}/atmosphere/fao_emissions_intensities.csv", "FAO Emissions intensities"),
    (f"{CLN}/demographie/who_immunization_coverage.csv", "WHO Immunization"),
    (f"{CLN}/elevage/eurostat_cattle_livestock.csv", "Eurostat cattle"),
    (f"{CLN}/elevage/eurostat_pig_livestock.csv", "Eurostat pig"),
    (f"{CLN}/elevage/eurostat_goat_livestock.csv", "Eurostat goat"),
]
for path, label in NEW_FILES:
    df, n = merge_safe(df, path, label)

# Imputation honnête sur les nouvelles features
v13_cols = set(pd.read_csv(f"{CLN}/shared/dataset_final_v13_couche1.csv", nrows=0).columns)
new_cols = [c for c in df.columns if c not in v13_cols and not c.startswith("target_")
              and df[c].dtype in ("float64","int64")]
print(f"   {len(new_cols)} nouvelles features à imputer")
df = df.sort_values(["ISO","Annee"]).reset_index(drop=True)
for c in new_cols:
    df[c] = df.groupby("ISO")[c].transform(lambda s: s.interpolate(method="linear", limit_direction="both", limit=5))
    df[c] = df.groupby("ISO")[c].transform(lambda s: s.ffill().bfill())
    if "cluster" in df.columns:
        df[c] = df.groupby(["cluster","Annee"])[c].transform(lambda s: s.fillna(s.median()))
    df[c] = df.groupby("Annee")[c].transform(lambda s: s.fillna(s.median()))

out = f"{CLN}/shared/dataset_final_v14_couche1.csv"
df.to_csv(out, index=False)
print(f"\n[OK] {out}")
print(f"     Shape : {df.shape}  (+{df.shape[1] - 1070} cols vs v13)")


# ════════════════════════════════════════════════════════════════════════
print("\n[3] Retrain cibles faibles avec v14…")
# ════════════════════════════════════════════════════════════════════════

XGB = dict(n_estimators=600, max_depth=6, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            min_child_weight=3, reg_alpha=0.1, reg_lambda=0.5,
            random_state=42, n_jobs=-1, verbosity=0)

def make_pipe():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("model", XGBRegressor(**XGB))])

def select_top(X, y, k=150):
    X = X.dropna(axis=1, how="all")
    if X.shape[1] <= k: return list(X.columns)
    pipe_rf = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=4,
                                       n_jobs=-1, random_state=42))])
    pipe_rf.fit(X, y)
    rf = pipe_rf.named_steps["rf"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(k).index.tolist()

TARGETS_REFIT = {
    "target_seismic_activity":   ["eq_count","eq_max_mag","eq_mean_mag","eq_mag_ge6","eq_count_m5",
                                    "eq_max_mag_hist","eq_mean_mag_hist","eq_mag_ge6_hist","eq_mag_ge7_hist",
                                    "earthquake_count","earthquake_max_mag"],
    "target_biodiversity_species":["iucn_observations","gbif_occurrences"],
    "target_forest_share":       ["forest_area_km2","forest_change","tree_cover_loss_ha",
                                    "deforestation_annual","forest_per_capita_km2",
                                    "fra_forest_area_kha","fra_other_wooded_kha"],
    "target_tree_cover_loss":    ["forest_change","forest_share_pct","forest_area_km2",
                                    "deforestation_annual","fra_deforestation"],
    "target_cattle_carcass":     ["livestock_cattle_slaughtered","meat_beef_kg_pc","glw4_"],
    "target_pig_carcass":        ["livestock_pig_slaughtered","meat_pig_kg_pc"],
    "target_sheepgoat_carcass":  ["livestock_sheepgoat_slaughtered","meat_sheepgoat_kg_pc"],
    "target_chicken_carcass":    ["livestock_chicken_slaughtered","meat_poultry_kg_pc"],
    "target_yield_cereals":      ["cereal_yield","cereal_production_t","food_production_index"],
    "target_yield_rapeseed":     ["yield_rapeseed_lag"],
    "target_yield_drypea":       ["yield_drypea_lag"],
    "target_yield_strawberry":   [],
    "target_yield_apple":        [],
    "target_yield_drybean":      [],
    "target_yield_eggplant":     [],
}

TARGET_SOURCES = {
    "target_seismic_activity": "eq_count",
    "target_biodiversity_species": "iucn_species_count",
    "target_forest_share": "forest_share_pct",
    "target_tree_cover_loss": "tree_cover_loss_ha",
    "target_cattle_carcass": "livestock_cattle_carcass_kg",
    "target_pig_carcass": "livestock_pig_carcass_kg",
    "target_sheepgoat_carcass": "livestock_sheepgoat_carcass_kg",
    "target_chicken_carcass": "livestock_chicken_carcass_g",
    "target_yield_cereals": "yield_cereals_kgha",
    "target_yield_rapeseed": "yield_rapeseed",
    "target_yield_drypea": "yield_drypea",
    "target_yield_strawberry": "yield_strawberry",
    "target_yield_apple": "yield_apple",
    "target_yield_drybean": "yield_drybean",
    "target_yield_eggplant": "yield_eggplant",
}

# Load v3 results for comparison
old = pd.read_csv("couche1_planete/reports/cascade_v3_results.csv")
old_map = dict(zip(old["Technique"], old["R² cascade V3"]))

results = []
for tgt, extra_leaks in TARGETS_REFIT.items():
    if tgt not in df.columns:
        print(f"   ⏭ {tgt} absent")
        continue
    src = TARGET_SOURCES.get(tgt)
    drop_cols = {"ISO","Annee","T_ref","P_ref"}
    drop_cols |= {c for c in df.columns if c.startswith("target_")}
    if src:
        drop_cols |= {c for c in df.columns if c == src or c.startswith(src+"_")}
    # Yield cross-leaks
    if tgt.startswith("target_yield_"):
        drop_cols |= {c for c in df.columns if c.startswith("yield_") and c not in (src,)}
        drop_cols |= {c for c in df.columns if c.startswith("spam_yield_") or c.startswith("spam_harvest_")}
    for l in extra_leaks:
        drop_cols |= {c for c in df.columns if c == l or c.startswith(l)}

    feats = [c for c in df.columns if c not in drop_cols and df[c].dtype != object]
    d = df.dropna(subset=[tgt]).copy()
    feats = [c for c in feats if d[c].notna().sum() > 0]
    if len(d) < 200 or d["ISO"].nunique() < 10: continue

    X_full, y, groups = d[feats], d[tgt], d["ISO"]
    X_full = X_full.replace([np.inf, -np.inf], np.nan)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(splitter.split(d, groups=groups))
    sel = select_top(X_full.iloc[tr_idx], y.iloc[tr_idx], k=150)
    X = X_full[sel]
    t0 = time.time()
    pipe = make_pipe()
    pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred = pipe.predict(X.iloc[te_idx])
    r2 = r2_score(y.iloc[te_idx], pred)
    mae = mean_absolute_error(y.iloc[te_idx], pred)
    dt = time.time() - t0

    old_r2 = old_map.get(tgt)
    delta_str = f"({'+' if r2-old_r2>=0 else ''}{r2-old_r2:.3f})" if old_r2 is not None else "(NEW)"
    print(f"   🎯 {tgt:35s} R²={r2:+.4f} MAE={mae:.3f}  {delta_str}  {dt:.0f}s")

    joblib.dump({"pipe": pipe, "features": sel, "sublayer":"v14_fix"},
                f"couche1_planete/models_cascade_v3/best_{tgt}.joblib")
    results.append({"Cible":tgt, "R²V14":round(r2,4), "R²V3":round(old_r2,4) if old_r2 else None,
                    "Δ":round(r2-old_r2,4) if old_r2 else None, "MAE":round(mae,3), "N":len(d)})

out_path = "couche1_planete/reports/cascade_v14_fix_results.csv"
pd.DataFrame(results).to_csv(out_path, index=False)
print(f"\n→ {out_path}")
