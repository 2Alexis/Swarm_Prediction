"""
fix_and_optimize_targets.py — Corrige cibles ratées + booste cibles faibles.

1. FIX PÊCHE : recréer target_fish_total/capture/aquaculture dans v13 + train
2. BOOST CARCASSES : enrichir avec features dérivées (densité, ratios)
3. BOOST WAHIS : décomposer par grandes catégories maladies
4. BOOST AUBERGINE : ajouter suit + précip culture-spécifique

Retrain seulement les 7-8 cibles concernées avec XGBoost optimisé.
"""
import os, sys, io, glob, warnings, time
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

CLN = "data/cleaned"
MODELS_DIR = "couche1_planete/models_cascade_v3"
os.makedirs(MODELS_DIR, exist_ok=True)


# ── 1. Charger V13 + recréer cibles pêche + nouvelles features ────────────
print("[1] Chargement V13 + corrections…\n")
df = pd.read_csv(f"{CLN}/shared/dataset_final_v13_couche1.csv", low_memory=False)
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
if "cluster" in df.columns:
    df["cluster"] = df["cluster"].astype(int)
print(f"   Shape : {df.shape}")

# === FIX PÊCHE : récréer cibles ===
fish_total = pd.read_csv(f"{CLN}/peche/fish_production_total.csv")
fish_src   = pd.read_csv(f"{CLN}/peche/fish_production_by_source.csv")
df = df.merge(fish_total, on=["ISO","Annee"], how="left", suffixes=("","_DUP"))
df = df.merge(fish_src,   on=["ISO","Annee"], how="left", suffixes=("","_DUP"))
# Drop dup cols
df = df.drop(columns=[c for c in df.columns if c.endswith("_DUP")])

# Combiner aquaculture
aqua_cols = [c for c in df.columns if "Aquaculture" in c and c.endswith("_t")]
if aqua_cols:
    df["fish_aquaculture_total_t"] = df[aqua_cols].sum(axis=1, min_count=1)

if "fish_total_t" in df.columns:
    df["target_fish_total"] = np.log1p(df["fish_total_t"].clip(lower=0))
    print(f"   ✓ target_fish_total recréée : {df['target_fish_total'].notna().sum()} obs")
if "fish_Capture_t" in df.columns:
    df["target_fish_capture"] = np.log1p(df["fish_Capture_t"].clip(lower=0))
    print(f"   ✓ target_fish_capture : {df['target_fish_capture'].notna().sum()} obs")
if "fish_aquaculture_total_t" in df.columns:
    df["target_fish_aquaculture"] = np.log1p(df["fish_aquaculture_total_t"].clip(lower=0))
    print(f"   ✓ target_fish_aquaculture : {df['target_fish_aquaculture'].notna().sum()} obs")


# ── 2. BOOST CARCASSES : features dérivées élevage ────────────────────────
print("\n[2] Boost carcasses — features dérivées…")

# Charger production_animaux pour features additionnelles
animaux = f"{CLN}/elevage" if False else None
# On a déjà livestock_cattle_heads, livestock_poultry_heads, etc. dans V13

# Densité bétail / surface agricole
if "livestock_cattle_heads" in df.columns and "Terres_agricoles_ha" in df.columns:
    df["cattle_per_ha"] = df["livestock_cattle_heads"] / df["Terres_agricoles_ha"].clip(lower=1)
    df["cattle_per_ha_log"] = np.log1p(df["cattle_per_ha"].fillna(0))
    print(f"   + cattle_per_ha + log")

# Ratio dairy animals / total cattle
if "livestock_dairy_animals" in df.columns and "livestock_cattle_heads" in df.columns:
    df["dairy_ratio"] = df["livestock_dairy_animals"] / df["livestock_cattle_heads"].clip(lower=1)
    print(f"   + dairy_ratio")

# Densité pigs / surface (utile pour porc)
if "wb_population_total" in df.columns and "Terres_agricoles_ha" in df.columns:
    df["pop_density_agri"] = df["wb_population_total"] / df["Terres_agricoles_ha"].clip(lower=1)
    print(f"   + pop_density_agri")

# Ratio investissement élevage proxy (PIB agri / population)
if "wb_gdp_pc" in df.columns and "agri_value_pct_gdp" in df.columns:
    df["agri_intensity_proxy"] = df["wb_gdp_pc"] * df["agri_value_pct_gdp"] / 100
    print(f"   + agri_intensity_proxy")

# Index élevage industriel (proxy : kg viande par habitant total)
meat_cols = [c for c in df.columns if c.startswith("meat_") and c.endswith("_kg_pc")]
if meat_cols:
    df["meat_total_kg_pc"] = df[meat_cols].sum(axis=1, min_count=1)
    df["meat_intensity_log"] = np.log1p(df["meat_total_kg_pc"].clip(lower=0))
    print(f"   + meat_total_kg_pc + log")

# GLW4 density / population (densité bétail per capita)
if "glw4_cattle_total" in df.columns and "wb_population_total" in df.columns:
    df["cattle_per_capita_glw4"] = df["glw4_cattle_total"] / df["wb_population_total"].clip(lower=1)
    print(f"   + cattle_per_capita_glw4")


# ── 3. BOOST WAHIS : grouper maladies par grandes catégories ──────────────
print("\n[3] Boost WAHIS — décomposition par catégorie maladie…")
wahis_src = f"data/raw/elevage/Données quantitatives 2026-06-24.csv"
if not os.path.exists(wahis_src):
    wahis_src = f"data/raw/misc/Données quantitatives 2026-06-24.csv"
if os.path.exists(wahis_src):
    try:
        wahis = pd.read_csv(wahis_src, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        wahis = pd.read_csv(wahis_src, low_memory=False, encoding="latin-1")

    sys.path.insert(0, ".")
    from build_dataset import custom_mappings, get_english_iso
    def name_to_iso2(name):
        if pd.isna(name): return None
        s = str(name).strip().lower()
        code = custom_mappings.get(s)
        if code: return code
        return get_english_iso(name)

    wahis["ISO"] = wahis["Pays"].apply(name_to_iso2)
    wahis = wahis.dropna(subset=["ISO","Année"])
    wahis["Année"] = pd.to_numeric(wahis["Année"], errors="coerce").astype("Int64")
    wahis = wahis.dropna(subset=["Année"])
    wahis["Année"] = wahis["Année"].astype(int)
    wahis = wahis.rename(columns={"Année": "Annee"})

    # Catégoriser maladies
    def categorize_disease(m):
        if pd.isna(m): return "Other"
        m = str(m).lower()
        if any(k in m for k in ["bird","avian","poultry","newcastle"]): return "Avian"
        if any(k in m for k in ["foot","fmd","mouth"]): return "FMD"
        if any(k in m for k in ["swine","african swine","classical swine","pig"]): return "Swine"
        if any(k in m for k in ["cattle","bovine","mad cow","bse"]): return "Cattle"
        if any(k in m for k in ["lumpy","skin"]): return "Lumpy"
        if any(k in m for k in ["rabies","fish","aqua","koi"]): return "Other"
        return "Other"
    wahis["Cat"] = wahis["Maladie"].apply(categorize_disease)

    # Pivot count par catégorie
    by_cat = wahis.groupby(["ISO","Annee","Cat"])["Outbreak_id"].nunique().unstack(fill_value=0).reset_index()
    rename = {c: f"wahis_{c.lower()}_outbreaks" for c in by_cat.columns if c not in ["ISO","Annee"]}
    by_cat = by_cat.rename(columns=rename)
    # Merger dans df
    df = df.merge(by_cat, on=["ISO","Annee"], how="left", suffixes=("","_DUP"))
    df = df.drop(columns=[c for c in df.columns if c.endswith("_DUP")])
    print(f"   + {len(rename)} colonnes wahis par catégorie")


# ── 4. BOOST AUBERGINE : features culture-spécifiques ─────────────────────
print("\n[4] Boost Aubergine — features culture-spécifiques…")
# spam_yield_vege déjà, suit_eggplant déjà
# Ajouter température culture-spécifique (eggplant T_opt 20-35°C)
if "nasa_t2m" in df.columns:
    df["eggplant_T_dist"] = abs(df["nasa_t2m"] - 27)  # T optimum = 27°C
    df["eggplant_T_score"] = (1 - df["eggplant_T_dist"]/15).clip(lower=0, upper=1)
    print(f"   + eggplant_T_score (proxy de viabilité)")
if "nasa_prectotcorr" in df.columns:
    df["eggplant_P_score"] = ((df["nasa_prectotcorr"] - 600) / 1000).clip(lower=0, upper=1)
    print(f"   + eggplant_P_score")


# ── 5. ENTRAÎNEMENT : XGBoost optimisé sur les cibles à corriger ──────────
print("\n[5] Entraînement cibles corrigées…")
TOP_K = 150
XGB_PARAMS_OPT = dict(n_estimators=600, max_depth=6, learning_rate=0.04,
                       subsample=0.85, colsample_bytree=0.85,
                       min_child_weight=3, reg_alpha=0.1, reg_lambda=0.5,
                       random_state=42, n_jobs=-1, verbosity=0)

def make_xgb_pipe():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("model", XGBRegressor(**XGB_PARAMS_OPT))])

def make_rf_for_selection():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("rf", RandomForestRegressor(n_estimators=80, max_depth=12,
                                                    min_samples_leaf=4, n_jobs=-1, random_state=42))])

def select_top_k(X, y, k=TOP_K):
    X = X.dropna(axis=1, how="all")
    if X.shape[1] <= k: return list(X.columns)
    pipe_rf = make_rf_for_selection()
    pipe_rf.fit(X, y)
    rf = pipe_rf.named_steps["rf"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(k).index.tolist()


# Cibles à entraîner + leur blacklist
TARGETS_TO_FIX = {
    # Pêche
    "target_fish_total":          {"src": "fish_total_t",
                                    "leaks": ["fish_Capture_t","fish_Aquaculture_Marine_t",
                                              "fish_Aquaculture_Freshwater_t","fish_Aquaculture_Brackish_t",
                                              "fish_species_count","fish_isscaap_count","aquaculture_t",
                                              "fish_aquaculture_total_t","target_fish_capture","target_fish_aquaculture"]},
    "target_fish_capture":        {"src": "fish_Capture_t",
                                    "leaks": ["fish_total_t","fish_Aquaculture_Marine_t",
                                              "fish_Aquaculture_Freshwater_t","fish_Aquaculture_Brackish_t",
                                              "fish_aquaculture_total_t","target_fish_total","target_fish_aquaculture"]},
    "target_fish_aquaculture":    {"src": "fish_aquaculture_total_t",
                                    "leaks": ["fish_total_t","fish_Capture_t","aquaculture_t",
                                              "fish_Aquaculture_Marine_t","fish_Aquaculture_Freshwater_t",
                                              "fish_Aquaculture_Brackish_t","target_fish_total","target_fish_capture"]},
    # Carcasses
    "target_cattle_carcass":      {"src": "livestock_cattle_carcass_kg",
                                    "leaks": ["livestock_cattle_slaughtered","meat_beef_kg_pc","glw4_"]},
    "target_pig_carcass":         {"src": "livestock_pig_carcass_kg",
                                    "leaks": ["livestock_pig_slaughtered","meat_pig_kg_pc"]},
    "target_sheepgoat_carcass":   {"src": "livestock_sheepgoat_carcass_kg",
                                    "leaks": ["livestock_sheepgoat_slaughtered","meat_sheepgoat_kg_pc"]},
    # WAHIS
    "target_animal_disease_outbreaks": {"src": "wahis_outbreaks_total",
                                         "leaks": ["wahis_diseases_unique","wahis_cases","wahis_deaths"]},
    # Aubergine
    "target_yield_eggplant":      {"src": "yield_eggplant",
                                    "leaks": []},
}

results = []
for tgt, info in TARGETS_TO_FIX.items():
    if tgt not in df.columns:
        print(f"   ⏭  {tgt} absent")
        continue

    # Construire features
    src = info["src"]
    leaks = info["leaks"]
    # Drop tous les target_*, source brute, leaks, ID
    drop_cols = {"ISO","Annee","T_ref","P_ref"}
    drop_cols |= {c for c in df.columns if c.startswith("target_")}
    if src:
        drop_cols |= {c for c in df.columns if c == src or c.startswith(src+"_")}
    for l in leaks:
        drop_cols |= {c for c in df.columns if c == l or c.startswith(l)}

    feats = [c for c in df.columns if c not in drop_cols and df[c].dtype != object]
    d = df.dropna(subset=[tgt]).copy()
    feats = [c for c in feats if d[c].notna().sum() > 0]

    if len(d) < 200 or d["ISO"].nunique() < 10:
        print(f"   ⏭  {tgt} (n={len(d)}, pays={d['ISO'].nunique()})")
        continue

    X_full, y, groups = d[feats], d[tgt], d["ISO"]
    X_full = X_full.replace([np.inf, -np.inf], np.nan)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, te_idx = next(splitter.split(d, groups=groups))

    sel = select_top_k(X_full.iloc[tr_idx], y.iloc[tr_idx], k=min(TOP_K, len(feats)))
    X = X_full[sel]

    t0 = time.time()
    pipe = make_xgb_pipe()
    pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred = pipe.predict(X.iloc[te_idx])
    r2 = r2_score(y.iloc[te_idx], pred)
    mae = mean_absolute_error(y.iloc[te_idx], pred)
    dt = time.time() - t0

    # Sauvegarder
    joblib.dump({"pipe": pipe, "features": sel, "sublayer": "fix"},
                f"{MODELS_DIR}/best_{tgt}.joblib")

    print(f"   🎯 {tgt:38s} R²={r2:+.4f}  MAE={mae:.3f}  ({len(sel)} feats, n={len(d)}) {dt:.0f}s")
    results.append({
        "Cible": tgt, "R²": round(r2, 4), "MAE": round(mae, 3),
        "N obs": len(d), "N features": len(sel),
    })


# ── 6. BILAN ──────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════════════════════")
print("📊 BILAN CORRECTIONS")
print("══════════════════════════════════════════════════════════════")
out = pd.DataFrame(results)
print(out.to_string(index=False))

# Comparaison avec v3
old = pd.read_csv("couche1_planete/reports/cascade_v3_results.csv")
old_map = dict(zip(old["Technique"], old["R² cascade V3"]))
print("\n📈 ÉVOLUTION vs V3 :")
for r in results:
    tgt = r["Cible"]
    new_r2 = r["R²"]
    old_r2 = old_map.get(tgt, None)
    if old_r2 is not None:
        delta = new_r2 - old_r2
        sign = "↑" if delta > 0 else "↓"
        print(f"   {tgt:38s} V3={old_r2:+.3f} → Fix={new_r2:+.3f}  {sign}{abs(delta):.3f}")
    else:
        print(f"   {tgt:38s} Fix={new_r2:+.3f}  (NOUVELLE)")

# Save report
out.to_csv("couche1_planete/reports/cascade_v3_fix_results.csv", index=False)
print(f"\n→ couche1_planete/reports/cascade_v3_fix_results.csv")
