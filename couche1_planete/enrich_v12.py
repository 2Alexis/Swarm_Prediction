"""
enrich_v12.py — V11 → V12 : suitability adoucie + indices climatiques dérivés.

Améliorations :
  - Suitability EcoCrop : remplace les 0 stricts (lat/frost) par décroissance douce
  - Heat stress index (jours T > 30°C estimés)
  - Cold stress / frost risk
  - Aridity index calculé (P/PET)
  - Growing season proxies (T moy × P moy)
  - Continentality (amplitude T)
  - PET (évapotranspiration Thornthwaite simplifiée)
"""
import os, sys, io
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = "data/cleaned"

print("[1] Chargement v11…")
df = pd.read_csv(f"{D}/dataset_final_v11_couche1.csv", low_memory=False)
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
print(f"   shape : {df.shape}")


# ── 2. Recalculer suitability avec décroissance douce ──────────────────────
print("\n[2] Suitability adoucie (sans 0 stricts)…")

CROP_SCIENTIFIC = {
    "wheat":("Triticum aestivum",None), "rice":("Oryza sativa",None), "maize":("Zea mays",None),
    "soybeans":("Glycine max",None), "rapeseed":("Brassica napus",None),
    "sunflower":("Helianthus annuus",None), "groundnut":("Arachis hypogaea",None),
    "olives":("Olea europaea",None), "sesame":("Sesamum indicum",None),
    "coconut":("Cocos nucifera",None), "cotton":("Gossypium hirsutum",None),
    "apple":("Malus domestica",["Malus pumila"]),
    "banana":("Musa acuminata",["Musa x paradisiaca","Musa sapientum"]),
    "orange":("Citrus sinensis",None), "grape":("Vitis vinifera",None),
    "strawberry":("Fragaria x ananassa",["Fragaria vesca"]),
    "pineapple":("Ananas comosus",None), "mango":("Mangifera indica",None),
    "avocado":("Persea americana",None), "lemon":("Citrus limon",None),
    "peach":("Prunus persica",None), "pear":("Pyrus communis",None),
    "watermelon":("Citrullus lanatus",None), "dates":("Phoenix dactylifera",None),
    "apricot":("Prunus armeniaca",None), "cherry":("Prunus avium",None),
    "plum":("Prunus domestica",None),
    "tomato":("Solanum lycopersicum",["Lycopersicon esculentum"]),
    "potato":("Solanum tuberosum",None), "onion":("Allium cepa",None),
    "cabbage":("Brassica oleracea",None), "carrot":("Daucus carota",None),
    "cucumber":("Cucumis sativus",None), "eggplant":("Solanum melongena",None),
    "cauliflower":("Brassica oleracea var. botrytis",["Brassica oleracea"]),
    "lettuce":("Lactuca sativa",None),
    "chickpea":("Cicer arietinum",None), "drybean":("Phaseolus vulgaris",None),
    "drypea":("Pisum sativum",None),
}
eco = pd.read_csv(f"{D}/EcoCrop_DB.csv", low_memory=False, encoding="latin-1")

def find_in_ecocrop(scientific, synonyms):
    for name in [scientific] + (synonyms or []):
        rows = eco[eco["ScientificName"].str.contains(name, case=False, na=False, regex=False)]
        if not rows.empty:
            return rows.iloc[0]
    return None

def smooth_score(x, opt_min, opt_max, abs_min, abs_max, edge_score=0.3):
    """Score 0.3-1 : entre opt = 1, hors abs_min/max = edge_score (sans descendre à 0)."""
    x = np.asarray(x, dtype=float)
    if pd.isna(opt_min) or pd.isna(opt_max): return np.ones_like(x) * 0.5
    if pd.isna(abs_min): abs_min = opt_min - (opt_max - opt_min) * 1.5
    if pd.isna(abs_max): abs_max = opt_max + (opt_max - opt_min) * 1.5
    score = np.full_like(x, edge_score, dtype=float)
    mask_opt = (x >= opt_min) & (x <= opt_max)
    score[mask_opt] = 1.0
    mask_low = (x >= abs_min) & (x < opt_min)
    score[mask_low] = edge_score + (1 - edge_score) * (x[mask_low] - abs_min) / max(opt_min - abs_min, 0.001)
    mask_hi = (x > opt_max) & (x <= abs_max)
    score[mask_hi] = edge_score + (1 - edge_score) * (abs_max - x[mask_hi]) / max(abs_max - opt_max, 0.001)
    return np.clip(score, edge_score, 1)


T_col = "nasa_t2m"
P_col = "nasa_prectotcorr"
T_min_col = "nasa_t2m_min"

for crop, (sci, syn) in CROP_SCIENTIFIC.items():
    row = find_in_ecocrop(sci, syn)
    if row is None: continue
    p = {
        "T_opt_min": pd.to_numeric(row.get("TOPMN"), errors="coerce"),
        "T_opt_max": pd.to_numeric(row.get("TOPMX"), errors="coerce"),
        "T_abs_min": pd.to_numeric(row.get("TMIN"),  errors="coerce"),
        "T_abs_max": pd.to_numeric(row.get("TMAX"),  errors="coerce"),
        "P_opt_min": pd.to_numeric(row.get("ROPMN"), errors="coerce"),
        "P_opt_max": pd.to_numeric(row.get("ROPMX"), errors="coerce"),
        "P_abs_min": pd.to_numeric(row.get("RMIN"),  errors="coerce"),
        "P_abs_max": pd.to_numeric(row.get("RMAX"),  errors="coerce"),
        "K_TMP":     pd.to_numeric(row.get("KTMP"),  errors="coerce"),
        "Lat_max":   pd.to_numeric(row.get("LATMX"), errors="coerce"),
    }
    T_score = smooth_score(df[T_col].values, p["T_opt_min"], p["T_opt_max"],
                            p["T_abs_min"], p["T_abs_max"], edge_score=0.2)
    P_score = smooth_score(df[P_col].values, p["P_opt_min"], p["P_opt_max"],
                            p["P_abs_min"], p["P_abs_max"], edge_score=0.2)
    # Frost : pénalisation continue
    if T_min_col in df.columns and pd.notna(p["K_TMP"]):
        frost_score = 1.0 - np.clip((p["K_TMP"] - df[T_min_col].values) / 10.0, 0, 0.7)
    else:
        frost_score = np.ones(len(df))
    # Latitude : décroissance hors range
    if pd.notna(p["Lat_max"]):
        lat_abs = df["latitude"].abs().values
        lat_score = np.where(lat_abs <= p["Lat_max"], 1.0,
                              np.clip(1 - (lat_abs - p["Lat_max"]) / 30.0, 0.3, 1.0))
    else:
        lat_score = np.ones(len(df))
    suitability = T_score * P_score * frost_score * lat_score
    df[f"suit_{crop}"] = suitability  # nouveau nom plus court

print(f"   + {len(CROP_SCIENTIFIC)} colonnes suit_* (suitability adoucie)")


# ── 3. Indices de stress climatique ────────────────────────────────────────
print("\n[3] Indices de stress climatique dérivés…")

# Heat stress : T > 30°C en moyenne annuelle = très chaud
df["heat_stress_index"] = np.clip(df[T_col] - 25, 0, 15) / 15  # 0-1

# Cold stress : T_min < 0 = frost risk
if T_min_col in df.columns:
    df["frost_risk_index"] = np.clip(-df[T_min_col], 0, 30) / 30

# Aridité Thornthwaite simplifié (P / PET annuel)
# PET = 16 * (10 * T_mean / 12) ** 0.5 * jours
T_safe = np.where(df[T_col] > 0, df[T_col], 0.1)
df["pet_annual"] = 16 * (10 * T_safe / 12) ** 1.514 * 30
df["aridity_index_calc"] = np.where(df["pet_annual"] > 0,
                                     df[P_col] / df["pet_annual"], 0)

# Continentalité : amplitude T (max - min) — pays continentaux = élevage extensif
if "nasa_t2m_max" in df.columns and "nasa_t2m_min" in df.columns:
    df["continentality"] = df["nasa_t2m_max"] - df["nasa_t2m_min"]

# Growing season index : T entre 10-30°C × P > 100mm
T_gs = np.clip((df[T_col] - 5) / 20, 0, 1) * np.clip((35 - df[T_col]) / 20, 0, 1)
P_gs = np.clip(df[P_col] / 1000, 0.1, 1.5)
df["growing_season_index_v2"] = T_gs * P_gs

# Aride/Humide labels
df["climate_zone_arid"] = (df["aridity_index_calc"] < 0.5).astype(int)
df["climate_zone_humid"] = (df["aridity_index_calc"] > 1.0).astype(int)

new_idx_cols = ["heat_stress_index","frost_risk_index","pet_annual","aridity_index_calc",
                 "continentality","growing_season_index_v2","climate_zone_arid","climate_zone_humid"]
print(f"   + {len(new_idx_cols)} colonnes indices")


# ── 4. Sauvegarde ──────────────────────────────────────────────────────────
out = f"{D}/dataset_final_v12_couche1.csv"
df.to_csv(out, index=False)
print(f"\n[OK] {out}")
print(f"   shape : {df.shape}")
print(f"   +{df.shape[1] - 694} vs v11")

# Stats suitability adoucie
print("\n  Quelques suitability adoucies (mean) :")
for c in ["suit_wheat","suit_rice","suit_maize","suit_tomato","suit_apple","suit_banana","suit_potato","suit_coconut"]:
    if c in df.columns:
        print(f"    {c:25s} mean={df[c].mean():.3f}  std={df[c].std():.3f}")
