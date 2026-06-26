"""
compute_suitability.py — Score de suitability climatique par (pays, année, culture).

Source : FAO EcoCrop Database (via GitHub OpenCLIM/ecocrop)
  - TOPMN/TOPMX : température optimale min/max
  - TMIN/TMAX  : température absolue tolérée
  - ROPMN/ROPMX : précipitation annuelle optimale
  - KTMP       : killing temperature (frost)

Pour chaque culture, on calcule un score 0-1 par (pays, année) :
  - T_score = courbe en cloche autour de T_opt
  - P_score = courbe en cloche autour de P_opt
  - Frost_score = 0 si Tmin < KTMP (frost-intolerant crop)
  - Lat_score = compatibilité latitude

suitability = T_score × P_score × Frost_score × Lat_score
"""
import os, sys, io, urllib.request
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = "data/cleaned"

# ── 1. Télécharger EcoCrop ────────────────────────────────────────────────
ECOCROP_LOCAL = f"{D}/EcoCrop_DB.csv"
if not os.path.exists(ECOCROP_LOCAL):
    print("[1] Téléchargement EcoCrop…")
    url = "https://raw.githubusercontent.com/OpenCLIM/ecocrop/main/EcoCrop_DB.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        with open(ECOCROP_LOCAL, "wb") as f:
            f.write(r.read())
    print(f"   OK {ECOCROP_LOCAL}")
else:
    print(f"[1] EcoCrop déjà téléchargé : {ECOCROP_LOCAL}")

eco = pd.read_csv(ECOCROP_LOCAL, low_memory=False, encoding="latin-1")
print(f"   {len(eco)} espèces, {eco.shape[1]} colonnes")

# ── 2. Mapping cultures → noms scientifiques EcoCrop ──────────────────────
# Trouvé en cherchant ScientificName ou COMNAME dans EcoCrop
CROP_SCIENTIFIC = {
    "wheat":      ("Triticum aestivum",   None),
    "rice":       ("Oryza sativa",        None),
    "maize":      ("Zea mays",            None),
    "soybeans":   ("Glycine max",         None),
    "rapeseed":   ("Brassica napus",      None),
    "sunflower":  ("Helianthus annuus",   None),
    "groundnut":  ("Arachis hypogaea",    None),
    "olives":     ("Olea europaea",       None),
    "sesame":     ("Sesamum indicum",     None),
    "coconut":    ("Cocos nucifera",      None),
    "cotton":     ("Gossypium hirsutum",  None),
    "apple":      ("Malus domestica",     ["Malus pumila"]),
    "banana":     ("Musa acuminata",      ["Musa x paradisiaca","Musa sapientum"]),
    "orange":     ("Citrus sinensis",     None),
    "grape":      ("Vitis vinifera",      None),
    "strawberry": ("Fragaria x ananassa", ["Fragaria vesca"]),
    "pineapple":  ("Ananas comosus",      None),
    "mango":      ("Mangifera indica",    None),
    "avocado":    ("Persea americana",    None),
    "lemon":      ("Citrus limon",        None),
    "peach":      ("Prunus persica",      None),
    "pear":       ("Pyrus communis",      None),
    "watermelon": ("Citrullus lanatus",   None),
    "dates":      ("Phoenix dactylifera", None),
    "apricot":    ("Prunus armeniaca",    None),
    "cherry":     ("Prunus avium",        None),
    "plum":       ("Prunus domestica",    None),
    "tomato":     ("Solanum lycopersicum", ["Lycopersicon esculentum"]),
    "potato":     ("Solanum tuberosum",   None),
    "onion":      ("Allium cepa",         None),
    "cabbage":    ("Brassica oleracea",   None),
    "carrot":     ("Daucus carota",       None),
    "cucumber":   ("Cucumis sativus",     None),
    "eggplant":   ("Solanum melongena",   None),
    "cauliflower":("Brassica oleracea var. botrytis", ["Brassica oleracea"]),
    "lettuce":    ("Lactuca sativa",      None),
    "chickpea":   ("Cicer arietinum",     None),
    "drybean":    ("Phaseolus vulgaris",  None),
    "drypea":     ("Pisum sativum",       None),
}

def find_in_ecocrop(scientific, synonyms):
    for name in [scientific] + (synonyms or []):
        rows = eco[eco["ScientificName"].str.contains(name, case=False, na=False, regex=False)]
        if not rows.empty:
            return rows.iloc[0]
    return None

print("\n[2] Mapping cultures → EcoCrop…")
crop_params = {}
for crop, (sci, syn) in CROP_SCIENTIFIC.items():
    row = find_in_ecocrop(sci, syn)
    if row is not None:
        params = {
            "T_opt_min": pd.to_numeric(row.get("TOPMN"), errors="coerce"),
            "T_opt_max": pd.to_numeric(row.get("TOPMX"), errors="coerce"),
            "T_abs_min": pd.to_numeric(row.get("TMIN"),  errors="coerce"),
            "T_abs_max": pd.to_numeric(row.get("TMAX"),  errors="coerce"),
            "P_opt_min": pd.to_numeric(row.get("ROPMN"), errors="coerce"),
            "P_opt_max": pd.to_numeric(row.get("ROPMX"), errors="coerce"),
            "P_abs_min": pd.to_numeric(row.get("RMIN"),  errors="coerce"),
            "P_abs_max": pd.to_numeric(row.get("RMAX"),  errors="coerce"),
            "K_TMP":     pd.to_numeric(row.get("KTMP"),  errors="coerce"),
            "Lat_min":   pd.to_numeric(row.get("LATMN"), errors="coerce"),
            "Lat_max":   pd.to_numeric(row.get("LATMX"), errors="coerce"),
            "Alt_max":   pd.to_numeric(row.get("ALTMX"), errors="coerce"),
        }
        # Détection issues : si NaN sur valeurs critiques, garde par défaut
        crop_params[crop] = params
        print(f"  + {crop:14s} ({sci}) — T_opt={params['T_opt_min']:.0f}-{params['T_opt_max']:.0f}°C  "
              f"P_opt={params['P_opt_min']:.0f}-{params['P_opt_max']:.0f}mm")
    else:
        print(f"  ✗ {crop:14s} non trouvé ({sci})")

# ── 3. Calculer suitability score pour le dataset ──────────────────────────
print(f"\n[3] Calcul suitability sur dataset V10…")
df = pd.read_csv(f"{D}/dataset_final_v10_couche1.csv", low_memory=False)
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
print(f"   shape : {df.shape}")

# Features climatiques utilisées :
# - nasa_t2m (T annuelle réelle, °C)
# - nasa_prectotcorr (précip annuelle réelle, mm)
# - nasa_t2m_min (T min, °C)
# - latitude (abs)
# - elevation

def trapezoid_score(x, opt_min, opt_max, abs_min, abs_max):
    """Score 0-1 : 1 entre opt_min et opt_max, décroît linéairement vers abs_min/max."""
    x = np.asarray(x, dtype=float)
    score = np.zeros_like(x)
    if pd.isna(opt_min) or pd.isna(opt_max):
        return None
    if pd.isna(abs_min): abs_min = opt_min - (opt_max - opt_min)
    if pd.isna(abs_max): abs_max = opt_max + (opt_max - opt_min)
    # zone optimale
    mask_opt = (x >= opt_min) & (x <= opt_max)
    score[mask_opt] = 1.0
    # zone décroissance basse
    mask_low = (x >= abs_min) & (x < opt_min)
    score[mask_low] = (x[mask_low] - abs_min) / max(opt_min - abs_min, 0.001)
    # zone décroissance haute
    mask_hi = (x > opt_max) & (x <= abs_max)
    score[mask_hi] = (abs_max - x[mask_hi]) / max(abs_max - opt_max, 0.001)
    return np.clip(score, 0, 1)


T_col = "nasa_t2m" if "nasa_t2m" in df.columns else "T_annual_C"
P_col = "nasa_prectotcorr" if "nasa_prectotcorr" in df.columns else "P_annual_mm"
T_min_col = "nasa_t2m_min" if "nasa_t2m_min" in df.columns else None

print(f"   T col = {T_col}, P col = {P_col}, T_min col = {T_min_col}")

added_cols = []
for crop, p in crop_params.items():
    # Score T
    T_score = trapezoid_score(df[T_col].values, p["T_opt_min"], p["T_opt_max"],
                              p["T_abs_min"], p["T_abs_max"])
    # Score P (précip annuelle)
    P_score = trapezoid_score(df[P_col].values, p["P_opt_min"], p["P_opt_max"],
                              p["P_abs_min"], p["P_abs_max"])
    # Frost penalty
    if T_min_col and pd.notna(p["K_TMP"]):
        frost_mask = df[T_min_col].values < p["K_TMP"]
        frost_score = np.where(frost_mask, 0.0, 1.0)
    else:
        frost_score = np.ones(len(df))
    # Latitude
    if pd.notna(p["Lat_min"]) and pd.notna(p["Lat_max"]):
        lat_abs = df["latitude"].abs().values
        lat_score = np.where((lat_abs >= p["Lat_min"]) & (lat_abs <= p["Lat_max"]), 1.0, 0.3)
    else:
        lat_score = np.ones(len(df))
    # Altitude max
    if pd.notna(p["Alt_max"]):
        alt_score = np.where(df["elevation"].values <= p["Alt_max"], 1.0, 0.3)
    else:
        alt_score = np.ones(len(df))

    # Score combiné
    if T_score is None: T_score = np.ones(len(df))
    if P_score is None: P_score = np.ones(len(df))
    suitability = T_score * P_score * frost_score * lat_score * alt_score

    col_name = f"suitability_{crop}"
    df[col_name] = suitability
    added_cols.append(col_name)

print(f"\n   + {len(added_cols)} colonnes suitability ajoutées")

# Stats : suitability moyenne par culture
print("\n[4] Suitability moyenne par culture :")
for c in added_cols[:10]:
    print(f"   {c:30s} mean={df[c].mean():.3f}  median={df[c].median():.3f}")

# ── 5. Sauvegarder ───────────────────────────────────────────────────────
out = f"{D}/dataset_final_v11_couche1.csv"
df.to_csv(out, index=False)
print(f"\n[OK] {out}")
print(f"     +{df.shape[1] - 655} colonnes vs v10")
