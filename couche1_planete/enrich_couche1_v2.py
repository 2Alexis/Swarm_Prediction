"""
enrich_couche1_v2.py — V9 → V10 :
  - Cultures spécifiques en tant que CIBLES PRIMAIRES (à la place des agrégats faibles)
  - OWID meat consumption per type (5 colonnes : beef, pig, poultry, sheep_goat, other)
  - Pew religion 2010 (statique par pays, 7 colonnes)
  - Réorganisation Couche 1
"""
import os, sys, io
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from build_dataset import custom_mappings, get_english_iso
from couche1_planete.religion_static import build_religion_df

D = "data/cleaned"

def to_iso(series):
    out = []
    for p in series:
        if pd.isna(p): out.append(None); continue
        s = str(p).strip().lower()
        code = custom_mappings.get(s)
        if not code:
            code = get_english_iso(p)
        out.append(code)
    return out

print("[1] Chargement v9 Couche 1…")
df = pd.read_csv(f"{D}/dataset_final_v9_couche1.csv", low_memory=False)
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
print(f"   départ : {df.shape}")


# ── 2. Religion statique (Pew 2010) ────────────────────────────────────────
print("\n[2] Religion Pew 2010…")
rel = build_religion_df()
df = df.merge(rel, on="ISO", how="left")
# Compléter avec valeurs médianes pour pays manquants
for col in rel.columns:
    if col == "ISO": continue
    df[col] = df[col].fillna(df[col].median())
print(f"   + 7 colonnes religion")


# ── 3. Meat consumption par type (OWID) ────────────────────────────────────
print("\n[3] Meat consumption par type…")
mp = f"{D}/owid_meat_by_type.csv"
if os.path.exists(mp):
    meat = pd.read_csv(mp)
    # Renommer les colonnes (raccourcir noms)
    rename = {}
    for c in meat.columns:
        cl = c.lower()
        if "beef" in cl: rename[c] = "meat_beef_kg_pc"
        elif "pig" in cl: rename[c] = "meat_pig_kg_pc"
        elif "poultry" in cl: rename[c] = "meat_poultry_kg_pc"
        elif "sheep" in cl: rename[c] = "meat_sheepgoat_kg_pc"
        elif "other" in cl and "meat" in cl: rename[c] = "meat_other_kg_pc"
    meat = meat.rename(columns=rename)
    meat["ISO"] = to_iso(meat["Pays"])
    meat = meat.dropna(subset=["ISO"])
    meat["Annee"] = pd.to_numeric(meat["Annee"], errors="coerce").astype("Int64")
    meat = meat.dropna(subset=["Annee"])
    meat["Annee"] = meat["Annee"].astype(int)
    cols_meat = [c for c in meat.columns if c.startswith("meat_")]
    meat = meat.groupby(["ISO", "Annee"], as_index=False)[cols_meat].mean()
    df = df.merge(meat, on=["ISO", "Annee"], how="left")
    print(f"   + {len(cols_meat)} colonnes meat per type")


# ── 4. Autres datasets meat/dairy ──────────────────────────────────────────
print("\n[4] Lait/œufs/poultry consumption (OWID)…")
for f, name in [
    ("owid_milk_consumption.csv",      "milk_consumption_kg_pc"),
    ("owid_egg_consumption.csv",       "egg_consumption_kg_pc"),
    ("owid_meat_production_t.csv",     "meat_total_production_t"),
]:
    p = f"{D}/{f}"
    if not os.path.exists(p): continue
    sub = pd.read_csv(p)
    if "Pays" not in sub.columns: continue
    sub["ISO"] = to_iso(sub["Pays"])
    sub = sub.dropna(subset=["ISO"])
    sub["Annee"] = pd.to_numeric(sub["Annee"], errors="coerce").astype("Int64").dropna().astype(int)
    val_col = "Valeur" if "Valeur" in sub.columns else [c for c in sub.columns if c not in ("Pays","Annee","ISO")][0]
    sub[name] = pd.to_numeric(sub[val_col], errors="coerce")
    sub = sub.dropna(subset=[name])
    sub = sub.groupby(["ISO", "Annee"], as_index=False)[name].mean()
    df = df.merge(sub, on=["ISO", "Annee"], how="left")
    print(f"   + {name}")


# ── 5. Cultures spécifiques en tant que cibles primaires ──────────────────
print("\n[5] Cultures spécifiques → cibles primaires (remplace agrégats)…")
# Les colonnes yield_{crop} existent déjà dans le dataset (depuis v6)
# On crée des cibles correspondantes
SPECIFIC_CROP_TARGETS = {
    # OLÉAGINEUX
    "yield_soybeans":    "target_yield_soybeans",
    "yield_rapeseed":    "target_yield_rapeseed",
    "yield_sunflower":   "target_yield_sunflower",
    "yield_groundnut":   "target_yield_groundnut",
    "yield_olives":      "target_yield_olives",
    "yield_sesame":      "target_yield_sesame",
    "yield_coconut":     "target_yield_coconut",
    "yield_cotton":      "target_yield_cotton",
    # FRUITS
    "yield_apple":       "target_yield_apple",
    "yield_banana":      "target_yield_banana",
    "yield_orange":      "target_yield_orange",
    "yield_grape":       "target_yield_grape",
    "yield_strawberry":  "target_yield_strawberry",
    "yield_pineapple":   "target_yield_pineapple",
    "yield_mango":       "target_yield_mango",
    "yield_avocado":     "target_yield_avocado",
    "yield_lemon":       "target_yield_lemon",
    "yield_peach":       "target_yield_peach",
    "yield_pear":        "target_yield_pear",
    "yield_watermelon":  "target_yield_watermelon",
    "yield_dates":       "target_yield_dates",
    "yield_apricot":     "target_yield_apricot",
    "yield_cherry":      "target_yield_cherry",
    "yield_plum":        "target_yield_plum",
    # LÉGUMES
    "yield_tomato":      "target_yield_tomato",
    "yield_potato":      "target_yield_potato",
    "yield_onion":       "target_yield_onion",
    "yield_cabbage":     "target_yield_cabbage",
    "yield_carrot":      "target_yield_carrot",
    "yield_cucumber":    "target_yield_cucumber",
    "yield_eggplant":    "target_yield_eggplant",
    "yield_cauliflower": "target_yield_cauliflower",
    "yield_lettuce":     "target_yield_lettuce",
    # LÉGUMINEUSES
    "yield_chickpea":    "target_yield_chickpea",
    "yield_drybean":     "target_yield_drybean",
    "yield_drypea":      "target_yield_drypea",
}
for src, tgt in SPECIFIC_CROP_TARGETS.items():
    if src in df.columns and tgt not in df.columns:
        df[tgt] = np.log1p(df[src].clip(lower=0))
n_new_crops = sum(1 for s, t in SPECIFIC_CROP_TARGETS.items()
                  if s in df.columns and t in df.columns and df[t].notna().sum() > 100)
print(f"   + {n_new_crops} cultures spécifiques comme cibles")


# ── 6. Imputation honnête features ajoutées ───────────────────────────────
print("\n[6] Imputation features ajoutées…")
v9_cols = set(pd.read_csv(f"{D}/dataset_final_v9_couche1.csv", nrows=0).columns)
new_cols = [c for c in df.columns if c not in v9_cols and not c.startswith("target_")]
print(f"   {len(new_cols)} nouvelles features à imputer")

df = df.sort_values(["ISO", "Annee"]).reset_index(drop=True)
for c in new_cols:
    if df[c].dtype not in ("float64", "int64"): continue
    df[c] = df.groupby("ISO")[c].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both", limit=5))
    df[c] = df.groupby("ISO")[c].transform(lambda s: s.ffill().bfill())
    df[c] = df.groupby(["cluster", "Annee"])[c].transform(lambda s: s.fillna(s.median()))
    df[c] = df.groupby("Annee")[c].transform(lambda s: s.fillna(s.median()))
    df[c] = df[c].fillna(df[c].median())


# ── 7. Sauvegarde ─────────────────────────────────────────────────────────
out = f"{D}/dataset_final_v10_couche1.csv"
df.to_csv(out, index=False)
print(f"\n[OK] {out}")
print(f"   shape : {df.shape}  (+{df.shape[1] - 640} cols vs v9)")
new_targets = [t for t in SPECIFIC_CROP_TARGETS.values() if t in df.columns]
print(f"   {len(new_targets)} nouvelles cibles culture spécifique")
print(f"   Top 10 par couverture :")
for t in sorted(new_targets, key=lambda c: -df[c].notna().sum())[:10]:
    print(f"     {t:35s} {df[t].notna().sum():6,d} obs")
