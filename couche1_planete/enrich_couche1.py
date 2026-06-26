"""
enrich_couche1.py — Enrichit dataset_final_v8_honest.csv pour la Couche 1.

Ajoute :
  - Élevage (Production+Rendement) : viande bovine, viande poulet, lait, œufs
  - Aquaculture FAO/OWID
  - Émissions atmosphériques (CO2 annuel, CH4, N2O par pays)
  - Énergies renouvelables (solar, wind, hydro consumption)
  - Production fossile (coal, oil, gas)
  - Eau détaillée (withdraw, accès)

Sortie : data/cleaned/dataset_final_v9_couche1.csv
"""
import os, sys, io
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from build_dataset import custom_mappings, get_english_iso

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


# ── 1. Charger v8 honnête ──────────────────────────────────────────────────
print("[1] Chargement v8 honnête…")
df = pd.read_csv(f"{D}/dataset_final_v8_honest.csv", low_memory=False)
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
print(f"   départ : {df.shape}")


# ── 2. Production animale FAO (élevage) ────────────────────────────────────
print("\n[2] Élevage FAO…")
anim = pd.read_csv(f"{D}/production_animaux.csv")
anim["Pays_Clean"] = anim["Pays"].str.strip().str.lower()
anim["ISO"] = anim["Pays_Clean"].map(custom_mappings)
anim = anim.dropna(subset=["ISO"])
anim["Valeur"] = pd.to_numeric(anim["Valeur"], errors="coerce")

LIVESTOCK_PRODUCTS = {
    # (Produit, Element) → nom colonne
    # Lait : Rendement/Poids Carcasse = kg/animal, Animaux laitiers = nombre
    ("Lait, total", "Rendement/Poids Carcasse"):              "livestock_milk_yield",
    ("Lait, total", "Animaux laitiers"):                       "livestock_dairy_animals",
    # Viandes : Rendement = kg/carcasse, Animaux abattus = nombre
    ("Viande Bovins et Buffles, primaire", "Rendement/Poids Carcasse"):  "livestock_cattle_carcass_kg",
    ("Viande Bovins et Buffles, primaire", "Animaux Producteurs/Abattus"): "livestock_cattle_slaughtered",
    ("Viande, poulet, fraîche ou réfrigérée", "Rendement/Poids Carcasse"): "livestock_chicken_carcass_g",
    ("Viande, poulet, fraîche ou réfrigérée", "Animaux Producteurs/Abattus"): "livestock_chicken_slaughtered",
    ("Viande Ovins et Caprins", "Rendement/Poids Carcasse"):  "livestock_sheepgoat_carcass_kg",
    ("Viande Ovins et Caprins", "Animaux Producteurs/Abattus"): "livestock_sheepgoat_slaughtered",
    ("Viande, suidés, fraîche ou réfrigérée", "Rendement/Poids Carcasse"): "livestock_pig_carcass_kg",
    ("Viande, suidés, fraîche ou réfrigérée", "Animaux Producteurs/Abattus"): "livestock_pig_slaughtered",
    # Œufs
    ("Œufs de poule en coquille frais", "Production"):        "livestock_eggs_t",
    ("Œufs de poule en coquille frais", "Rendement"):         "livestock_eggs_yield",
    # Cheptel
    ("Bovins", "Réserves"):                                    "livestock_cattle_heads",
    ("Volaille", "Réserves"):                                  "livestock_poultry_heads",
    ("Ovins et caprins", "Réserves"):                          "livestock_sheepgoat_heads",
}

for (prod, elem), col in LIVESTOCK_PRODUCTS.items():
    sub = anim[(anim["Produit"] == prod) & (anim["Element"] == elem)].copy()
    if sub.empty: continue
    sub = sub.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(
        columns={"Valeur": col})
    df = df.merge(sub, on=["ISO", "Annee"], how="left")
    n = sub[col].notna().sum()
    print(f"   + {col:35s} {n:6,d} obs")


# ── 3. Aquaculture & marine ────────────────────────────────────────────────
print("\n[3] Pêche / Aquaculture…")
for f, name in [
    ("owid_aquaculture.csv",       "aquaculture_t"),
    ("owid_marine_protected.csv",  "marine_protected_pct"),
]:
    p = f"{D}/{f}"
    if not os.path.exists(p): continue
    sub = pd.read_csv(p)
    sub["ISO"] = to_iso(sub["Pays"])
    sub = sub.dropna(subset=["ISO"])
    sub["Annee"] = pd.to_numeric(sub["Annee"], errors="coerce").astype("Int64").dropna().astype(int)
    sub["Valeur"] = pd.to_numeric(sub["Valeur"], errors="coerce")
    sub = sub.dropna(subset=["Valeur"])
    sub = sub.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": name})
    df = df.merge(sub, on=["ISO", "Annee"], how="left")
    print(f"   + {name}")


# ── 4. Émissions atmosphériques détaillées ────────────────────────────────
print("\n[4] Émissions atmosphériques…")
for f, name in [
    ("owid_co2_emissions_annual.csv",  "co2_annual_t"),
    ("owid_co_per_capita.csv",         "co_per_capita_alt"),
    ("owid_methane_total.csv",         "methane_total_co2eq"),
    ("owid_n2o_total.csv",             "n2o_total_co2eq"),
]:
    p = f"{D}/{f}"
    if not os.path.exists(p): continue
    sub = pd.read_csv(p)
    sub["ISO"] = to_iso(sub["Pays"])
    sub = sub.dropna(subset=["ISO"])
    sub["Annee"] = pd.to_numeric(sub["Annee"], errors="coerce").astype("Int64").dropna().astype(int)
    sub["Valeur"] = pd.to_numeric(sub["Valeur"], errors="coerce")
    sub = sub.dropna(subset=["Valeur"])
    sub = sub.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": name})
    df = df.merge(sub, on=["ISO", "Annee"], how="left")
    print(f"   + {name}")


# ── 5. Énergies renouvelables production ──────────────────────────────────
print("\n[5] Énergies renouvelables…")
for f, name in [
    ("owid_solar_consumption.csv",   "solar_consumption_twh"),
    ("owid_wind_generation.csv",     "wind_generation_twh"),
    ("owid_hydro_consumption.csv",   "hydro_consumption_twh"),
]:
    p = f"{D}/{f}"
    if not os.path.exists(p): continue
    sub = pd.read_csv(p)
    sub["ISO"] = to_iso(sub["Pays"])
    sub = sub.dropna(subset=["ISO"])
    sub["Annee"] = pd.to_numeric(sub["Annee"], errors="coerce").astype("Int64").dropna().astype(int)
    sub["Valeur"] = pd.to_numeric(sub["Valeur"], errors="coerce")
    sub = sub.dropna(subset=["Valeur"])
    sub = sub.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": name})
    df = df.merge(sub, on=["ISO", "Annee"], how="left")
    print(f"   + {name}")


# ── 6. Production fossile ──────────────────────────────────────────────────
print("\n[6] Production fossile…")
for f, name in [
    ("owid_coal_production.csv",   "coal_production_twh"),
    ("owid_oil_production.csv",    "oil_production_twh"),
    ("owid_gas_production.csv",    "gas_production_twh"),
]:
    p = f"{D}/{f}"
    if not os.path.exists(p): continue
    sub = pd.read_csv(p)
    sub["ISO"] = to_iso(sub["Pays"])
    sub = sub.dropna(subset=["ISO"])
    sub["Annee"] = pd.to_numeric(sub["Annee"], errors="coerce").astype("Int64").dropna().astype(int)
    sub["Valeur"] = pd.to_numeric(sub["Valeur"], errors="coerce")
    sub = sub.dropna(subset=["Valeur"])
    sub = sub.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": name})
    df = df.merge(sub, on=["ISO", "Annee"], how="left")
    print(f"   + {name}")


# ── 7. Eau détaillée ──────────────────────────────────────────────────────
print("\n[7] Eau détaillée…")
for f, name in [
    ("owid_water_withdraw_total.csv", "water_withdraw_total_km3"),
    ("owid_water_withdraw_pc.csv",    "water_withdraw_per_capita"),
    ("owid_water_access.csv",         "water_access_pct"),
    ("owid_pesticide_tonnes.csv",     "pesticide_tonnes_total"),
]:
    p = f"{D}/{f}"
    if not os.path.exists(p): continue
    sub = pd.read_csv(p)
    sub["ISO"] = to_iso(sub["Pays"])
    sub = sub.dropna(subset=["ISO"])
    sub["Annee"] = pd.to_numeric(sub["Annee"], errors="coerce").astype("Int64").dropna().astype(int)
    sub["Valeur"] = pd.to_numeric(sub["Valeur"], errors="coerce")
    sub = sub.dropna(subset=["Valeur"])
    sub = sub.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": name})
    df = df.merge(sub, on=["ISO", "Annee"], how="left")
    print(f"   + {name}")


# ── 8. Nouvelles cibles Couche 1 ──────────────────────────────────────────
print("\n[8] Création des nouvelles cibles Couche 1…")

# Élevage (rendements/poids carcasse + production en log)
livestock_targets = {
    "target_milk_yield":             ("livestock_milk_yield", False),  # kg/animal
    "target_cattle_carcass":         ("livestock_cattle_carcass_kg", False),
    "target_chicken_carcass":        ("livestock_chicken_carcass_g", False),
    "target_sheepgoat_carcass":      ("livestock_sheepgoat_carcass_kg", False),
    "target_pig_carcass":            ("livestock_pig_carcass_kg", False),
    "target_eggs_yield":             ("livestock_eggs_yield", False),
    "target_livestock_eggs_prod":    ("livestock_eggs_t", True),
}
for tgt, (src, do_log) in livestock_targets.items():
    if src in df.columns:
        if do_log:
            df[tgt] = np.log1p(df[src].clip(lower=0))
        else:
            df[tgt] = df[src]
        print(f"   + {tgt}: {df[tgt].notna().sum()} obs")

# Aquaculture
if "aquaculture_t" in df.columns:
    df["target_aquaculture"] = np.log1p(df["aquaculture_t"].clip(lower=0))
    print(f"   + target_aquaculture: {df['target_aquaculture'].notna().sum()} obs")

# Émissions
if "co2_annual_t" in df.columns:
    df["target_co2_emissions"] = np.log1p(df["co2_annual_t"].clip(lower=0))
if "methane_total_co2eq" in df.columns:
    df["target_methane_emissions"] = np.log1p(df["methane_total_co2eq"].clip(lower=0))
if "n2o_total_co2eq" in df.columns:
    df["target_n2o_emissions"] = np.log1p(df["n2o_total_co2eq"].clip(lower=0))

# Énergie renouvelable
if "solar_consumption_twh" in df.columns:
    df["target_solar_consumption"] = np.log1p(df["solar_consumption_twh"].clip(lower=0))
if "wind_generation_twh" in df.columns:
    df["target_wind_generation"] = np.log1p(df["wind_generation_twh"].clip(lower=0))
if "hydro_consumption_twh" in df.columns:
    df["target_hydro_consumption"] = np.log1p(df["hydro_consumption_twh"].clip(lower=0))

# Production fossile
if "coal_production_twh" in df.columns:
    df["target_coal_production"] = np.log1p(df["coal_production_twh"].clip(lower=0))
if "oil_production_twh" in df.columns:
    df["target_oil_production"] = np.log1p(df["oil_production_twh"].clip(lower=0))
if "gas_production_twh" in df.columns:
    df["target_gas_production"] = np.log1p(df["gas_production_twh"].clip(lower=0))

# Eau
if "water_access_pct" in df.columns:
    df["target_water_access"] = df["water_access_pct"]

# Marine protégée
if "marine_protected_pct" in df.columns:
    df["target_marine_protected"] = df["marine_protected_pct"]


# ── 9. Imputation honnête des NEW features (jamais les nouvelles cibles) ──
print("\n[9] Imputation honnête des nouvelles features…")
# Toutes les nouvelles features sont à imputer (sauf les sources des nouvelles cibles)
new_targets = [c for c in df.columns if c.startswith("target_") and c not in
               pd.read_csv(f"{D}/dataset_final_v8_honest.csv", nrows=0).columns]
new_target_sources = set()
for tgt in new_targets:
    src = tgt.replace("target_", "")
    src_candidates = [f"{src}_t", f"{src}_twh", f"{src}_pct", src, f"livestock_{src.replace('livestock_','')}_t"]
    for sc in src_candidates:
        if sc in df.columns:
            new_target_sources.add(sc)

# Identifier les colonnes ajoutées au dataset (pas dans v8)
v8_cols = set(pd.read_csv(f"{D}/dataset_final_v8_honest.csv", nrows=0).columns)
new_cols = [c for c in df.columns if c not in v8_cols and c not in new_targets]
print(f"   {len(new_cols)} nouvelles colonnes features à imputer")

# Pour chaque nouvelle colonne, imputer par pays (interpolation) puis cluster × année
df = df.sort_values(["ISO", "Annee"]).reset_index(drop=True)
for c in new_cols:
    if c in new_target_sources: continue  # ne pas imputer les sources des cibles
    df[c] = df.groupby("ISO")[c].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both", limit=5))
    df[c] = df.groupby("ISO")[c].transform(lambda s: s.ffill().bfill())
    # Cluster × année pour ce qui reste
    df[c] = df.groupby(["cluster", "Annee"])[c].transform(lambda s: s.fillna(s.median()))
    df[c] = df.groupby("Annee")[c].transform(lambda s: s.fillna(s.median()))


# ── 10. Sauvegarde ────────────────────────────────────────────────────────
print("\n[10] Sauvegarde…")
out = f"{D}/dataset_final_v9_couche1.csv"
df.to_csv(out, index=False)
print(f"[OK] {out}")
print(f"   Shape : {df.shape}")
print(f"   +{df.shape[1] - 590} colonnes vs v8 honnête")
print(f"   {len(new_targets)} nouvelles cibles ajoutées :")
for t in sorted(new_targets):
    n = df[t].notna().sum()
    print(f"     {t:40s} {n:6,d} obs ({n/len(df)*100:.0f}%)")
