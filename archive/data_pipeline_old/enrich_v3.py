"""
enrich_v3.py — Enrichit dataset_final_v3.csv pour produire dataset_final_v4.csv

  - 14 nouvelles variables BIO climatiques (saisonnalité, growing season)
  - 39 indicateurs World Bank supplémentaires
  - 5 datasets OWID supplémentaires
  - Features dérivées (densités, ratios, intensités)
  - Lags sur les variables-clés ajoutées
"""
import os, sys, io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = "data/cleaned"

from build_dataset import custom_mappings, get_english_iso, get_iso_from_alpha3

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

# ── 1. Charger v3 ──────────────────────────────────────────────────────────
print("[1] Chargement dataset_final_v3.csv…")
df = pd.read_csv(f"{D}/dataset_final_v3.csv")
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
print(f"   shape de départ : {df.shape}")

# ── 2. BIO 2-19 (statiques par ISO) ────────────────────────────────────────
print("[2] Ajout des 14 BIO WorldClim saisonniers…")
bio = pd.read_csv(f"{D}/worldclim_bio_extra.csv")
df = df.merge(bio, on="ISO", how="left")
print(f"   +{bio.shape[1]-1} colonnes BIO")

# ── 3. Nouveaux indicateurs WB ─────────────────────────────────────────────
print("[3] Ajout des 39 nouveaux indicateurs WB…")
WB_FILES = [f for f in os.listdir(D) if f.startswith("wb_") and not f in (
    # ne pas ré-ajouter ceux déjà en v3
    "wb_gdp_per_capita.csv","wb_gdp_current_usd.csv","wb_life_expectancy.csv",
    "wb_child_mortality.csv","wb_birth_rate.csv","wb_death_rate.csv",
    "wb_population_total.csv","wb_population_growth.csv","wb_net_migration.csv",
    "wb_urban_population_pct.csv","wb_agricultural_land_pct.csv","wb_inflation_cpi.csv",
    "wb_unemployment_rate.csv","wb_gini_index.csv","wb_poverty_rate_190.csv",
    "wb_public_debt_gdp.csv","wb_trade_gdp.csv","wb_renewable_energy_pct.csv",
    "wb_electricity_access_pct.csv","wb_energy_use_per_capita.csv","wb_internet_users_pct.csv",
    "wb_mobile_subscriptions.csv","wb_health_expenditure_gdp.csv","wb_hospital_beds_per_1000.csv",
    "wb_rd_expenditure_gdp.csv","wb_malaria_incidence.csv","wb_hiv_prevalence.csv",
    "wb_deaths_communicable_disease.csv","wb_freshwater_withdrawal_pct.csv"
)]
added = 0
for f in WB_FILES:
    name = f.replace("wb_", "").replace(".csv", "")
    try:
        sub = pd.read_csv(f"{D}/{f}")
        sub["ISO"] = to_iso(sub["Pays"])
        sub = sub.dropna(subset=["ISO"])
        sub["Annee"] = pd.to_numeric(sub["Annee"], errors="coerce")
        sub = sub.dropna(subset=["Annee"])
        sub["Annee"] = sub["Annee"].astype(int)
        sub["Valeur"] = pd.to_numeric(sub["Valeur"], errors="coerce")
        sub = sub.dropna(subset=["Valeur"])
        sub = sub.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(
            columns={"Valeur": name})
        df = df.merge(sub, on=["ISO", "Annee"], how="left")
        added += 1
    except Exception as e:
        print(f"   ✗ {f}: {e}")
print(f"   +{added} indicateurs WB")

# ── 4. Nouveaux OWID ──────────────────────────────────────────────────────
print("[4] Ajout des OWID extras…")
OWID_NEW = [
    ("owid_schooling_years.csv",       "schooling_years"),
    ("owid_meat_consumption.csv",      "meat_consumption_pc"),
    ("owid_co_emissions.csv",          "co_emissions"),
    ("owid_cereal_production.csv",     "cereal_production_t"),
    ("owid_extreme_poverty.csv",       "extreme_poverty_pct"),
]
for f, name in OWID_NEW:
    p = f"{D}/{f}"
    if not os.path.exists(p): continue
    try:
        sub = pd.read_csv(p)
        sub["ISO"] = to_iso(sub["Pays"])
        sub = sub.dropna(subset=["ISO"])
        sub["Annee"] = pd.to_numeric(sub["Annee"], errors="coerce")
        sub = sub.dropna(subset=["Annee"])
        sub["Annee"] = sub["Annee"].astype(int)
        sub["Valeur"] = pd.to_numeric(sub["Valeur"], errors="coerce")
        sub = sub.dropna(subset=["Valeur"])
        sub = sub.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(
            columns={"Valeur": name})
        df = df.merge(sub, on=["ISO", "Annee"], how="left")
        print(f"   + {name}")
    except Exception as e:
        print(f"   ✗ {f}: {e}")

# ── 5. Features dérivées (densités, ratios, intensités) ────────────────────
print("[5] Features dérivées…")
def safe_div(a, b, default=np.nan):
    return np.where((b > 0) & b.notna() & a.notna(), a / b, default)

# Densité de population
if "Population" in df.columns and "area_km2" in df.columns:
    df["pop_density_per_km2"] = safe_div(df["Population"], df["area_km2"])

# PIB par hectare arable
if "GDP_total_usd" in df.columns and "Terres_arables_ha" in df.columns:
    df["gdp_per_arable_ha"] = safe_div(df["GDP_total_usd"], df["Terres_arables_ha"])

# Production électrique par habitant
if "elec_generation_gwh" in df.columns and "Population" in df.columns:
    df["elec_gen_per_capita"] = safe_div(df["elec_generation_gwh"] * 1e6, df["Population"])

# Population urbaine absolue
if "Urban_pct" in df.columns and "Population" in df.columns:
    df["urban_pop_abs"] = df["Urban_pct"] * df["Population"] / 100.0

# Conflits par million d'habitants
if "conflict_count" in df.columns and "Population" in df.columns:
    df["conflict_per_million"] = safe_div(df["conflict_count"] * 1e6, df["Population"])

# Catastrophes par million
if "disaster_deaths" in df.columns and "Population" in df.columns:
    df["disaster_deaths_per_million"] = safe_div(df["disaster_deaths"] * 1e6, df["Population"])
    df["disaster_affected_per_capita"] = safe_div(df["disaster_affected"], df["Population"])

# Ratio terres agricoles / superficie totale (déjà partiellement Part_terres_agricoles)
if "Terres_agricoles_ha" in df.columns and "area_km2" in df.columns:
    df["agri_density"] = safe_div(df["Terres_agricoles_ha"], df["area_km2"] * 100)  # *100 pour km²→ha

# Diversité des cultures (combien de familles avec yield non-null)
yield_cols = [c for c in df.columns if c.startswith("yield_") and c.endswith("_kgha")]
if yield_cols:
    df["crop_diversity"] = df[yield_cols].notna().sum(axis=1)

# CO2 par habitant (si pas déjà existant)
if "CO2_emissions" in df.columns and "Population" in df.columns:
    df["co2_per_capita_calc"] = safe_div(df["CO2_emissions"] * 1e6, df["Population"])

# Ratio renouvelables / total
if "elec_renew_share" in df.columns:
    df["elec_fossil_share"] = 1.0 - df["elec_renew_share"]

# Income proxy : ratio rural employment
if "employ_agri_pct" in df.columns and "employ_services_pct" in df.columns:
    df["agri_vs_services_ratio"] = safe_div(df["employ_agri_pct"], df["employ_services_pct"])

# Education combined
if "school_primary_enrollment" in df.columns and "school_secondary_enrollment" in df.columns:
    df["school_total_enrollment"] = df[["school_primary_enrollment", "school_secondary_enrollment"]].mean(axis=1)

# Stunting + Wasting combined (malnutrition compound)
if "stunting_pct" in df.columns and "wasting_pct" in df.columns:
    df["malnutrition_compound"] = df["stunting_pct"] + df["wasting_pct"]

# Index de richesse en ressources (oil + mineral + gas + coal rents)
rent_cols = [c for c in df.columns if c.endswith("_rents_pct_gdp")]
if rent_cols:
    df["resource_rents_total"] = df[rent_cols].sum(axis=1, min_count=1)

# Index hydrique (rapport withdraw / internal)
if "freshwater_withdraw_total" in df.columns and "freshwater_internal_per_cap" in df.columns:
    df["water_stress_ratio_raw"] = safe_div(df["freshwater_withdraw_total"],
                                            df["freshwater_internal_per_cap"] * df["Population"] / 1e9)

# Saisonalité hydrique : BIO16 / BIO17 (wettest vs driest quarter ratio)
if "bio16_precip_wettest_quarter" in df.columns and "bio17_precip_driest_quarter" in df.columns:
    df["precip_seasonal_amplitude"] = safe_div(
        df["bio16_precip_wettest_quarter"], df["bio17_precip_driest_quarter"].clip(lower=1))

# Growing season precip-temp interaction
if "bio18_precip_warmest_quarter" in df.columns and "bio10_temp_warmest_quarter" in df.columns:
    df["growing_season_index"] = df["bio18_precip_warmest_quarter"] * np.exp(-(df["bio10_temp_warmest_quarter"] - 22)**2 / 100)

# Drought stress index (BIO14 = driest month precip — plus c'est faible, plus c'est sec)
if "bio14_precip_driest_month" in df.columns:
    df["drought_stress_index"] = np.exp(-df["bio14_precip_driest_month"] / 50.0)

print(f"   ajout features dérivées OK")

# ── 6. Lags sur les nouvelles variables-clés ───────────────────────────────
print("[6] Lags sur variables-clés ajoutées…")
df = df.sort_values(["ISO", "Annee"])
key_lag_vars = ["pm25_annual", "agri_value_pct_gdp", "stunting_pct", "wasting_pct",
                "pop_65_plus_pct", "pop_under14_pct", "dependency_ratio",
                "urban_growth", "schooling_years", "meat_consumption_pc",
                "employ_agri_pct", "co_emissions", "cereal_production_t",
                "pop_density_per_km2", "gdp_per_arable_ha", "urban_pop_abs",
                "elec_gen_per_capita", "growing_season_index", "drought_stress_index"]
for v in key_lag_vars:
    if v in df.columns:
        for k in (1, 3, 5):
            df[f"{v}_lag{k}"] = df.groupby("ISO")[v].shift(k)
        df[f"{v}_roll5"] = df.groupby("ISO")[v].transform(lambda s: s.rolling(5, min_periods=2).mean())

# ── 7. Sauvegarde ─────────────────────────────────────────────────────────
out = f"{D}/dataset_final_v4.csv"
df.to_csv(out, index=False)
print(f"\n[OK] {out}")
print(f"     shape = {df.shape}")
print(f"     +{df.shape[1] - 215} colonnes vs v3 (était 215)")

# Récap dynamic
num = df.select_dtypes(include="number").columns.tolist()
static = []
dynamic = []
for c in num:
    s = df.groupby("ISO")[c].std().mean()
    if pd.notna(s) and s < 0.01:
        static.append(c)
    else:
        dynamic.append(c)
print(f"     DYNAMIC: {len(dynamic)}/{len(num)}")
