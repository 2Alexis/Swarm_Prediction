"""
build_v13.py — V12 → V13 : merge TOUS les nouveaux datasets organisés par catégorie.

Ajoute dans le dataset principal :
  ATMOSPHÈRE  : edgar 2025 par secteur (CO2, CH4, N2O, F-gases) + CMM emissions + CMM satellite
  HYDROLOGIE  : aqueduct 40 indicateurs + aquastat top30 + aquastat bulk + psmsl sea level
  SOL/ÉCOL    : epi 2024 (31 indicateurs) + iucn 245 pays
  AGRICULTURE : spam2020_v2 yield (46 cultures) + spam harvested + fao N/P/K + fao pesticides + fao landuse + fao machinery
  ÉLEVAGE     : wahis disease + glw4 cattle density
  PÊCHE       : fish_production_total + by_source + by_isscaap
  ÉNERGIE     : ember 16 sources + IRENA reshare + heatgen + r-eleccap + r-elecgen + global_power_plants + irena targets
  GÉOLOGIE    : mrds, oil_gas, coal, iron, steel, earthquakes, volcans, tectonic plates
  ÉCONOMIE    : fao value of production

Crée aussi de NOUVELLES cibles :
  - target_biodiversity_species (IUCN)
  - target_sea_level_rise (PSMSL)
  - target_animal_disease_outbreaks (WAHIS)
  - target_cattle_density (GLW4)
  - target_powerplant_capacity_mw (global power plants)
"""
import os, sys, io, glob
import pandas as pd
import numpy as np
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

CLN = "data/cleaned"

print("[1] Chargement v12 base…")
df = pd.read_csv(f"{CLN}/shared/dataset_final_v12_couche1.csv", low_memory=False)
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
if "cluster" in df.columns:
    df["cluster"] = df["cluster"].astype(int)
print(f"   départ : {df.shape}")


def merge_file(df, path, key_cols=("ISO","Annee"), how="left"):
    """Merge un CSV cleaned dans df si possible."""
    if not os.path.exists(path):
        return df, 0
    try:
        sub = pd.read_csv(path, low_memory=False)
        # Si pas d'Annee, c'est statique → merge sur ISO seul puis broadcast
        if "Annee" not in sub.columns:
            keys = ["ISO"]
        else:
            keys = list(key_cols)
        # Skip si ISO absent
        if "ISO" not in sub.columns:
            return df, 0
        # Drop colonnes déjà présentes (sauf clés)
        new_cols = [c for c in sub.columns if c in keys or c not in df.columns]
        sub = sub[new_cols]
        df = df.merge(sub, on=keys, how=how)
        return df, sub.shape[1] - len(keys)
    except Exception as e:
        print(f"   ⚠️ {path}: {str(e)[:100]}")
        return df, 0


# ── 2. ATMOSPHÈRE ────────────────────────────────────────────────────────
print("\n[2] 🌫️ Atmosphère…")
total_added = 0
for f in ["edgar_2025_fossil_co2.csv", "edgar_2025_ch4.csv", "edgar_2025_n2o.csv",
          "edgar_2025_f-gases.csv", "cmm_emissions_ch4.csv", "cmm_satellite_monitoring.csv"]:
    df, n = merge_file(df, f"{CLN}/atmosphere/{f}")
    if n > 0:
        print(f"   + {f}: {n} cols")
        total_added += n


# ── 3. HYDROLOGIE ────────────────────────────────────────────────────────
print("\n[3] 💧 Hydrologie…")
for f in ["wri_aqueduct40_by_country.csv", "aquastat_top30_variables.csv",
          "aquastat_bulk_eng.csv", "psmsl_sea_level_by_country.csv"]:
    df, n = merge_file(df, f"{CLN}/hydrologie/{f}")
    if n > 0:
        print(f"   + {f}: {n} cols")
        total_added += n


# ── 4. SOL / ÉCOLOGIE ────────────────────────────────────────────────────
print("\n[4] 🌿 Sol/Écologie…")
for f in ["epi_2024_indicators.csv", "iucn_species_by_country.csv"]:
    df, n = merge_file(df, f"{CLN}/sol_ecologie/{f}")
    if n > 0:
        print(f"   + {f}: {n} cols")
        total_added += n


# ── 5. AGRICULTURE ────────────────────────────────────────────────────────
print("\n[5] 🌾 Agriculture…")
for f in ["spam2020_v2_yield_by_country.csv", "spam2020_v2_harvested_area_by_country.csv",
          "fao_fertilizers_NPK.csv", "fao_pesticides_categories.csv",
          "fao_landuse_inputs.csv", "fao_machinery.csv"]:
    df, n = merge_file(df, f"{CLN}/agriculture/{f}")
    if n > 0:
        print(f"   + {f}: {n} cols")
        total_added += n


# ── 6. ÉLEVAGE ────────────────────────────────────────────────────────────
print("\n[6] 🐄 Élevage…")
for f in ["wahis_animal_disease.csv", "glw4_cattle_density.csv"]:
    df, n = merge_file(df, f"{CLN}/elevage/{f}")
    if n > 0:
        print(f"   + {f}: {n} cols")
        total_added += n


# ── 7. PÊCHE ─────────────────────────────────────────────────────────────
print("\n[7] 🐟 Pêche…")
for f in ["fish_production_total.csv", "fish_production_by_source.csv",
          "fish_production_by_isscaap.csv"]:
    df, n = merge_file(df, f"{CLN}/peche/{f}")
    if n > 0:
        print(f"   + {f}: {n} cols")
        total_added += n


# ── 8. ÉNERGIE ────────────────────────────────────────────────────────────
print("\n[8] ⚡ Énergie…")
for f in ["ember_electricity_by_source.csv", "global_power_plants_by_country.csv",
          "irena_renewable_targets.csv", "reshare_by_country.csv",
          "heatgen_by_country.csv", "r-eleccap_by_country.csv", "r-elecgen_by_country.csv",
          "ember_monthly_capacity_yearend.csv",
          "wb_renewable_elec_pct.csv", "wb_renewable_final_pct.csv", "wb_elec_access_pct.csv"]:
    df, n = merge_file(df, f"{CLN}/energie/{f}")
    if n > 0:
        print(f"   + {f}: {n} cols")
        total_added += n


# ── 9. GÉOLOGIE ───────────────────────────────────────────────────────────
print("\n[9] ⛏️ Géologie…")
for f in ["mrds_minerals_by_country.csv", "oil_gas_extraction_by_country.csv",
          "global_coal_mines_by_country.csv", "global_iron_mines_by_country.csv",
          "steel_industry_raw.csv", "earthquakes_by_country_year.csv",
          "volcanoes_by_country.csv", "tectonic_plates_by_country.csv"]:
    df, n = merge_file(df, f"{CLN}/geologie/{f}")
    if n > 0:
        print(f"   + {f}: {n} cols")
        total_added += n


# ── 10. ÉCONOMIE ─────────────────────────────────────────────────────────
print("\n[10] 💵 Économie…")
for f in ["fao_value_of_production.csv"]:
    df, n = merge_file(df, f"{CLN}/economie/{f}")
    if n > 0:
        print(f"   + {f}: {n} cols")
        total_added += n


# ── 11. NOUVELLES CIBLES ─────────────────────────────────────────────────
print("\n[11] 🎯 Création nouvelles cibles…")

# Biodiversité (IUCN)
if "iucn_species_count" in df.columns:
    df["target_biodiversity_species"] = np.log1p(df["iucn_species_count"].clip(lower=0))
    print(f"   + target_biodiversity_species : {df['target_biodiversity_species'].notna().sum()} obs")

# Niveau marin (PSMSL)
if "sea_level_mean" in df.columns:
    df["target_sea_level_rise"] = df["sea_level_mean"]
    print(f"   + target_sea_level_rise : {df['target_sea_level_rise'].notna().sum()} obs")

# Maladies animales (WAHIS)
if "wahis_outbreaks_total" in df.columns:
    df["target_animal_disease_outbreaks"] = np.log1p(df["wahis_outbreaks_total"].clip(lower=0))
    print(f"   + target_animal_disease_outbreaks : {df['target_animal_disease_outbreaks'].notna().sum()} obs")

# Densité bétail (GLW4)
if "glw4_cattle_total" in df.columns:
    df["target_cattle_density"] = np.log1p(df["glw4_cattle_total"].clip(lower=0))
    print(f"   + target_cattle_density : {df['target_cattle_density'].notna().sum()} obs")

# Capacité électrique installée (Global Power Plants)
if "powerplant_total_mw" in df.columns:
    df["target_powerplant_capacity_mw"] = np.log1p(df["powerplant_total_mw"].clip(lower=0))
    print(f"   + target_powerplant_capacity_mw : {df['target_powerplant_capacity_mw'].notna().sum()} obs")

# Activité sismique
if "eq_count" in df.columns:
    df["target_seismic_activity"] = np.log1p(df["eq_count"].clip(lower=0))
    print(f"   + target_seismic_activity : {df['target_seismic_activity'].notna().sum()} obs")


# ── 12. Imputation honnête pour les nouvelles features (pas les cibles) ──
print("\n[12] Imputation forward-fill par pays sur nouvelles features…")
v12_cols = set(pd.read_csv(f"{CLN}/shared/dataset_final_v12_couche1.csv", nrows=0).columns)
new_cols_features = [c for c in df.columns
                       if c not in v12_cols
                       and not c.startswith("target_")
                       and df[c].dtype in ("float64","int64")]
print(f"   {len(new_cols_features)} nouvelles colonnes features à imputer")
df = df.sort_values(["ISO","Annee"]).reset_index(drop=True)
for c in new_cols_features:
    df[c] = df.groupby("ISO")[c].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both", limit=5))
    df[c] = df.groupby("ISO")[c].transform(lambda s: s.ffill().bfill())
    if "cluster" in df.columns:
        df[c] = df.groupby(["cluster","Annee"])[c].transform(lambda s: s.fillna(s.median()))
    df[c] = df.groupby("Annee")[c].transform(lambda s: s.fillna(s.median()))


# ── 13. Sauvegarde ──────────────────────────────────────────────────────
out = f"{CLN}/shared/dataset_final_v13_couche1.csv"
df.to_csv(out, index=False)
print(f"\n[OK] {out}")
print(f"   Shape v13 : {df.shape}")
print(f"   +{df.shape[1] - 741} colonnes vs v12")
print(f"   Total cibles : {sum(1 for c in df.columns if c.startswith('target_'))}")
