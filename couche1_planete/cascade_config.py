"""
cascade_config.py — Architecture en cascade de la Couche 1.

Pipeline de modèles en cascade : chaque sous-couche prédit des cibles physiques,
et ses prédictions deviennent des FEATURES d'entrée pour la sous-couche suivante.

  SL1.1 Atmosphère
       ↓ (prédictions thermiques, émissions)
  SL1.2 Hydrologie
       ↓ (prédictions eau, humidité, pêche)
  SL1.3 Sols & Écologie
       ↓ (prédictions sol, forêt)
  SL1.4 Agriculture végétale
       ↓ (prédictions rendements)
  SL1.5 Élevage
       ↓ (prédictions production animale)
  SL1.6 Énergie

Chaque sous-couche utilise UNIQUEMENT :
  - les features brutes du dataset
  - + les prédictions OOF des sous-couches précédentes (cascade)
"""

# ── ORDRE DE LA CASCADE ────────────────────────────────────────────────────
SUBLAYER_ORDER = ["SL1_atmosphere", "SL1_hydrologie", "SL1_sols_ecologie",
                  "SL1_agriculture", "SL1_elevage", "SL1_energie"]


# ── CONFIG PAR SOUS-COUCHE ────────────────────────────────────────────────
SUBLAYERS = {
    # ════════════════════════════════════════════════════════════════════
    "SL1_atmosphere": {
        "label": "1.1 Atmosphère",
        "description": "Climat dynamique, émissions, qualité air",
        "targets": {
            "target_thermal_anomaly":    "Anomalie thermique",
            "target_co2_emissions":      "Émissions CO2 (log)",
            "target_methane_emissions":  "Émissions CH4 (log)",
            "target_n2o_emissions":      "Émissions N2O (log)",
        },
        # Features autorisées (régex / préfixes) — pas de cibles dispatchées
        "feature_groups": [
            "bio",                    # WorldClim BIO 1-19
            "temp_", "precip_",       # climatologies
            "nasa_t2m", "nasa_prec", "nasa_rh2m", "nasa_ps", "nasa_ws10m",
            "nasa_allsky", "be_t_anom", "be_t_baseline",
            "enso_", "nao", "amo", "pdo", "soi", "ao_lag", "ao_",
            "co2_ppm_global",         # CO2 mondial seulement
            "latitude", "longitude", "elevation", "cluster",
            "heat_stress", "frost_risk", "aridity", "continentality",
            "growing_season", "pet_annual",
        ],
    },

    # ════════════════════════════════════════════════════════════════════
    "SL1_hydrologie": {
        "label": "1.2 Hydrologie",
        "description": "Eau, humidité sol, pêche",
        "targets": {
            "target_water_stress":       "Stress hydrique",
            "target_water_access":       "Accès eau potable",
            "target_soil_moisture_root": "Humidité sol racinaire",
            "target_fish_total":         "Production halieutique totale (log)",
            "target_fish_capture":       "Pêche capture sauvage (log)",
            "target_fish_aquaculture":   "Aquaculture totale (log)",
        },
        "feature_groups": [
            "dist_to_coast", "dist_to_river", "dist_to_lake", "dist_to_freshwater",
            "nasa_gwet", "nasa_prectotcorr", "nasa_rh2m",
            "precip_mean", "precip_seasonality",
            "bio12", "bio13", "bio14", "bio15", "bio16", "bio17", "bio18", "bio19",
            "P_annual", "T_annual",
            "groundwater", "tide_amplitude",
            "Population", "Urban_pct", "cluster",
            "EcoClass",
        ],
        # On accepte aussi les OOF de SL1.1
        "use_cascade_from": ["SL1_atmosphere"],
    },

    # ════════════════════════════════════════════════════════════════════
    "SL1_sols_ecologie": {
        "label": "1.3 Sols & Écologie",
        "description": "Dégradation sol, forêt, couvert arboré",
        "targets": {
            "target_soil_degradation":   "Dégradation du sol",
            "target_forest_share":       "% forêt national",
            "target_tree_cover_loss":    "Perte couvert arboré (log)",
        },
        "feature_groups": [
            "clay_pct", "silt_pct", "sand_pct", "soil_pH", "organic_carbon",
            "Bilan_sols", "Engrais", "Pesticides",
            "Terres_agricoles", "Terres_arables", "Part_terres", "Part_irriguee",
            "elevation", "slope", "roughness", "latitude", "longitude", "cluster",
            "feature_npp", "feature_fauna",
            "be_t_anom_vs_preindustrial",
            "earthquake", "volcan",
        ],
        "use_cascade_from": ["SL1_atmosphere", "SL1_hydrologie"],
    },

    # ════════════════════════════════════════════════════════════════════
    "SL1_agriculture": {
        "label": "1.4 Agriculture végétale",
        "description": "Rendements par culture",
        "targets": {
            # 6 familles
            "target_yield_cereals":      "Rendement céréales",
            "target_yield_roots":        "Rendement racines/tubercules",
            # Top cultures spécifiques (R² > 0.3 en V12)
            "target_yield_tomato":       "Tomate",
            "target_yield_potato":       "Pomme de terre",
            "target_yield_cucumber":     "Concombre",
            "target_yield_rapeseed":     "Colza",
            "target_yield_drypea":       "Pois sec",
            "target_yield_apple":        "Pomme",
            "target_yield_eggplant":     "Aubergine",
            "target_yield_strawberry":   "Fraise",
            "target_yield_cotton":       "Coton",
            "target_yield_groundnut":    "Arachide",
            "target_yield_drybean":      "Haricot sec",
        },
        "feature_groups": [
            "Engrais", "Pesticides", "Terres_agricoles", "Terres_arables",
            "Part_irriguee", "Part_bio", "Part_terres",
            "Irrigation",
            "suit_",                  # Toutes suitabilities EcoCrop
            "growing_season",
            "share_muslim", "share_christian", "share_hindu", "share_buddhist",
            "cluster", "latitude",
            "feature_photoperiod",
            "be_t_anom",
        ],
        "use_cascade_from": ["SL1_atmosphere", "SL1_hydrologie", "SL1_sols_ecologie"],
    },

    # ════════════════════════════════════════════════════════════════════
    "SL1_elevage": {
        "label": "1.5 Élevage",
        "description": "Production animale (lait, viandes, œufs)",
        "targets": {
            "target_milk_yield":          "Rendement lait (kg/animal)",
            "target_cattle_carcass":      "Poids carcasse bovine",
            "target_chicken_carcass":     "Poids carcasse poulet",
            "target_sheepgoat_carcass":   "Poids carcasse ovin/caprin",
            "target_pig_carcass":         "Poids carcasse porc",
            "target_eggs_yield":          "Rendement œufs",
            "target_livestock_eggs_prod": "Production œufs (log)",
        },
        "feature_groups": [
            "share_muslim", "share_christian", "share_hindu", "share_buddhist",
            "share_folk", "share_unaffiliated",
            "meat_",                  # Consommation viande (sans leak car distinct par type)
            "milk_consumption",
            "egg_consumption",
            "agri_value_pct",
            "cluster", "latitude", "longitude",
            "Population",
        ],
        "use_cascade_from": ["SL1_atmosphere", "SL1_hydrologie", "SL1_sols_ecologie",
                              "SL1_agriculture"],
    },

    # ════════════════════════════════════════════════════════════════════
    "SL1_energie": {
        "label": "1.6 Énergie",
        "description": "Production énergétique (renouvelable + fossile)",
        "targets": {
            "target_solar_consumption":  "Solaire TWh (log)",
            "target_wind_generation":    "Éolien TWh (log)",
            "target_hydro_consumption":  "Hydroélectrique TWh (log)",
            "target_coal_production":    "Charbon TWh (log)",
            "target_oil_production":     "Pétrole TWh (log)",
            "target_gas_production":     "Gaz TWh (log)",
        },
        "feature_groups": [
            "Population", "GDP_pc", "GDP_total",
            "Urban_pct", "Electricity_pct",
            "dist_to_fossil", "dist_to_mineral", "dist_to_energy_source",
            "dist_to_coast",
            "elevation", "wind_speed_mean", "solar_radiation_mean",
            "latitude", "longitude", "cluster", "area_km2",
            "coal_rents_pct", "oil_rents_pct", "natgas_rents",
        ],
        "use_cascade_from": ["SL1_atmosphere", "SL1_hydrologie", "SL1_sols_ecologie",
                              "SL1_agriculture", "SL1_elevage"],
    },
}


# ── ANTI-LEAK SOURCES BRUTES (à dropper par cible) ────────────────────────
TARGET_SOURCE = {
    "target_thermal_anomaly":    "T_anomaly",
    "target_co2_emissions":      "co2_annual_t",
    "target_methane_emissions":  "methane_total_co2eq",
    "target_n2o_emissions":      "n2o_total_co2eq",
    "target_water_stress":       "Water_Withdrawal_pct",
    "target_water_access":       "water_access_pct",
    "target_soil_moisture_root": "nasa_gwetroot",
    "target_fish_total":         "fish_total_t",
    "target_fish_capture":       "fish_Capture_t",
    "target_fish_aquaculture":   "fish_aquaculture_total_t",
    "target_soil_degradation":   "Bilan_sols_kgha",
    "target_forest_share":       "forest_share_pct",
    "target_tree_cover_loss":    "tree_cover_loss_ha",
    "target_yield_cereals":      "yield_cereals_kgha",
    "target_yield_roots":        "yield_roots_kgha",
    "target_yield_tomato":       "yield_tomato",
    "target_yield_potato":       "yield_potato",
    "target_yield_cucumber":     "yield_cucumber",
    "target_yield_rapeseed":     "yield_rapeseed",
    "target_yield_drypea":       "yield_drypea",
    "target_yield_apple":        "yield_apple",
    "target_yield_eggplant":     "yield_eggplant",
    "target_yield_strawberry":   "yield_strawberry",
    "target_yield_cotton":       "yield_cotton",
    "target_yield_groundnut":    "yield_groundnut",
    "target_yield_drybean":      "yield_drybean",
    "target_milk_yield":         "livestock_milk_yield",
    "target_cattle_carcass":     "livestock_cattle_carcass_kg",
    "target_chicken_carcass":    "livestock_chicken_carcass_g",
    "target_sheepgoat_carcass":  "livestock_sheepgoat_carcass_kg",
    "target_pig_carcass":        "livestock_pig_carcass_kg",
    "target_eggs_yield":         "livestock_eggs_yield",
    "target_livestock_eggs_prod":"livestock_eggs_t",
    "target_solar_consumption":  "solar_consumption_twh",
    "target_wind_generation":    "wind_generation_twh",
    "target_hydro_consumption":  "hydro_consumption_twh",
    "target_coal_production":    "coal_production_twh",
    "target_oil_production":     "oil_production_twh",
    "target_gas_production":     "gas_production_twh",
}

# Leaks supplémentaires (variables qui = cible avec autre nom)
EXTRA_LEAKS = {
    "target_thermal_anomaly":   ["owid_temp_anomaly", "be_t_anom_annual",
                                  "be_t_anom_vs_preindustrial", "be_t_anom_lag1",
                                  "be_t_anom_lag3","be_t_anom_lag5"],
    "target_co2_emissions":     ["methane_total_co2eq","n2o_total_co2eq","owid_co2_pc","co2_per_capita_calc"],
    "target_methane_emissions": ["co2_annual_t","n2o_total_co2eq","owid_methane"],
    "target_n2o_emissions":     ["co2_annual_t","methane_total_co2eq","owid_n2o"],
    "target_water_stress":      ["freshwater_withdraw_total","freshwater_internal_per_cap",
                                  "owid_water_stress","water_withdraw_total_km3"],
    "target_water_access":      ["safe_water_pct","sanitation_pct"],
    "target_soil_moisture_root":["nasa_gwettop","nasa_gwetprof"],
    "target_fish_total":        ["fish_Capture_t","fish_Aquaculture_Marine_t",
                                  "fish_Aquaculture_Freshwater_t","fish_Aquaculture_Brackish_t",
                                  "fish_species_count","fish_isscaap_count","aquaculture_t",
                                  "fish_aquaculture_total_t"],
    "target_fish_capture":      ["fish_total_t","fish_Aquaculture_Marine_t",
                                  "fish_Aquaculture_Freshwater_t","fish_Aquaculture_Brackish_t",
                                  "fish_aquaculture_total_t"],
    "target_fish_aquaculture":  ["fish_total_t","fish_Capture_t","aquaculture_t",
                                  "fish_Aquaculture_Marine_t","fish_Aquaculture_Freshwater_t",
                                  "fish_Aquaculture_Brackish_t"],
    "target_forest_share":      ["forest_area_km2","forest_change","tree_cover_loss_ha",
                                  "deforestation_annual","forest_per_capita_km2"],
    "target_tree_cover_loss":   ["forest_change","forest_share_pct","forest_area_km2",
                                  "deforestation_annual"],
    "target_milk_yield":          ["livestock_dairy_animals","milk_consumption_kg_pc"],
    "target_cattle_carcass":      ["livestock_cattle_slaughtered","meat_beef_kg_pc"],
    "target_chicken_carcass":     ["livestock_chicken_slaughtered","meat_poultry_kg_pc"],
    "target_sheepgoat_carcass":   ["livestock_sheepgoat_slaughtered","meat_sheepgoat_kg_pc"],
    "target_pig_carcass":         ["livestock_pig_slaughtered","meat_pig_kg_pc"],
    "target_eggs_yield":          ["livestock_eggs_t","egg_consumption_kg_pc"],
    "target_livestock_eggs_prod": ["livestock_eggs_yield","egg_consumption_kg_pc"],
    "target_solar_consumption":   ["wind_generation_twh","hydro_consumption_twh"],
    "target_wind_generation":     ["solar_consumption_twh","hydro_consumption_twh"],
    "target_hydro_consumption":   ["solar_consumption_twh","wind_generation_twh"],
    "target_coal_production":     ["oil_production_twh","gas_production_twh"],
    "target_oil_production":      ["coal_production_twh","gas_production_twh"],
    "target_gas_production":      ["coal_production_twh","oil_production_twh"],
}
