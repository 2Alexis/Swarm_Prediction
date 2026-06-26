"""cascade_config_v3.py — Config V3 boostée avec nouveaux datasets."""

SUBLAYER_ORDER = ["SL1_atmosphere","SL1_hydrologie","SL1_sols_ecologie",
                  "SL1_agriculture","SL1_elevage","SL1_peche",
                  "SL1_energie","SL1_geologie"]

SUBLAYERS = {
    # ════════════════════════════════════════════════════════════════════
    "SL1_atmosphere": {
        "label": "1.1 Atmosphère",
        "description": "Climat dynamique + émissions par secteur + CMM",
        "targets": {
            "target_thermal_anomaly":     "Anomalie thermique",
            "target_co2_emissions":       "Émissions CO2 (log)",
            "target_methane_emissions":   "Émissions CH4 (log)",
            "target_n2o_emissions":       "Émissions N2O (log)",
        },
        "feature_groups": [
            "bio","temp_","precip_","nasa_t2m","nasa_prec","nasa_rh2m","nasa_ps",
            "nasa_ws10m","nasa_allsky","nasa_gwet","be_t_anom","be_t_baseline",
            "enso_","nao","amo","pdo","soi","ao_","co2_ppm_global",
            "latitude","longitude","elevation","cluster",
            "heat_stress","frost_risk","aridity","continentality",
            "growing_season","pet_annual",
            # NOUVEAUX V13
            "edgar_fossil_co2_","edgar_ch4_","edgar_n2o_","edgar_f-gases_",
            "cmm_ch4_kt","cmm_satellite_pct_favorable",
        ],
    },

    # ════════════════════════════════════════════════════════════════════
    "SL1_hydrologie": {
        "label": "1.2 Hydrologie",
        "description": "Eau + niveau mer + WRI Aqueduct + AQUASTAT",
        "targets": {
            "target_water_stress":        "Stress hydrique",
            "target_water_access":        "Accès eau potable",
            "target_soil_moisture_root":  "Humidité sol racinaire",
        },
        "feature_groups": [
            "dist_to_coast","dist_to_river","dist_to_lake","dist_to_freshwater",
            "nasa_gwet","nasa_prectotcorr","nasa_rh2m",
            "precip_mean","precip_seasonality",
            "bio12","bio13","bio14","bio15","bio16","bio17","bio18","bio19",
            "P_annual","T_annual","groundwater","tide_amplitude",
            "Population","Urban_pct","cluster","EcoClass",
            # NOUVEAUX V13
            "aqueduct_","aquastat_","aqua_bulk_","sea_level_",
        ],
        "use_cascade_from": ["SL1_atmosphere"],
    },

    # ════════════════════════════════════════════════════════════════════
    "SL1_sols_ecologie": {
        "label": "1.3 Sols & Écologie",
        "description": "Sol + forêt + biodiversité IUCN + EPI Yale + tectonique",
        "targets": {
            "target_soil_degradation":    "Dégradation du sol",
            "target_forest_share":        "% forêt national",
            "target_tree_cover_loss":     "Perte couvert arboré (log)",
            "target_biodiversity_species":"Diversité espèces IUCN (log)",
        },
        "feature_groups": [
            "clay_pct","silt_pct","sand_pct","soil_pH","organic_carbon",
            "Bilan_sols","Engrais","Pesticides",
            "Terres_agricoles","Terres_arables","Part_terres","Part_irriguee",
            "elevation","slope","roughness","latitude","longitude","cluster",
            "feature_npp","feature_fauna","be_t_anom_vs_preindustrial",
            "earthquake","volcan",
            # NOUVEAUX V13
            "epi_","iucn_observations","tectonic_plates_count","volcanoes_count",
        ],
        "use_cascade_from": ["SL1_atmosphere","SL1_hydrologie"],
    },

    # ════════════════════════════════════════════════════════════════════
    "SL1_agriculture": {
        "label": "1.4 Agriculture végétale",
        "description": "Rendements + SPAM 46 cultures + FAO inputs N/P/K + Machinery",
        "targets": {
            "target_yield_cereals":       "Rendement céréales",
            "target_yield_roots":         "Rendement racines/tubercules",
            "target_yield_tomato":        "Tomate",
            "target_yield_potato":        "Pomme de terre",
            "target_yield_cucumber":      "Concombre",
            "target_yield_rapeseed":      "Colza",
            "target_yield_drypea":        "Pois sec",
            "target_yield_apple":         "Pomme",
            "target_yield_eggplant":      "Aubergine",
            "target_yield_strawberry":    "Fraise",
            "target_yield_cotton":        "Coton",
            "target_yield_groundnut":     "Arachide",
            "target_yield_drybean":       "Haricot sec",
        },
        "feature_groups": [
            "Engrais","Pesticides","Terres_agricoles","Terres_arables",
            "Part_irriguee","Part_bio","Part_terres","Irrigation",
            "suit_","growing_season",
            "share_muslim","share_christian","share_hindu","share_buddhist",
            "cluster","latitude","feature_photoperiod","be_t_anom",
            # NOUVEAUX V13
            "spam_yield_","spam_harvest_","fertilizer_N_","fertilizer_P_","fertilizer_K_",
            "pest_","landuse_","machinery_",
        ],
        "use_cascade_from": ["SL1_atmosphere","SL1_hydrologie","SL1_sols_ecologie"],
    },

    # ════════════════════════════════════════════════════════════════════
    "SL1_elevage": {
        "label": "1.5 Élevage",
        "description": "Production animale + GLW4 + WAHIS maladies",
        "targets": {
            "target_milk_yield":          "Rendement lait (kg/animal)",
            "target_cattle_carcass":      "Poids carcasse bovine",
            "target_chicken_carcass":     "Poids carcasse poulet",
            "target_sheepgoat_carcass":   "Poids carcasse ovin/caprin",
            "target_pig_carcass":         "Poids carcasse porc",
            "target_eggs_yield":          "Rendement œufs",
            "target_livestock_eggs_prod": "Production œufs (log)",
            "target_animal_disease_outbreaks": "Outbreaks WAHIS (log)",
            "target_cattle_density":      "Densité bétail GLW4 (log)",
        },
        "feature_groups": [
            "share_muslim","share_christian","share_hindu","share_buddhist",
            "share_folk","share_unaffiliated",
            "meat_","milk_consumption","egg_consumption",
            "agri_value_pct","cluster","latitude","longitude","Population",
            # NOUVEAUX V13
            "wahis_","glw4_",
        ],
        "use_cascade_from": ["SL1_atmosphere","SL1_hydrologie","SL1_sols_ecologie",
                              "SL1_agriculture"],
    },

    # ════════════════════════════════════════════════════════════════════
    "SL1_peche": {
        "label": "1.6 Pêche",
        "description": "Production halieutique FAO 2024",
        "targets": {
            "target_fish_total":          "Production halieutique totale (log)",
            "target_fish_capture":        "Pêche capture sauvage (log)",
            "target_fish_aquaculture":    "Aquaculture totale (log)",
        },
        "feature_groups": [
            "dist_to_coast","nasa_t2m","P_annual","Population","cluster",
            "latitude","longitude","EcoClass",
            # NOUVEAUX V13
            "fish_isscaap_","aqueduct_",
        ],
        "use_cascade_from": ["SL1_atmosphere","SL1_hydrologie","SL1_sols_ecologie"],
    },

    # ════════════════════════════════════════════════════════════════════
    "SL1_energie": {
        "label": "1.7 Énergie",
        "description": "Production énergétique + IRENA + Ember + Power Plants",
        "targets": {
            "target_solar_consumption":   "Solaire TWh (log)",
            "target_wind_generation":     "Éolien TWh (log)",
            "target_hydro_consumption":   "Hydroélectrique TWh (log)",
            "target_coal_production":     "Charbon TWh (log)",
            "target_oil_production":      "Pétrole TWh (log)",
            "target_gas_production":      "Gaz TWh (log)",
            "target_powerplant_capacity_mw": "Capacité PP installée (log)",
        },
        "feature_groups": [
            "Population","GDP_pc","GDP_total","Urban_pct","Electricity_pct",
            "dist_to_fossil","dist_to_mineral","dist_to_energy_source","dist_to_coast",
            "elevation","wind_speed_mean","solar_radiation_mean",
            "latitude","longitude","cluster","area_km2",
            "coal_rents_pct","oil_rents_pct","natgas_rents",
            # NOUVEAUX V13
            "ember_","powerplant_","irena_target_","reshare_","heatgen_",
            "wb_renewable_","wb_elec_access_pct",
            # géologie source
            "mrds_","oil_gas_","coal_n_mines","iron_n_mines","coal_capacity",
        ],
        "use_cascade_from": ["SL1_atmosphere","SL1_hydrologie","SL1_sols_ecologie",
                              "SL1_agriculture","SL1_elevage"],
    },

    # ════════════════════════════════════════════════════════════════════
    "SL1_geologie": {
        "label": "1.8 Géologie",
        "description": "Activité sismique, ressources minières",
        "targets": {
            "target_seismic_activity":    "Activité sismique (log eq count)",
        },
        "feature_groups": [
            "latitude","longitude","elevation","slope","cluster",
            "tectonic_plates_count","volcanoes_count","earthquake_count",
            "eq_count","eq_max_mag","eq_mean_mag","eq_mag_ge6",
            "mrds_","oil_gas_","coal_n_mines","iron_n_mines",
        ],
        # Pas de cascade — la géologie est indépendante
        "use_cascade_from": [],
    },
}


# ── ANTI-LEAK ─────────────────────────────────────────────────────────────
TARGET_SOURCE = {
    "target_thermal_anomaly":     "T_anomaly",
    "target_co2_emissions":       "co2_annual_t",
    "target_methane_emissions":   "methane_total_co2eq",
    "target_n2o_emissions":       "n2o_total_co2eq",
    "target_water_stress":        "Water_Withdrawal_pct",
    "target_water_access":        "water_access_pct",
    "target_soil_moisture_root":  "nasa_gwetroot",
    "target_fish_total":          "fish_total_t",
    "target_fish_capture":        "fish_Capture_t",
    "target_fish_aquaculture":    "fish_aquaculture_total_t",
    "target_soil_degradation":    "Bilan_sols_kgha",
    "target_forest_share":        "forest_share_pct",
    "target_tree_cover_loss":     "tree_cover_loss_ha",
    "target_yield_cereals":       "yield_cereals_kgha",
    "target_yield_roots":         "yield_roots_kgha",
    "target_yield_tomato":        "yield_tomato",
    "target_yield_potato":        "yield_potato",
    "target_yield_cucumber":      "yield_cucumber",
    "target_yield_rapeseed":      "yield_rapeseed",
    "target_yield_drypea":        "yield_drypea",
    "target_yield_apple":         "yield_apple",
    "target_yield_eggplant":      "yield_eggplant",
    "target_yield_strawberry":    "yield_strawberry",
    "target_yield_cotton":        "yield_cotton",
    "target_yield_groundnut":     "yield_groundnut",
    "target_yield_drybean":       "yield_drybean",
    "target_milk_yield":          "livestock_milk_yield",
    "target_cattle_carcass":      "livestock_cattle_carcass_kg",
    "target_chicken_carcass":     "livestock_chicken_carcass_g",
    "target_sheepgoat_carcass":   "livestock_sheepgoat_carcass_kg",
    "target_pig_carcass":         "livestock_pig_carcass_kg",
    "target_eggs_yield":          "livestock_eggs_yield",
    "target_livestock_eggs_prod": "livestock_eggs_t",
    "target_solar_consumption":   "solar_consumption_twh",
    "target_wind_generation":     "wind_generation_twh",
    "target_hydro_consumption":   "hydro_consumption_twh",
    "target_coal_production":     "coal_production_twh",
    "target_oil_production":      "oil_production_twh",
    "target_gas_production":      "gas_production_twh",
    "target_biodiversity_species":"iucn_species_count",
    "target_animal_disease_outbreaks": "wahis_outbreaks_total",
    "target_cattle_density":      "glw4_cattle_total",
    "target_powerplant_capacity_mw": "powerplant_total_mw",
    "target_seismic_activity":    "eq_count",
}

EXTRA_LEAKS = {
    "target_thermal_anomaly":   ["owid_temp_anomaly","be_t_anom_annual",
                                  "be_t_anom_vs_preindustrial","be_t_anom_lag1",
                                  "be_t_anom_lag3","be_t_anom_lag5"],
    "target_co2_emissions":     ["methane_total_co2eq","n2o_total_co2eq","owid_co2_pc",
                                  "co2_per_capita_calc","cmm_ch4_kt"],
    "target_methane_emissions": ["co2_annual_t","n2o_total_co2eq","owid_methane","cmm_ch4_kt"],
    "target_n2o_emissions":     ["co2_annual_t","methane_total_co2eq","owid_n2o"],
    "target_water_stress":      ["freshwater_withdraw_total","freshwater_internal_per_cap",
                                  "owid_water_stress","water_withdraw_total_km3",
                                  "aqueduct_bws_"],
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
    "target_cattle_carcass":      ["livestock_cattle_slaughtered","meat_beef_kg_pc","glw4_"],
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
    "target_biodiversity_species":["iucn_observations"],
    "target_animal_disease_outbreaks":["wahis_diseases_unique","wahis_cases","wahis_deaths"],
    "target_cattle_density":      ["glw4_cattle_mean","glw4_cattle_max"],
    "target_powerplant_capacity_mw":["powerplant_n_plants","powerplant_coal_mw","powerplant_gas_mw",
                                       "powerplant_oil_mw","powerplant_hydro_mw","powerplant_solar_mw",
                                       "powerplant_wind_mw"],
    "target_seismic_activity":    ["eq_max_mag","eq_mean_mag","eq_mag_ge6",
                                     "earthquake_count","earthquake_max_mag"],
}
