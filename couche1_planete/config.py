"""
couche1_planete/config.py — Configuration COUCHE 1 V3.

Changements V3 :
  - Remplacement des agrégats faibles (fruits, oléag, légumineuses) par cultures spécifiques
  - Ajout meat per type (beef/pig/poultry/sheep_goat) comme features
  - Ajout religion Pew 2010 (% muslim, christian, hindu, etc.) comme features
  - Total : ~60 cibles dans 7 sous-couches
"""

# ── CIBLES COUCHE 1 V3 ────────────────────────────────────────────────────
TARGETS = {
    # ── 1A. AGRICULTURE — CULTURES SPÉCIFIQUES (au lieu d'agrégats faibles) ──
    # Céréales (gardons l'agrégé qui marche)
    "target_yield_cereals":      "Rendement céréales",

    # Oléagineux — par culture (au lieu de l'agrégat R²=0.09)
    "target_yield_soybeans":     "Soja",
    "target_yield_rapeseed":     "Colza",
    "target_yield_sunflower":    "Tournesol",
    "target_yield_groundnut":    "Arachide",
    "target_yield_olives":       "Olives",
    "target_yield_sesame":       "Sésame",
    "target_yield_coconut":      "Coco",
    "target_yield_cotton":       "Coton",

    # Fruits — par culture (au lieu de l'agrégat R²=0.14)
    "target_yield_apple":        "Pomme",
    "target_yield_banana":       "Banane",
    "target_yield_orange":       "Orange",
    "target_yield_grape":        "Raisin",
    "target_yield_strawberry":   "Fraise",
    "target_yield_pineapple":    "Ananas",
    "target_yield_mango":        "Mangue",
    "target_yield_avocado":      "Avocat",
    "target_yield_lemon":        "Citron",
    "target_yield_peach":        "Pêche",
    "target_yield_pear":         "Poire",
    "target_yield_watermelon":   "Pastèque",
    "target_yield_dates":        "Datte",
    "target_yield_apricot":      "Abricot",
    "target_yield_cherry":       "Cerise",
    "target_yield_plum":         "Prune",

    # Légumes — par culture
    "target_yield_tomato":       "Tomate",
    "target_yield_potato":       "Pomme de terre",
    "target_yield_onion":        "Oignon",
    "target_yield_cabbage":      "Chou",
    "target_yield_carrot":       "Carotte",
    "target_yield_cucumber":     "Concombre",
    "target_yield_eggplant":     "Aubergine",
    "target_yield_cauliflower":  "Chou-fleur",
    "target_yield_lettuce":      "Laitue",

    # Racines (l'agrégé fonctionne)
    "target_yield_roots":        "Rendement racines/tubercules",

    # Légumineuses — par culture (au lieu de l'agrégat R²=-0.08)
    "target_yield_chickpea":     "Pois chiche",
    "target_yield_drybean":      "Haricot sec",
    "target_yield_drypea":       "Pois sec",

    # ── 1B. ÉLEVAGE ──
    "target_milk_yield":           "Rendement lait (kg/animal)",
    "target_cattle_carcass":       "Poids carcasse bovine (kg)",
    "target_chicken_carcass":     "Poids carcasse poulet (g)",
    "target_sheepgoat_carcass":    "Poids carcasse ovin/caprin (kg)",
    "target_pig_carcass":          "Poids carcasse porc (kg)",
    "target_eggs_yield":           "Rendement œufs (kg/poule)",
    "target_livestock_eggs_prod":  "Production œufs totale (log)",
    "target_aquaculture":          "Production aquaculture (log)",

    # ── 1C. ENVIRONNEMENT PHYSIQUE ──
    "target_water_stress":         "Stress hydrique",
    "target_soil_degradation":     "Dégradation du sol",
    "target_thermal_anomaly":      "Anomalie thermique",
    "target_soil_moisture_root":   "Humidité sol racinaire",
    "target_water_access":         "Accès eau potable (%)",

    # ── 1D. ÉCOLOGIE ──
    "target_forest_share":         "% forêt national",
    "target_tree_cover_loss":      "Perte couvert arboré (log)",

    # ── 1E. ÉMISSIONS ATMOSPHÉRIQUES ──
    "target_co2_emissions":        "Émissions CO2 annuelles (log)",
    "target_methane_emissions":    "Émissions CH4 totales (log)",
    "target_n2o_emissions":        "Émissions N2O totales (log)",

    # ── 1F. ÉNERGIE ──
    "target_solar_consumption":    "Consommation solaire TWh (log)",
    "target_wind_generation":      "Génération éolienne TWh (log)",
    "target_hydro_consumption":    "Consommation hydroélectrique TWh (log)",
    "target_coal_production":      "Production charbon TWh (log)",
    "target_oil_production":       "Production pétrole TWh (log)",
    "target_gas_production":       "Production gaz TWh (log)",
}

SUBLAYERS = {
    "Céréales & racines": [
        "target_yield_cereals", "target_yield_roots",
    ],
    "Oléagineux (par culture)": [
        "target_yield_soybeans","target_yield_rapeseed","target_yield_sunflower",
        "target_yield_groundnut","target_yield_olives","target_yield_sesame",
        "target_yield_coconut","target_yield_cotton",
    ],
    "Fruits (par culture)": [
        "target_yield_apple","target_yield_banana","target_yield_orange",
        "target_yield_grape","target_yield_strawberry","target_yield_pineapple",
        "target_yield_mango","target_yield_avocado","target_yield_lemon",
        "target_yield_peach","target_yield_pear","target_yield_watermelon",
        "target_yield_dates","target_yield_apricot","target_yield_cherry","target_yield_plum",
    ],
    "Légumes (par culture)": [
        "target_yield_tomato","target_yield_potato","target_yield_onion",
        "target_yield_cabbage","target_yield_carrot","target_yield_cucumber",
        "target_yield_eggplant","target_yield_cauliflower","target_yield_lettuce",
    ],
    "Légumineuses (par culture)": [
        "target_yield_chickpea","target_yield_drybean","target_yield_drypea",
    ],
    "Élevage": [
        "target_milk_yield","target_cattle_carcass","target_chicken_carcass",
        "target_sheepgoat_carcass","target_pig_carcass",
        "target_eggs_yield","target_livestock_eggs_prod","target_aquaculture",
    ],
    "Environnement physique": [
        "target_water_stress","target_soil_degradation","target_thermal_anomaly",
        "target_soil_moisture_root","target_water_access",
    ],
    "Écologie": [
        "target_forest_share","target_tree_cover_loss",
    ],
    "Émissions atmosphériques": [
        "target_co2_emissions","target_methane_emissions","target_n2o_emissions",
    ],
    "Énergie": [
        "target_solar_consumption","target_wind_generation","target_hydro_consumption",
        "target_coal_production","target_oil_production","target_gas_production",
    ],
}

TARGET_SOURCE = {
    # Céréales/racines
    "target_yield_cereals":      "yield_cereals_kgha",
    "target_yield_roots":        "yield_roots_kgha",
    # Oléagineux
    "target_yield_soybeans":     "yield_soybeans",
    "target_yield_rapeseed":     "yield_rapeseed",
    "target_yield_sunflower":    "yield_sunflower",
    "target_yield_groundnut":    "yield_groundnut",
    "target_yield_olives":       "yield_olives",
    "target_yield_sesame":       "yield_sesame",
    "target_yield_coconut":      "yield_coconut",
    "target_yield_cotton":       "yield_cotton",
    # Fruits
    "target_yield_apple":        "yield_apple",
    "target_yield_banana":       "yield_banana",
    "target_yield_orange":       "yield_orange",
    "target_yield_grape":        "yield_grape",
    "target_yield_strawberry":   "yield_strawberry",
    "target_yield_pineapple":    "yield_pineapple",
    "target_yield_mango":        "yield_mango",
    "target_yield_avocado":      "yield_avocado",
    "target_yield_lemon":        "yield_lemon",
    "target_yield_peach":        "yield_peach",
    "target_yield_pear":         "yield_pear",
    "target_yield_watermelon":   "yield_watermelon",
    "target_yield_dates":        "yield_dates",
    "target_yield_apricot":      "yield_apricot",
    "target_yield_cherry":       "yield_cherry",
    "target_yield_plum":         "yield_plum",
    # Légumes
    "target_yield_tomato":       "yield_tomato",
    "target_yield_potato":       "yield_potato",
    "target_yield_onion":        "yield_onion",
    "target_yield_cabbage":      "yield_cabbage",
    "target_yield_carrot":       "yield_carrot",
    "target_yield_cucumber":     "yield_cucumber",
    "target_yield_eggplant":     "yield_eggplant",
    "target_yield_cauliflower":  "yield_cauliflower",
    "target_yield_lettuce":      "yield_lettuce",
    # Légumineuses
    "target_yield_chickpea":     "yield_chickpea",
    "target_yield_drybean":      "yield_drybean",
    "target_yield_drypea":       "yield_drypea",
    # Élevage
    "target_milk_yield":           "livestock_milk_yield",
    "target_cattle_carcass":       "livestock_cattle_carcass_kg",
    "target_chicken_carcass":      "livestock_chicken_carcass_g",
    "target_sheepgoat_carcass":    "livestock_sheepgoat_carcass_kg",
    "target_pig_carcass":          "livestock_pig_carcass_kg",
    "target_eggs_yield":           "livestock_eggs_yield",
    "target_livestock_eggs_prod":  "livestock_eggs_t",
    "target_aquaculture":          "aquaculture_t",
    # Environnement
    "target_water_stress":         "Water_Withdrawal_pct",
    "target_soil_degradation":     "Bilan_sols_kgha",
    "target_thermal_anomaly":      "T_anomaly",
    "target_soil_moisture_root":   "nasa_gwetroot",
    "target_water_access":         "water_access_pct",
    # Écologie
    "target_forest_share":         "forest_share_pct",
    "target_tree_cover_loss":      "tree_cover_loss_ha",
    # Émissions
    "target_co2_emissions":        "co2_annual_t",
    "target_methane_emissions":    "methane_total_co2eq",
    "target_n2o_emissions":        "n2o_total_co2eq",
    # Énergie
    "target_solar_consumption":    "solar_consumption_twh",
    "target_wind_generation":      "wind_generation_twh",
    "target_hydro_consumption":    "hydro_consumption_twh",
    "target_coal_production":      "coal_production_twh",
    "target_oil_production":       "oil_production_twh",
    "target_gas_production":       "gas_production_twh",
}

# Leaks par cible : drop tous les autres yield_* pour cibles agro
EXTRA_LEAKS = {
    "target_yield_cereals": ["cereal_yield","cereal_production_t","food_production_index","owid_cereal_production"],
    "target_yield_roots":   ["food_production_index"],
    # Élevage
    "target_milk_yield":      ["livestock_dairy_animals","milk_consumption_kg_pc"],
    "target_cattle_carcass":  ["livestock_cattle_slaughtered","livestock_cattle_heads","meat_beef_kg_pc"],
    "target_chicken_carcass": ["livestock_chicken_slaughtered","livestock_poultry_heads","meat_poultry_kg_pc"],
    "target_sheepgoat_carcass":["livestock_sheepgoat_slaughtered","livestock_sheepgoat_heads","meat_sheepgoat_kg_pc"],
    "target_pig_carcass":     ["livestock_pig_slaughtered","meat_pig_kg_pc"],
    "target_eggs_yield":      ["livestock_eggs_t","egg_consumption_kg_pc"],
    "target_livestock_eggs_prod": ["livestock_eggs_yield","egg_consumption_kg_pc"],
    "target_aquaculture":     ["marine_protected_pct"],
    # Env
    "target_water_stress":    ["freshwater_withdraw_total","freshwater_internal_per_cap",
                               "water_stress_ratio_raw","owid_water_stress",
                               "water_withdraw_total_km3","water_withdraw_per_capita"],
    "target_soil_moisture_root": ["nasa_gwettop","nasa_gwetprof","soil_moisture_deficit",
                                  "combined_drought_index","heat_drought_stress",
                                  "soil_moisture_top_root_ratio"],
    "target_water_access":    ["water_withdraw_total_km3","water_withdraw_per_capita",
                               "safe_water_pct","sanitation_pct"],
    # Écologie
    "target_forest_share":    ["forest_area_km2","forest_change","tree_cover_loss_ha",
                               "tree_cover_loss_cumul5y","deforestation_annual",
                               "deforestation_pct_annual","forest_per_capita_km2"],
    "target_tree_cover_loss": ["forest_change","forest_share_pct","forest_area_km2",
                               "deforestation_annual","deforestation_pct_annual",
                               "tree_cover_loss_cumul5y","forest_per_capita_km2"],
    "target_thermal_anomaly": ["owid_temp_anomaly","be_t_anom_annual",
                               "be_t_anom_vs_preindustrial","be_t_baseline_1850_1900"],
    # Émissions
    "target_co2_emissions":     ["methane_total_co2eq","n2o_total_co2eq","co_emissions",
                                  "owid_co2_pc","co2_per_capita_calc","co2_ppm_global"],
    "target_methane_emissions": ["co2_annual_t","n2o_total_co2eq","owid_methane"],
    "target_n2o_emissions":     ["co2_annual_t","methane_total_co2eq","owid_n2o"],
    # Énergie
    "target_solar_consumption":  ["wind_generation_twh","hydro_consumption_twh",
                                   "elec_generation_gwh","elec_renew_share"],
    "target_wind_generation":    ["solar_consumption_twh","hydro_consumption_twh",
                                   "elec_generation_gwh","elec_renew_share"],
    "target_hydro_consumption":  ["solar_consumption_twh","wind_generation_twh",
                                   "elec_generation_gwh","elec_renew_share"],
    "target_coal_production":    ["oil_production_twh","gas_production_twh"],
    "target_oil_production":     ["coal_production_twh","gas_production_twh"],
    "target_gas_production":     ["coal_production_twh","oil_production_twh"],
}

# Toutes les cibles agricoles → drop les autres yields croisés
ALL_YIELD_TARGETS = {t for t in TARGETS if t.startswith("target_yield_")}
YIELD_TARGETS = ALL_YIELD_TARGETS

SOCIO_TARGETS = set()
DISASTER_TARGETS = set()
