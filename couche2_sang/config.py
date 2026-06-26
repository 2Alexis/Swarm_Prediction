"""
couche2_sang/config.py — Configuration spécifique COUCHE 2 : Le Sang.

Domaine : démographie + épidémiologie + santé publique + catastrophes humaines

Cibles :
  - Démographie : natalité, mortalité brut, mortalité infantile, espérance de vie,
                  croissance démo, fécondité, migration nette
  - Santé/Épidémiologie : retard de croissance (stunting)
  - Catastrophes humaines : décès et personnes affectées par catastrophes

PRINCIPE : on prédit Couche 2 (les gens) à partir UNIQUEMENT des features Couche 1
(environnement, agriculture, écologie) → c'est l'idée centrale du projet :
« la terre nourrit les gens, l'environnement détermine la démographie ».
"""

# ── CIBLES COUCHE 2 ────────────────────────────────────────────────────────
TARGETS = {
    # ── Démographie (sous-couche 2A) ──
    "target_birth_rate":         "Natalité (‰)",
    "target_death_rate":         "Mortalité brut (‰)",
    "target_child_mortality":    "Mortalité infantile (<5 ans)",
    "target_life_expectancy":    "Espérance de vie",
    "target_pop_growth":         "Croissance démographique",
    "target_net_migration":      "Migration nette",
    "target_fertility":          "Taux de fécondité",

    # ── Santé / Épidémiologie (sous-couche 2B) ──
    "target_stunting":           "Retard de croissance enfants (stunting)",

    # ── Catastrophes humaines (sous-couche 2C) ──
    "target_disaster_deaths":    "Décès dus aux catastrophes (log)",
    "target_disaster_affected":  "Personnes affectées (log)",
}

# Sous-catégories
SUBLAYERS = {
    "Démographie": [
        "target_birth_rate","target_death_rate","target_child_mortality",
        "target_life_expectancy","target_pop_growth","target_net_migration",
        "target_fertility",
    ],
    "Santé": [
        "target_stunting",
    ],
    "Catastrophes humaines": [
        "target_disaster_deaths","target_disaster_affected",
    ],
}

# Source brute de chaque cible (à blacklister)
TARGET_SOURCE = {
    "target_birth_rate":         "Birth_Rate",
    "target_death_rate":         "Death_Rate",
    "target_child_mortality":    "Child_Mort",
    "target_life_expectancy":    "Life_Exp",
    "target_pop_growth":         "Pop_Growth",
    "target_net_migration":      "Net_Migration",
    "target_fertility":          "Fertility_Rate",
    "target_stunting":           "stunting_pct",
    "target_disaster_deaths":    "disaster_deaths",
    "target_disaster_affected":  "disaster_affected",
}

# Leaks spécifiques (cibles démographiques entre elles + sources)
EXTRA_LEAKS = {
    "target_birth_rate":      ["Death_Rate","Net_Migration","Pop_Growth","Fertility_Rate",
                               "owid_crude_birth_rate","owid_crude_death_rate",
                               "owid_births_total","owid_deaths_total",
                               "births_per_1000_minus_deaths","target_birth_rate_owid"],
    "target_death_rate":      ["Birth_Rate","Net_Migration","Pop_Growth","Fertility_Rate",
                               "owid_crude_birth_rate","owid_crude_death_rate",
                               "owid_births_total","owid_deaths_total",
                               "births_per_1000_minus_deaths",
                               "adult_mortality_male","adult_mortality_female",
                               "infant_deaths_total","target_death_rate_owid"],
    "target_net_migration":   ["Birth_Rate","Death_Rate","Pop_Growth","Fertility_Rate",
                               "owid_crude_birth_rate","owid_crude_death_rate",
                               "owid_births_total","owid_deaths_total",
                               "refugees_origin","refugees_destination",
                               "refugees_origin_owid","idps_conflict"],
    "target_pop_growth":      ["Birth_Rate","Death_Rate","Net_Migration","Fertility_Rate",
                               "owid_crude_birth_rate","owid_crude_death_rate"],
    "target_fertility":       ["Birth_Rate","Death_Rate","Pop_Growth",
                               "owid_crude_birth_rate","owid_crude_death_rate"],
    "target_child_mortality": ["infant_deaths_total"],
    "target_stunting":        ["wasting_pct","overweight_pct","malnutrition_compound",
                               "infant_deaths_total","Hunger_Index","ghi_owid"],
    "target_disaster_deaths": ["disaster_affected","disaster_damages_usd","disaster_events",
                               "disaster_affected_per_capita","disaster_deaths_per_million",
                               "disaster_events_cumul5y"],
    "target_disaster_affected": ["disaster_deaths","disaster_damages_usd","disaster_events",
                                 "disaster_affected_per_capita","disaster_deaths_per_million",
                                 "disaster_events_cumul5y"],
}

# Pour la couche 2 : toutes les cibles démo/santé sont SOCIO_TARGETS
# → blacklist complète des variables socio-éco pour rester strict env→démo
SOCIO_TARGETS = {
    "target_child_mortality","target_life_expectancy","target_pop_growth",
    "target_stunting","target_fertility",
    "target_birth_rate","target_death_rate","target_net_migration",
}

# Catastrophes humaines = cibles disaster
DISASTER_TARGETS = {
    "target_disaster_deaths","target_disaster_affected",
}

# Pas de blacklist agro pour les cibles démo
YIELD_TARGETS = set()
