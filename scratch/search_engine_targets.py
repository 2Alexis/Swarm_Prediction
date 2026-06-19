import sys

with open('layer1_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

targets = [
    'predicted_yield_kgha', 'water_stress_index', 'koppen_geiger_class', 
    'flood_risk_index', 'soil_degradation_rate', 'fauna_herbivore_biomass_kg_km2', 
    'vector_suitability_index', 'local_heat_island_anomaly', 'soil_erosion_index'
]

print("Searching layer1_engine.py...")
for t in targets:
    if t in content:
        print(f"Found {t}!")
    else:
        print(f"NOT found: {t}")
