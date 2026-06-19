import json

with open('data_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def escape_str(s):
    return s.encode('ascii', 'backslashreplace').decode('ascii')

targets = [
    'predicted_yield_kgha', 'water_stress_index', 'koppen_geiger_class', 
    'flood_risk_index', 'soil_degradation_rate', 'vector_suitability_index', 
    'local_heat_island_anomaly', 'soil_erosion_index'
]

for i, cell in enumerate(nb['cells']):
    source_str = "".join(cell['source'])
    found = [t for t in targets if t in source_str]
    if found:
        print(f"Cell {i} ({cell['cell_type']}) contains {found}:")
        print(escape_str(source_str[:600]))
        print("=" * 80)
