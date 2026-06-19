import json
import sys

notebook_path = "data_visualization.ipynb"
output_path = "scratch/target_search_results.txt"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
keywords = ['inondation', 'crue', 'flood', 'erosion', 'érosion', 'stress', 'hydrique', 'faune', 'parasite', 'malaria', 'anomalie', 'thermique', 'biome', 'desertification', 'désertification']

with open(output_path, 'w', encoding='utf-8') as out:
    out.write(f"Total cells: {len(cells)}\n\n")
    
    for i, cell in enumerate(cells):
        source_str = "".join(cell.get('source', []))
        matched = [kw for kw in keywords if kw.lower() in source_str.lower()]
        if matched:
            out.write(f"--- Cell {i+1} (index {i}) matches: {matched} ---\n")
            out.write(f"Cell Type: {cell.get('cell_type')}\n")
            out.write("Source:\n")
            out.write(source_str[:1200])
            out.write("\n" + "="*50 + "\n\n")

print("Search completed. Output written to scratch/target_search_results.txt")
