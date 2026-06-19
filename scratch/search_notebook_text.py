import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

nb_path = 'data_visualization.ipynb'
query = 'belg'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Loaded notebook. Total cells: {len(nb['cells'])}")

matches = []
for idx, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    # Concatenate source lines
    source = "".join(cell.get('source', []))
    if query.lower() in source.lower():
        matches.append((idx, cell_type, source[:200] + "..."))

print(f"\nFound {len(matches)} cells matching '{query}':")
for idx, cell_type, snippet in matches:
    print(f"Cell {idx} ({cell_type}): {snippet}")
