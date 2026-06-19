import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

nb_path = 'data_visualization.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells_to_inspect = [4, 43, 45, 52, 61, 66, 74]

for idx in cells_to_inspect:
    if idx < len(nb['cells']):
        cell = nb['cells'][idx]
        print(f"\n==================== CELL {idx} ({cell['cell_type']}) ====================")
        print("".join(cell.get('source', [])))
    else:
        print(f"\nCell {idx} is out of range (total cells: {len(nb['cells'])})")
