import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('data_visualization.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [40, 41, 42]:
    cell = nb['cells'][idx]
    print(f"\n=== CELL {idx} ({cell['cell_type']}) ===")
    print("".join(cell.get('source', [])))
