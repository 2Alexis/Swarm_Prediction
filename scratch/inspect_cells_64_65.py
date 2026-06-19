import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('data_visualization.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [64, 65]:
    if idx < len(nb['cells']):
        cell = nb['cells'][idx]
        if cell['cell_type'] == 'code':
            print(f"\n==================== CODE CELL {idx} ====================")
            print("".join(cell.get('source', [])))
