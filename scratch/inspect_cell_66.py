import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('data_visualization.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Print index 66
idx = 66
if idx < len(nb['cells']):
    cell = nb['cells'][idx]
    print(f"==================== CELL {idx} ({cell['cell_type']}) ====================")
    print("".join(cell.get('source', [])))
else:
    print(f"Cell {idx} is out of range (total cells: {len(nb['cells'])})")
