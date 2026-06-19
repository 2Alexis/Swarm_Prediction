import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

nb_path = 'training_pipeline.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Loaded {nb_path}. Total cells: {len(nb['cells'])}")

for idx, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    source = "".join(cell.get('source', []))
    first_line = source.split('\n')[0] if source else ''
    if cell_type == 'markdown':
        if first_line.startswith('#'):
            print(f"Cell {idx} (markdown): {first_line}")
    elif cell_type == 'code':
        print(f"Cell {idx} (code): {first_line[:80]}...")
