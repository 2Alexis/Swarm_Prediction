import json

with open('data_visualization.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Loaded visualization notebook with {len(nb['cells'])} cells.")

for i, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    source_prefix = "".join(cell['source'][:2]) if cell['source'] else ""
    escaped_prefix = source_prefix.encode('ascii', 'backslashreplace').decode('ascii')
    print(f"Cell {i}: {cell_type} | {escaped_prefix[:100]}...")
