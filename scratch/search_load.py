import json

with open('data_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def escape_str(s):
    return s.encode('ascii', 'backslashreplace').decode('ascii')

for i, cell in enumerate(nb['cells']):
    source_str = "".join(cell['source'])
    if 'load' in source_str or 'best_model' in source_str:
        print(f"Cell {i} ({cell['cell_type']}):")
        print(escape_str(source_str))
        print("=" * 80)
