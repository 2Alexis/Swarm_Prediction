import json

with open('data_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def escape_str(s):
    return s.encode('ascii', 'backslashreplace').decode('ascii')

for i, cell in enumerate(nb['cells']):
    source_str = "".join(cell['source'])
    if 'Layer1Engine' in source_str:
        print(f"Cell {i}: {cell['cell_type']}")
        print("Source:")
        print(escape_str(source_str[:300]))
        print("-" * 50)
