import json

with open('data_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def escape_str(s):
    return s.encode('ascii', 'backslashreplace').decode('ascii')

for i in range(3, 12):
    cell = nb['cells'][i]
    if cell['cell_type'] == 'code':
        print(f"Cell {i} (code):")
        print(escape_str("".join(cell['source'])))
        print("-" * 50)
