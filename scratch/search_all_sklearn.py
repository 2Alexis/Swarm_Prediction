import json

with open('data_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def escape_str(s):
    return s.encode('ascii', 'backslashreplace').decode('ascii')

terms = ['sklearn', 'joblib', 'predict', 'xgb', 'model', 'imputer']

for i, cell in enumerate(nb['cells']):
    source_str = "".join(cell['source'])
    found = [t for t in terms if t in source_str.lower()]
    if found:
        print(f"Cell {i} ({cell['cell_type']}) contains {found}:")
        print(escape_str(source_str[:500]))
        print("=" * 80)
