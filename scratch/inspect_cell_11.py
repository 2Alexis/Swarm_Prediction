import json

with open('data_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def escape_str(s):
    return s.encode('ascii', 'backslashreplace').decode('ascii')

print("Cell 11 source:")
print(escape_str("".join(nb['cells'][11]['source'])))
