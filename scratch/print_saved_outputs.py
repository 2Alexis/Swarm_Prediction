import json

with open('data_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def escape_str(s):
    return s.encode('ascii', 'backslashreplace').decode('ascii')

cell = nb['cells'][11]
if 'outputs' in cell:
    for out in cell['outputs']:
        print("Output type:", out.get('output_type'))
        if 'text' in out:
            print(escape_str("".join(out['text'])))
        elif 'data' in out:
            print(escape_str("".join(out['data'].get('text/plain', []))))
        print("=" * 40)
