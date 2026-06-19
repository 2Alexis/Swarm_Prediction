import json

notebook_path = 'data_visualization.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixed_cells = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "style='Status'" in line:
                line = line.replace("style='Status'", "style='Statut'")
                fixed_cells += 1
            new_source.append(line)
        cell['source'] = new_source

if fixed_cells > 0:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    print(f"Fixed {fixed_cells} cell(s) in data_visualization.ipynb.")
else:
    print("Could not find style='Status' in the notebook cells!")
