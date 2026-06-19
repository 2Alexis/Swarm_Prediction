import json

# Define the source code of robust find_col for the notebook cell
find_col_code = [
    "def find_col(df, *names):\n",
    "    # Exact match first\n",
    "    for name in names:\n",
    "        if name in df.columns:\n",
    "            return name\n",
    "            \n",
    "    # Case-insensitive match\n",
    "    for name in names:\n",
    "        for col in df.columns:\n",
    "            if name.lower() == col.lower():\n",
    "                return col\n",
    "                \n",
    "    # Normalize unicode/strip accents and non-alphanumeric characters for robust matching\n",
    "    import unicodedata\n",
    "    def norm(s):\n",
    "        s = str(s).lower()\n",
    "        # strip accents\n",
    "        s = \"\".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')\n",
    "        # strip non-alphanumeric characters (like replacement character \\ufffd)\n",
    "        s = \"\".join(c for c in s if c.isalnum())\n",
    "        # common corruptions in these datasets:\n",
    "        s = s.replace('element', 'lment').replace('annee', 'anne').replace('unite', 'unit')\n",
    "        return s\n",
    "\n",
    "    for name in names:\n",
    "        norm_name = norm(name)\n",
    "        for col in df.columns:\n",
    "            if norm_name == norm(col):\n",
    "                return col\n",
    "\n",
    "    for name in names:\n",
    "        norm_name = norm(name)\n",
    "        for col in df.columns:\n",
    "            norm_col = norm(col)\n",
    "            if norm_name in norm_col or norm_col in norm_name:\n",
    "                # Avoid matching a 'code' column if the search target does not have 'code'\n",
    "                if 'code' in norm_col and 'code' not in norm_name:\n",
    "                    continue\n",
    "                return col\n",
    "                \n",
    "    # Fallback to the original matching logic\n",
    "    for name in names:\n",
    "        base = name.lower().replace('\u00e9', '').replace('\u00e8', '').replace('\u00ea', '').replace('\u00e0', '')\n",
    "        for col in df.columns:\n",
    "            col_clean = col.lower().replace('\u00e9', '').replace('\u00e8', '').replace('\u00ea', '').replace('\u00e0', '')\n",
    "            if len(base) >= 4 and (base[:4] in col_clean or base[:4] in col.lower()):\n",
    "                return col\n",
    "    return None\n"
]

geology_markdown_source = [
    "### 5\ufe0f\u20e3 Nouveaux Datasets G\u00e9ologiques (Minerais, Charbon, Fer, P\u00e9trole & Gaz)\n",
    "\n",
    "*   **Pourquoi ces datasets ?** Fournit la r\u00e9partition g\u00e9ographique des gisements miniers et \u00e9nerg\u00e9tiques mondiaux, indispensables pour simuler les cha\u00eenes industrielles de la civilisation.\n",
    "*   **Nettoyage appliqu\u00e9 :**\n",
    "    *   *MRDS (USGS)* : Chargement, exclusion des sites sans coordonn\u00e9es GPS valides, renommage des colonnes.\n",
    "    *   *Charbon (Global Coal Mine Tracker)* : Chargement de l'onglet `Non-closed mines`, extraction des coordonn\u00e9es, capacit\u00e9s, et filtrage.\n",
    "    *   *Fer (Global Iron Ore Mines Tracker)* : Extraction et parsing des coordonn\u00e9es textuelles `Latitude, Longitude` \u00e0 partir de la colonne `Coordinates`.\n",
    "    *   *P\u00e9trole et Gaz (Global Oil & Gas Extraction Tracker)* : Chargement de l'onglet `Field-level main data`, filtrage.\n"
]

geology_code_source = [
    "# 1. USGS MRDS Minerals\n",
    "print(\"Nettoyage mrds.csv...\")\n",
    "mrds_path = os.path.join(RAW_DIR, \"mrds.csv\")\n",
    "if os.path.exists(mrds_path):\n",
    "    df_mrds = pd.read_csv(mrds_path, low_memory=False)\n",
    "    cols = ['site_name', 'latitude', 'longitude', 'commod1', 'prod_size', 'dev_stat']\n",
    "    df_mrds = df_mrds[[c for c in cols if c in df_mrds.columns]].copy()\n",
    "    df_mrds = df_mrds.dropna(subset=['latitude', 'longitude'])\n",
    "    df_mrds = df_mrds.drop_duplicates()\n",
    "    df_mrds = df_mrds.rename(columns={\n",
    "        'site_name': 'Nom',\n",
    "        'latitude': 'Latitude',\n",
    "        'longitude': 'Longitude',\n",
    "        'commod1': 'Commodite',\n",
    "        'prod_size': 'Taille_Production',\n",
    "        'dev_stat': 'Statut_Developpement'\n",
    "    })\n",
    "    save_cleaned(df_mrds, \"mrds_cleaned.csv\")\n",
    "else:\n",
    "    print(\"Fichier mrds.csv absent.\")\n",
    "\n",
    "# 2. Mines de charbon\n",
    "print(\"Nettoyage mines de charbon...\")\n",
    "coal_path = os.path.join(RAW_DIR, \"geologie\", \"Global Coal Mine Tracker, May 2026__.xlsx\")\n",
    "if os.path.exists(coal_path):\n",
    "    df_coal = pd.read_excel(coal_path, sheet_name=\"Non-closed mines\")\n",
    "    df_coal = df_coal.dropna(subset=['Latitude', 'Longitude'])\n",
    "    cols_keep = {\n",
    "        'GEM Mine ID': 'Mine_ID',\n",
    "        'Mine Name': 'Nom',\n",
    "        'Country / Area': 'Pays',\n",
    "        'Status': 'Statut',\n",
    "        'Capacity (Mtpa)': 'Capacite_Mtpa',\n",
    "        'Production (Mtpa)': 'Production_Mtpa',\n",
    "        'Coal Type': 'Type_Charbon',\n",
    "        'Latitude': 'Latitude',\n",
    "        'Longitude': 'Longitude'\n",
    "    }\n",
    "    df_coal = df_coal[[c for c in cols_keep.keys() if c in df_coal.columns]].rename(columns=cols_keep)\n",
    "    df_coal = df_coal.drop_duplicates()\n",
    "    save_cleaned(df_coal, \"coal_mines_cleaned.csv\")\n",
    "else:\n",
    "    print(\"Tracker mines de charbon absent.\")\n",
    "\n",
    "# 3. Mines de fer\n",
    "print(\"Nettoyage mines de fer...\")\n",
    "iron_path = os.path.join(RAW_DIR, \"geologie\", \"Global-Iron-Ore-Mines-Tracker-August-2025-V1.xlsx\")\n",
    "if os.path.exists(iron_path):\n",
    "    df_iron = pd.read_excel(iron_path, sheet_name=\"Main Data\")\n",
    "    df_iron = df_iron.dropna(subset=['Coordinates'])\n",
    "    \n",
    "    def parse_coords(coord_str):\n",
    "        try:\n",
    "            parts = str(coord_str).split(\",\")\n",
    "            if len(parts) == 2:\n",
    "                return float(parts[0].strip()), float(parts[1].strip())\n",
    "        except Exception:\n",
    "            pass\n",
    "        return np.nan, np.nan\n",
    "        \n",
    "    coords = df_iron['Coordinates'].apply(parse_coords)\n",
    "    df_iron['Latitude'] = [c[0] for c in coords]\n",
    "    df_iron['Longitude'] = [c[1] for c in coords]\n",
    "    df_iron = df_iron.dropna(subset=['Latitude', 'Longitude'])\n",
    "    \n",
    "    cols_keep = {\n",
    "        'GEM Asset ID': 'Asset_ID',\n",
    "        'Asset name (English)': 'Nom',\n",
    "        'Country/Area': 'Pays',\n",
    "        'Operating status': 'Statut',\n",
    "        'Production 2024 (ttpa)': 'Production_2024_ttpa',\n",
    "        'Design capacity (ttpa)': 'Capacite_ttpa',\n",
    "        'Latitude': 'Latitude',\n",
    "        'Longitude': 'Longitude'\n",
    "    }\n",
    "    df_iron = df_iron[[c for c in cols_keep.keys() if c in df_iron.columns]].rename(columns=cols_keep)\n",
    "    df_iron = df_iron.drop_duplicates()\n",
    "    save_cleaned(df_iron, \"iron_mines_cleaned.csv\")\n",
    "else:\n",
    "    print(\"Tracker mines de fer absent.\")\n",
    "\n",
    "# 4. P\u00e9trole et Gaz\n",
    "print(\"Nettoyage p\u00e9trole et gaz...\")\n",
    "oil_path = os.path.join(RAW_DIR, \"geologie\", \"Global-Oil-and-Gas-Extraction-Tracker-March-2026.xlsx\")\n",
    "if os.path.exists(oil_path):\n",
    "    df_oil = pd.read_excel(oil_path, sheet_name=\"Field-level main data\")\n",
    "    df_oil = df_oil.dropna(subset=['Latitude', 'Longitude'])\n",
    "    cols_keep = {\n",
    "        'Unit ID': 'Unit_ID',\n",
    "        'Unit Name': 'Nom',\n",
    "        'Fuel type': 'Type_Combustible',\n",
    "        'Country/Area': 'Pays',\n",
    "        'Production Type': 'Type_Production',\n",
    "        'Status': 'Statut',\n",
    "        'Latitude': 'Latitude',\n",
    "        'Longitude': 'Longitude'\n",
    "    }\n",
    "    df_oil = df_oil[[c for c in cols_keep.keys() if c in df_oil.columns]].rename(columns=cols_keep)\n",
    "    df_oil = df_oil.drop_duplicates()\n",
    "    save_cleaned(df_oil, \"oil_gas_cleaned.csv\")\n",
    "else:\n",
    "    print(\"Tracker p\u00e9trole et gaz absent.\")\n"
]

notebook_path = 'data_cleaning.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update find_col in cell 2 (index 2)
# Let's verify cell 2 content has find_col
source = nb['cells'][2]['source']
# We want to replace find_col within the source lines
# Let's find where 'def find_col' starts and 'def clean_common' (or next function) starts.
start_idx = -1
end_idx = -1
for idx, line in enumerate(source):
    if line.strip().startswith('def find_col'):
        start_idx = idx
    if line.strip().startswith('def clean_common') and start_idx != -1:
        end_idx = idx
        break

if start_idx != -1 and end_idx != -1:
    source[start_idx:end_idx] = find_col_code
    print("Updated find_col in cell 2.")
else:
    print("Could not find boundaries of find_col in cell 2!")

# 2. Re-route geologie_risques to geologie in other cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_src = []
        for line in cell['source']:
            new_src.append(line.replace('geologie_risques', 'geologie'))
        cell['source'] = new_src

# 3. Create and insert the two geology cells after Cell 22
new_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": geology_markdown_source
}

new_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": geology_code_source
}

# Insert at index 23 (after 22)
nb['cells'].insert(23, new_markdown_cell)
nb['cells'].insert(24, new_code_cell)
print("Inserted new geology cells after cell 22.")

# Save the updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Successfully saved data_cleaning.ipynb.")
