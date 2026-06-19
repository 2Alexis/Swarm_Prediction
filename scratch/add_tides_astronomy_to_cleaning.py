import json

tides_markdown = [
    "# \U0001f30a Étape 8 & 9 : Marées & Astronomie (Vérification Algorithmique)\n",
    "\n",
    "Les deux dernières étapes ne reposent pas sur le nettoyage de fichiers de données brutes tabulaires statiques :\n",
    "1. **Marées et Dynamique Océanique** : Intégrées à l'aide de la bibliothèque open-source `pyTMD` (Tidal Model Data) pour modéliser l'amplitude de la marée d'équilibre.\n",
    "2. **Astronomie, Photopériode et Saisons** : Calculées dynamiquement par les équations physiques de déclinaison solaire et d'angle horaire pour obtenir la photopériode (durée du jour) en fonction de la latitude.\n",
    "\n",
    "Nous validons ci-dessous le bon fonctionnement des paquets et des modèles mathématiques associés.\n"
]

tides_code = [
    "import numpy as np\n",
    "import xarray as xr\n",
    "import pyTMD.predict\n",
    "import math\n",
    "\n",
    "# 1. Validation de pyTMD (Marées)\n",
    "print(\"Validation de pyTMD (Marées)...\")\n",
    "t = np.linspace(0.0, 1.0, 24)\n",
    "ds = xr.Dataset(coords={'y': [48.8566], 'x': [2.3522]})\n",
    "tide_series = pyTMD.predict.equilibrium_tide(t, ds)\n",
    "eq_range = float(tide_series.values.max() - tide_series.values.min())\n",
    "print(f\"  [OK] Amplitude de marée d'équilibre calculée pour Paris : {eq_range*1000:.3f} mm\")\n",
    "\n",
    "# 2. Validation des formules de photopériode (Astronomie)\n",
    "print(\"Validation de la photopériode...\")\n",
    "def get_photoperiod(lat, doy):\n",
    "    phi = math.radians(lat)\n",
    "    delta = 0.409 * math.sin(2.0 * math.pi * (doy - 80.0) / 365.0)\n",
    "    val = -math.tan(phi) * math.tan(delta)\n",
    "    if val >= 1.0:\n",
    "        return 0.0\n",
    "    elif val <= -1.0:\n",
    "        return 24.0\n",
    "    else:\n",
    "        h_s = math.acos(val)\n",
    "        return (24.0 / math.pi) * h_s\n",
    "\n",
    "paris_summer = get_photoperiod(48.8566, 172)\n",
    "paris_winter = get_photoperiod(48.8566, 355)\n",
    "print(f\"  [OK] Photopériode de Paris - Solstice d'été : {paris_summer:.2f} heures | Solstice d'hiver : {paris_winter:.2f} heures\")\n"
]

notebook_path = 'data_cleaning.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Insert before the last cell (which is index len(nb['cells'])-1)
last_index = len(nb['cells']) - 1

new_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": tides_markdown
}

new_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": tides_code
}

nb['cells'].insert(last_index, new_markdown_cell)
nb['cells'].insert(last_index + 1, new_code_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully added Steps 8 & 9 to data_cleaning.ipynb.")
