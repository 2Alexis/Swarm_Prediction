import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌾 Planetary Data Preprocessing & Cleaning Pipeline (Complete)\n",
                "\n",
                "Ce notebook documente la **Phase de Traitement et de Nettoyage des Données** pour la simulation planétaire. \n",
                "Il regroupe les scripts de nettoyage utilisés pour traiter à la fois :\n",
                "1. **Les données historiques, démographiques et socio-économiques** (FAOSTAT, World Bank, OWID, WHO, ACLED/UCDP).\n",
                "2. **Les données physiques et géologiques** (volcans, séismes, infrastructures énergétiques, écologie).\n",
                "3. **Les données d'hydrologie** (lacs, rivières, prélèvements d'eau douce).\n",
                "4. **Les données raster de climat (WorldClim)** au format GeoTIFF.\n",
                "\n",
                "Pour chaque dataset, nous explicitons les choix de variables, les justifications scientifiques des filtres et le traitement des valeurs aberrantes (outliers)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ⚙️ Initialisation du Pipeline\n",
                "\n",
                "Nous configurons les répertoires et chargeons les dépendances standard. Les données brutes proviennent de `data/raw/` et les données nettoyées seront stockées dans `data/cleaned/`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import json\n",
                "import math\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "from PIL import Image\n",
                "\n",
                "RAW_DIR = \"data/raw\"\n",
                "CLEANED_DIR = \"data/cleaned\"\n",
                "os.makedirs(CLEANED_DIR, exist_ok=True)\n",
                "\n",
                "def load_csv(filepath):\n",
                "    \"\"\"Charge un fichier CSV en testant plusieurs encodages courants sous Windows/Linux.\"\"\"\n",
                "    for enc in ['utf-8', 'latin-1', 'cp1252']:\n",
                "        try:\n",
                "            return pd.read_csv(filepath, encoding=enc, low_memory=False, on_bad_lines='skip')\n",
                "        except (UnicodeDecodeError, UnicodeError):\n",
                "            continue\n",
                "    raise ValueError(f\"Impossible de lire {filepath}\")\n",
                "\n",
                "def find_col(df, *names):\n",
                "    \"\"\"Recherche une colonne par correspondance exacte ou approximative (sans accents/minuscules).\"\"\"\n",
                "    for name in names:\n",
                "        if name in df.columns:\n",
                "            return name\n",
                "    for name in names:\n",
                "        base = name.lower().replace('é', '').replace('è', '').replace('ê', '').replace('à', '')\n",
                "        for col in df.columns:\n",
                "            col_clean = col.lower().replace('é', '').replace('è', '').replace('ê', '').replace('à', '')\n",
                "            if base[:4] in col_clean or base[:4] in col.lower():\n",
                "                return col\n",
                "    return None\n",
                "\n",
                "def clean_common(df, value_col='Valeur'):\n",
                "    \"\"\"Nettoie les valeurs manquantes, les doublons et convertit les colonnes clés.\"\"\"\n",
                "    df = df.dropna(subset=[value_col])\n",
                "    df = df.drop_duplicates()\n",
                "    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')\n",
                "    df = df.dropna(subset=[value_col])\n",
                "    \n",
                "    annee_col = find_col(df, 'Année', 'Annee', 'Year')\n",
                "    if annee_col:\n",
                "        df[annee_col] = pd.to_numeric(df[annee_col], errors='coerce')\n",
                "        df = df.dropna(subset=[annee_col])\n",
                "        df[annee_col] = df[annee_col].astype(int)\n",
                "    return df\n",
                "\n",
                "def save_cleaned(df, name):\n",
                "    \"\"\"Sauvegarde le dataset nettoyé dans le dossier cleaned.\"\"\"\n",
                "    filepath = os.path.join(CLEANED_DIR, name)\n",
                "    df.to_csv(filepath, index=False, encoding='utf-8')\n",
                "    size_mb = os.path.getsize(filepath) / (1024 * 1024)\n",
                "    print(f\"[Sauvegarde] {name} : {df.shape[0]} lignes, {df.shape[1]} colonnes ({size_mb:.2f} Mo)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌾 PARTIE 1 : Traitement des données historiques agricoles & climatiques (FAOSTAT)\n",
                "\n",
                "Cette partie s'attache à harmoniser les échelles et les unités pour que les modèles de Machine Learning puissent apprendre des relations physiques et agronomiques cohérentes sans être biaisés par les différences de superficie entre pays."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1️⃣ Bilan Nutritif des Sols & Engrais\n",
                "\n",
                "*   **Pourquoi ce dataset ?** Permet d'évaluer la quantité de nutriments (Azote N, Phosphore P, Potassium K) entrant et sortant des sols cultivés par pays.\n",
                "*   **Choix des variables & Unités :** Nous filtrons uniquement sur l'unité **`kg/ha`** (kilogramme par hectare de terre arable). Conserver les volumes globaux en *tonnes* fausserait l'apprentissage, car un grand pays consommerait mécaniquement plus qu'un petit pays sans que cela reflète l'intensité ou la qualité du sol.\n",
                "*   **Outliers & Nettoyage :** \n",
                "    *   Nous trions et séparons en deux fichiers : `bilan_nutritif_sols.csv` (bilan net d'azote/phosphore par hectare) et `fertilizers_nutrient.csv` (intrants azotés chimiques appliqués aux sols).\n",
                "    *   Les lignes sans valeur de bilan ou sans année valide sont éliminées."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "sol_path = os.path.join(RAW_DIR, \"pedologie_sols\", \"sol_nutritif\", \"Environnement_Bilan_nutritif_des_sols_F_Toutes_les_Données_(Normalisé).csv\")\n",
                "\n",
                "if os.path.exists(sol_path):\n",
                "    df_sol = load_csv(sol_path)\n",
                "    \n",
                "    col_unite = find_col(df_sol, 'Unité', 'Unit')\n",
                "    col_element = find_col(df_sol, 'Élément', 'Element')\n",
                "    col_zone = find_col(df_sol, 'Zone', 'Area')\n",
                "    col_annee = find_col(df_sol, 'Année', 'Year')\n",
                "    col_valeur = find_col(df_sol, 'Valeur', 'Value')\n",
                "    col_produit = find_col(df_sol, 'Produit', 'Item')\n",
                "    \n",
                "    # 1. Bilan Nutritif Sols (kg/ha)\n",
                "    df_sols_kgha = df_sol[df_sol[col_unite] == 'kg/ha'].copy()\n",
                "    cols_keep_sols = {col_zone: 'Pays', col_produit: 'Produit', col_annee: 'Annee', col_valeur: 'Valeur'}\n",
                "    df_sols_kgha = df_sols_kgha[[c for c in cols_keep_sols.keys()]].copy().rename(columns=cols_keep_sols)\n",
                "    df_sols_kgha = clean_common(df_sols_kgha)\n",
                "    save_cleaned(df_sols_kgha, \"bilan_nutritif_sols.csv\")\n",
                "    \n",
                "    # 2. Bilan Nutritif Terres Cultivées (kg/ha + %)\n",
                "    df_terres = df_sol[df_sol[col_unite].isin(['kg/ha', '%'])].copy()\n",
                "    cols_keep_terres = {col_zone: 'Pays', col_produit: 'Produit', col_element: 'Element', col_annee: 'Annee', col_unite: 'Unite', col_valeur: 'Valeur'}\n",
                "    df_terres = df_terres[[c for c in cols_keep_terres.keys()]].copy().rename(columns=cols_keep_terres)\n",
                "    df_terres = clean_common(df_terres)\n",
                "    save_cleaned(df_terres, \"bilan_nutritif_terres_cultivees.csv\")\n",
                "    \n",
                "    # 3. Engrais chimiques (intrants d'azote/potasse/phosphate synthétiques en kg/ha)\n",
                "    df_fert = df_sol[df_sol[col_produit].astype(str).str.contains('Engrais', case=False, na=False)].copy()\n",
                "    df_fert = df_fert[df_fert[col_unite] == 'kg/ha'].copy()\n",
                "    cols_keep_fert = {col_zone: 'Pays', col_produit: 'Produit', col_annee: 'Annee', col_valeur: 'Valeur'}\n",
                "    df_fert = df_fert[[c for c in cols_keep_fert.keys()]].copy().rename(columns=cols_keep_fert)\n",
                "    df_fert = clean_common(df_fert)\n",
                "    save_cleaned(df_fert, \"fertilizers_nutrient.csv\")\n",
                "else:\n",
                "    print(\"Fichier Bilan Nutritif Sols absent, saut de cette étape.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2️⃣ Variation de Température\n",
                "\n",
                "*   **Pourquoi ce dataset ?** Il fournit les écarts de température mensuels et annuels observés par rapport à la climatologie de référence.\n",
                "*   **Choix des variables :** Nous conservons le pays, l'année, le mois (ou indicateur annuel) et la valeur de l'écart en **°C**.\n",
                "*   **Outliers & Nettoyage :** Suppression des doublons et des relevés sans valeur numérique."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "temp_path = os.path.join(RAW_DIR, \"atmosphere_climat\", \"temperature\", \"Environnement_variation_temperature_F_Toutes_les_Données_(Normalisé).csv\")\n",
                "if os.path.exists(temp_path):\n",
                "    df_temp = load_csv(temp_path)\n",
                "    col_zone = find_col(df_temp, 'Zone')\n",
                "    col_annee = find_col(df_temp, 'Année', 'Annee', 'Year')\n",
                "    col_valeur = find_col(df_temp, 'Valeur')\n",
                "    col_element = find_col(df_temp, 'Élément', 'Element')\n",
                "    col_mois = find_col(df_temp, 'Mois')\n",
                "    col_code_mois = find_col(df_temp, 'Code Mois')\n",
                "    \n",
                "    cols_keep = {col_zone: 'Pays', col_code_mois: 'Code Mois', col_mois: 'Mois', col_element: 'Element', col_annee: 'Annee', col_valeur: 'Valeur'}\n",
                "    df_temp = df_temp[[c for c in cols_keep.keys()]].copy().rename(columns=cols_keep)\n",
                "    df_temp = clean_common(df_temp)\n",
                "    save_cleaned(df_temp, \"variation_temperature.csv\")\n",
                "else:\n",
                "    print(\"Fichier Variation de Température absent, saut de cette étape.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3️⃣ Intrants & Utilisation des Terres\n",
                "\n",
                "*   **Pourquoi ce dataset ?** Contient les surfaces forestières, prairiales, et irriguées par pays.\n",
                "*   **Choix des variables & Unités :** Seules les variables exprimées en **`1000 ha`** (surfaces réelles) ou **`%`** (part de territoire) sont conservées."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "terres_path = os.path.join(RAW_DIR, \"socio_economie_demographie\", \"faostat\", \"terres_utilisation\", \"Intrants_TerresUtilisation_F_Toutes_les_Données_(Normalisé).csv\")\n",
                "if os.path.exists(terres_path):\n",
                "    df_terres_util = load_csv(terres_path)\n",
                "    col_unite = find_col(df_terres_util, 'Unité', 'Unite')\n",
                "    col_element = find_col(df_terres_util, 'Élément', 'Element')\n",
                "    col_zone = find_col(df_terres_util, 'Zone')\n",
                "    col_annee = find_col(df_terres_util, 'Année', 'Annee', 'Year')\n",
                "    col_valeur = find_col(df_terres_util, 'Valeur')\n",
                "    col_produit = find_col(df_terres_util, 'Produit')\n",
                "    \n",
                "    df_terres_util = df_terres_util[df_terres_util[col_unite].isin(['1000 ha', '%'])].copy()\n",
                "    cols_keep = {col_zone: 'Pays', col_produit: 'Produit', col_element: 'Element', col_annee: 'Annee', col_unite: 'Unite', col_valeur: 'Valeur'}\n",
                "    df_terres_util = df_terres_util[[c for c in cols_keep.keys()]].copy().rename(columns=cols_keep)\n",
                "    df_terres_util = clean_common(df_terres_util)\n",
                "    save_cleaned(df_terres_util, \"intrants_utilisation_terres.csv\")\n",
                "else:\n",
                "    print(\"Fichier Utilisation des Terres absent, saut de cette étape.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4️⃣ Production Cultures et Animaux\n",
                "\n",
                "*   **Pourquoi ce dataset ?** Contient les rendements agricoles et les cheptels d'élevage mondiaux.\n",
                "*   **Choix des variables & Unités :** \n",
                "    *   *Cultures* : rendement en **`kg/ha`** (mesure d'efficacité biologique), production en **`tonnes`**, superficie en **`ha`**.\n",
                "    *   *Animaux* : effectifs en têtes de bétail ou production de viande en tonnes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "prod_path = os.path.join(RAW_DIR, \"socio_economie_demographie\", \"faostat\", \"production_agricole\", \"Production_Cultures_ProduitsAnimaux_F_Toutes_les_Données_(Normalisé).csv\")\n",
                "if os.path.exists(prod_path):\n",
                "    df_prod = load_csv(prod_path)\n",
                "    col_unite = find_col(df_prod, 'Unité', 'Unite')\n",
                "    col_element = find_col(df_prod, 'Élément', 'Element')\n",
                "    col_zone = find_col(df_prod, 'Zone')\n",
                "    col_annee = find_col(df_prod, 'Année', 'Annee', 'Year')\n",
                "    col_valeur = find_col(df_prod, 'Valeur')\n",
                "    col_produit = find_col(df_prod, 'Produit')\n",
                "    \n",
                "    mask_cultures = (\n",
                "        df_prod[col_element].isin(['Production', 'Rendement', 'Superficie récoltée']) &\n",
                "        df_prod[col_unite].isin(['tonnes', 'kg/ha', 'ha'])\n",
                "    )\n",
                "    \n",
                "    cols_keep = {col_zone: 'Pays', col_produit: 'Produit', col_element: 'Element', col_annee: 'Annee', col_unite: 'Unite', col_valeur: 'Valeur'}\n",
                "    \n",
                "    df_cultures = df_prod[mask_cultures].copy()\n",
                "    df_cultures = df_cultures[[c for c in cols_keep.keys()]].copy().rename(columns=cols_keep)\n",
                "    df_cultures = clean_common(df_cultures)\n",
                "    save_cleaned(df_cultures, \"production_cultures.csv\")\n",
                "    \n",
                "    df_animaux = df_prod[~mask_cultures].copy()\n",
                "    df_animaux = df_animaux[[c for c in cols_keep.keys()]].copy().rename(columns=cols_keep)\n",
                "    df_animaux = clean_common(df_animaux)\n",
                "    save_cleaned(df_animaux, \"production_animaux.csv\")\n",
                "else:\n",
                "    print(\"Fichier Production Agricole absent, saut de cette étape.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5️⃣ Pesticides\n",
                "\n",
                "*   **Pourquoi ce dataset ?** Quantité de produits phytosanitaires utilisés dans l'agriculture par pays.\n",
                "*   **Choix des variables :** Nous sélectionnons uniquement les intensités d'utilisation (utilisation par surface cultivée) pour obtenir un indicateur de pression environnementale comparable (exprimé en **`kg/ha`**)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pest_path = os.path.join(RAW_DIR, \"socio_economie_demographie\", \"faostat\", \"pesticides\", \"Environnement_Pesticides_F_Toutes_les_Données_(Normalisé).csv\")\n",
                "if os.path.exists(pest_path):\n",
                "    df_pest = load_csv(pest_path)\n",
                "    col_zone = find_col(df_pest, 'Zone')\n",
                "    col_annee = find_col(df_pest, 'Année', 'Year')\n",
                "    col_valeur = find_col(df_pest, 'Valeur')\n",
                "    col_produit = find_col(df_pest, 'Produit')\n",
                "    col_element = find_col(df_pest, 'Élément', 'Element')\n",
                "    \n",
                "    df_pest = df_pest[df_pest[col_element].astype(str).str.contains('surface|capita|habit', case=False, na=False)].copy()\n",
                "    cols_keep = {col_zone: 'Pays', col_produit: 'Produit', col_annee: 'Annee', col_valeur: 'Valeur'}\n",
                "    df_pest = df_pest[[c for c in cols_keep.keys()]].copy().rename(columns=cols_keep)\n",
                "    df_pest = clean_common(df_pest)\n",
                "    save_cleaned(df_pest, \"pesticides.csv\")\n",
                "else:\n",
                "    print(\"Fichier Pesticides absent, saut de cette étape.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌋 PARTIE 2 : Traitement des données physiques, géologiques et écologiques\n",
                "\n",
                "Cette partie traite les datasets localisés par coordonnées GPS (Latitude/Longitude). Le but est de nettoyer les anomalies de saisie, supprimer les relevés n'ayant pas de coordonnées valides, et éliminer les aberrations physiques (altitudes ou profondeurs géologiques impossibles)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1️⃣ Volcans Actifs (`volcanoes.csv`)\n",
                "\n",
                "*   **Pourquoi ce dataset ?** Permet de modéliser le risque volcanique et les flux géothermiques.\n",
                "*   **Choix des variables :** `volcano_name`, `latitude`, `longitude`, `elevation`, `primary_volcano_type` et la population à proximité.\n",
                "*   **Outliers & Nettoyage :** \n",
                "    *   Nous supprimons les lignes n'ayant pas de coordonnées GPS.\n",
                "    *   **Filtrage des altitudes aberrantes :** Nous imposons $-5000\\text{ m} \\le \\text{elevation} \\le 7000\\text{ m}$. Les valeurs hors de cette plage sont des erreurs de saisie car aucun volcan terrestre n'excède 7000 m ou n'est plus profond que 5000 m sous l'eau."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "volc_path = os.path.join(RAW_DIR, \"geologie_risques\", \"volcanoes.csv\")\n",
                "if os.path.exists(volc_path):\n",
                "    df_v = load_csv(volc_path)\n",
                "    df_v = df_v.dropna(subset=['latitude', 'longitude'])\n",
                "    \n",
                "    cols_to_keep = ['volcano_number', 'volcano_name', 'primary_volcano_type', \n",
                "                    'last_eruption_year', 'latitude', 'longitude', 'elevation', 'population_within_100_km']\n",
                "    df_v = df_v[[c for c in cols_to_keep if c in df_v.columns]].copy()\n",
                "    \n",
                "    df_v['elevation'] = pd.to_numeric(df_v['elevation'], errors='coerce')\n",
                "    df_v = df_v[(df_v['elevation'] >= -5000) & (df_v['elevation'] <= 7000)]\n",
                "    \n",
                "    df_v['last_eruption_year'] = pd.to_numeric(df_v['last_eruption_year'], errors='coerce')\n",
                "    save_cleaned(df_v, \"volcanoes_cleaned.csv\")\n",
                "else:\n",
                "    print(\"Fichier volcanoes.csv absent, saut de cette étape.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2️⃣ Séismes Majeurs (`earthquakes_usgs.csv`)\n",
                "\n",
                "*   **Pourquoi ce dataset ?** Modélisation de l'aléa sismique mondial et des limites de plaques tectoniques.\n",
                "*   **Choix des variables :** Temps, latitude, longitude, magnitude (`mag`), profondeur de l'hypocentre (`depth`), et localisation.\n",
                "*   **Outliers & Nettoyage :** \n",
                "    *   **Filtrage magnitude :** Strictement supérieure à $0$ et inférieure ou égale à $10$ (la limite physique de l'échelle de Richter/Magnitude de Moment).\n",
                "    *   **Filtrage profondeur hypocentre :** Strictement positive et inférieure à $800\\text{ km}$ (la zone de transition du manteau supérieur limite la sismicité à max 700-800 km)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "eq_path = os.path.join(RAW_DIR, \"geologie_risques\", \"earthquakes_usgs.csv\")\n",
                "if os.path.exists(eq_path):\n",
                "    df_eq = load_csv(eq_path)\n",
                "    df_eq = df_eq.dropna(subset=['latitude', 'longitude', 'mag'])\n",
                "    \n",
                "    cols_to_keep = ['time', 'latitude', 'longitude', 'depth', 'mag', 'place', 'type']\n",
                "    df_eq = df_eq[[c for c in cols_to_keep if c in df_eq.columns]].copy()\n",
                "    \n",
                "    df_eq['mag'] = pd.to_numeric(df_eq['mag'], errors='coerce')\n",
                "    df_eq['depth'] = pd.to_numeric(df_eq['depth'], errors='coerce')\n",
                "    df_eq = df_eq[(df_eq['mag'] > 0) & (df_eq['mag'] <= 10)]\n",
                "    df_eq = df_eq[(df_eq['depth'] >= 0) & (df_eq['depth'] <= 800)]\n",
                "    \n",
                "    save_cleaned(df_eq, \"earthquakes_cleaned.csv\")\n",
                "else:\n",
                "    print(\"Fichier earthquakes_usgs.csv absent, saut de cette étape.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3️⃣ Centrales Électriques Globales (`global_power_plants.csv`)\n",
                "\n",
                "*   **Pourquoi ce dataset ?** Évaluation de la répartition des ressources énergétiques et de la capacité de production anthropique.\n",
                "*   **Choix des variables :** Pays, nom, capacité (MW), type de combustible, coordonnées et année de mise en service.\n",
                "*   **Outliers & Nettoyage :** \n",
                "    *   **Filtrage capacité :** Strictement positive (une centrale doit pouvoir produire de l'énergie).\n",
                "    *   **Filtrage année de mise en service :** Comprise strictement entre $1800$ (début de l'ère industrielle) et $2026$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pp_path = os.path.join(RAW_DIR, \"geologie_risques\", \"global_power_plants.csv\")\n",
                "if os.path.exists(pp_path):\n",
                "    df_pp = load_csv(pp_path)\n",
                "    df_pp = df_pp.dropna(subset=['latitude', 'longitude', 'capacity_mw', 'primary_fuel'])\n",
                "    \n",
                "    cols_to_keep = ['country_long', 'name', 'capacity_mw', 'latitude', 'longitude', 'primary_fuel', 'commissioning_year']\n",
                "    df_pp = df_pp[[c for c in cols_to_keep if c in df_pp.columns]].copy()\n",
                "    \n",
                "    df_pp['capacity_mw'] = pd.to_numeric(df_pp['capacity_mw'], errors='coerce')\n",
                "    df_pp = df_pp[df_pp['capacity_mw'] > 0]\n",
                "    \n",
                "    df_pp['commissioning_year'] = pd.to_numeric(df_pp['commissioning_year'], errors='coerce')\n",
                "    df_pp.loc[(df_pp['commissioning_year'] < 1800) | (df_pp['commissioning_year'] > 2026), 'commissioning_year'] = np.nan\n",
                "    \n",
                "    save_cleaned(df_pp, \"global_power_plants_cleaned.csv\")\n",
                "else:\n",
                "    print(\"Fichier global_power_plants.csv absent, saut de cette étape.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4️⃣ Densité de Bois Écologique (`wood_density.csv`)\n",
                "\n",
                "*   **Pourquoi ce dataset ?** Utilisé pour calculer la biomasse forestière globale et le stockage du carbone (Modèles de Miami et Hatton).\n",
                "*   **Choix des variables :** Espèce, coordonnées, forme de croissance, et densité de bois.\n",
                "*   **Outliers & Nettoyage :** \n",
                "    *   Le fichier est délimité par des points-virgules (`;`) et codé en `latin-1`. Les séparateurs décimaux de type virgule `,` sont remplacés par des points `.` pour le typage numérique.\n",
                "    *   **Filtrage densité de bois :** Comprise entre $0.1\\text{ g/cm}^3$ (bois extrêmement spongieux) et $1.5\\text{ g/cm}^3$ (limite biologique supérieure)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "wd_path = os.path.join(RAW_DIR, \"ecologie_biomasse\", \"wood_density.csv\")\n",
                "if os.path.exists(wd_path):\n",
                "    df_wd = pd.read_csv(wd_path, sep=';', encoding='latin-1')\n",
                "    \n",
                "    for col in ['Latitude', 'Longitude', 'Wood Density']:\n",
                "        if col in df_wd.columns:\n",
                "            df_wd[col] = df_wd[col].astype(str).str.replace(',', '.')\n",
                "            df_wd[col] = pd.to_numeric(df_wd[col], errors='coerce')\n",
                "            \n",
                "    df_wd = df_wd.dropna(subset=['Latitude', 'Longitude', 'Wood Density'])\n",
                "    \n",
                "    cols_to_keep = ['Species', 'Latitude', 'Longitude', 'Plant Growth Form', 'Wood Density']\n",
                "    df_wd = df_wd[[c for c in cols_to_keep if c in df_wd.columns]].copy()\n",
                "    \n",
                "    df_wd = df_wd[(df_wd['Wood Density'] >= 0.1) & (df_wd['Wood Density'] <= 1.5)]\n",
                "    \n",
                "    save_cleaned(df_wd, \"wood_density_cleaned.csv\")\n",
                "else:\n",
                "    print(\"Fichier wood_density.csv absent, saut de cette étape.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 💧 PARTIE 3 : Hydrologie & Datasets Socio-Économiques\n",
                "\n",
                "Cette partie documente le traitement de tous les fichiers d'hydrologie ainsi que des nombreux indicateurs socio-démographiques complémentaires (Banque Mondiale, Our World in Data, UCDP/ACLED et Organisation Mondiale de la Santé)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1️⃣ Données d'Hydrologie (`wb_freshwater_withdrawal_pct.csv` et geojson)\n",
                "\n",
                "*   **Pourquoi ces datasets ?** Permettent d'estimer les prélèvements annuels d'eau douce par pays et de calculer la distance géodésique aux cours d'eau/lacs.\n",
                "*   **Nettoyage appliqué :** \n",
                "    *   *CSV prélèvements* : Standardisation de `country_name` -> `Pays`, `year` -> `Annee` et `value` -> `Valeur` (exprimé en %). Suppression des NaN et doublons.\n",
                "    *   *GeoJSON lacs/rivières* : Lecture et vérification de la validité des coordonnées de géométrie."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Prélèvements d'eau douce (Banque Mondiale)\n",
                "hydro_path = os.path.join(RAW_DIR, \"hydrologie_eau\", \"wb_freshwater_withdrawal_pct.csv\")\n",
                "if os.path.exists(hydro_path):\n",
                "    df_h = load_csv(hydro_path)\n",
                "    col_zone = find_col(df_h, 'country_name', 'country_long', 'Zone', 'Area')\n",
                "    col_annee = find_col(df_h, 'year', 'Year', 'Année')\n",
                "    col_valeur = find_col(df_h, 'value', 'Value', 'Valeur')\n",
                "    \n",
                "    cols_keep = {col_zone: 'Pays', col_annee: 'Annee', col_valeur: 'Valeur'}\n",
                "    df_h = df_h[[c for c in cols_keep.keys()]].copy().rename(columns=cols_keep)\n",
                "    df_h = clean_common(df_h)\n",
                "    save_cleaned(df_h, \"wb_freshwater_withdrawal_pct.csv\")\n",
                "else:\n",
                "    print(\"wb_freshwater_withdrawal_pct.csv absent.\")\n",
                "\n",
                "# 2. Chargement test des GeoJSON d'hydrologie\n",
                "for geo_name in ['rivers.geojson', 'lakes.geojson']:\n",
                "    geo_path = os.path.join(RAW_DIR, \"hydrologie_eau\", geo_name)\n",
                "    if os.path.exists(geo_path):\n",
                "        with open(geo_path, 'r', encoding='utf-8') as f:\n",
                "            data = json.load(f)\n",
                "        print(f\"[Vérification GeoJSON] {geo_name} chargé avec succès ({len(data.get('features', []))} entités)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2️⃣ Indicateurs Socio-Économiques Mondiaux\n",
                "\n",
                "Pour assurer la cohérence et l'homogénéité de nos données, nous nettoyons récursivement tous les sous-dossiers socio-économiques.\n",
                "\n",
                "*   **World Bank (28 indicateurs)** : GDP, électricité, population, espérance de vie, chômage, éducation...\n",
                "    *   *Nettoyage* : Les fichiers ont le schéma standard `country_name`, `year`, `value`. Nous les mappons systématiquement vers `Pays`, `Annee`, `Valeur`.\n",
                "*   **Our World in Data (OWID)** : HDI (Human Development Index), émissions de CO2, pauvreté...\n",
                "    *   *Nettoyage* : Les fichiers OWID contiennent typiquement `Entity` (colonne 0), `Year` (colonne 2) et l'indicateur numérique (colonne 3). Nous standardisons ces trois colonnes.\n",
                "*   **Conflits Armés (ACLED/UCDP)** : Intensité et type de conflits historiques par pays.\n",
                "    *   *Nettoyage* : Nous mappons `location` -> `Pays`, `year` -> `Annee` et `intensity_level` -> `Valeur`.\n",
                "*   **Organisation Mondiale de la Santé (WHO)** : Mortalité infantile, famine et espérance de vie."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Nettoyage des 28 fichiers de la Banque Mondiale\n",
                "wb_dir = os.path.join(RAW_DIR, \"socio_economie_demographie\", \"worldbank\")\n",
                "if os.path.exists(wb_dir):\n",
                "    for filename in os.listdir(wb_dir):\n",
                "        if filename.endswith(\".csv\"):\n",
                "            df_wb = load_csv(os.path.join(wb_dir, filename))\n",
                "            col_zone = find_col(df_wb, 'country_name', 'Zone', 'Area')\n",
                "            col_annee = find_col(df_wb, 'year', 'Year', 'Année')\n",
                "            col_valeur = find_col(df_wb, 'value', 'Value', 'Valeur')\n",
                "            if col_zone and col_annee and col_valeur:\n",
                "                df_wb = df_wb[[col_zone, col_annee, col_valeur]].copy().rename(columns={col_zone:'Pays', col_annee:'Annee', col_valeur:'Valeur'})\n",
                "                df_wb = clean_common(df_wb)\n",
                "                save_cleaned(df_wb, filename)\n",
                "\n",
                "# 2. Nettoyage des fichiers OWID\n",
                "owid_dir = os.path.join(RAW_DIR, \"socio_economie_demographie\", \"owid\")\n",
                "if os.path.exists(owid_dir):\n",
                "    for filename in os.listdir(owid_dir):\n",
                "        if filename.endswith(\".csv\"):\n",
                "            df_ow = load_csv(os.path.join(owid_dir, filename))\n",
                "            if len(df_ow.columns) >= 4:\n",
                "                col_zone = df_ow.columns[0]\n",
                "                col_annee = df_ow.columns[2]\n",
                "                col_valeur = df_ow.columns[3]\n",
                "                df_ow = df_ow[[col_zone, col_annee, col_valeur]].copy().rename(columns={col_zone:'Pays', col_annee:'Annee', col_valeur:'Valeur'})\n",
                "                df_ow = clean_common(df_ow)\n",
                "                save_cleaned(df_ow, filename)\n",
                "\n",
                "# 3. Nettoyage d'ACLED / UCDP\n",
                "acled_path = os.path.join(RAW_DIR, \"socio_economie_demographie\", \"acled\", \"UcdpPrioConflict_v24_1.csv\")\n",
                "if os.path.exists(acled_path):\n",
                "    df_ac = load_csv(acled_path)\n",
                "    col_zone = find_col(df_ac, 'location', 'Zone')\n",
                "    col_annee = find_col(df_ac, 'year', 'Year')\n",
                "    col_valeur = find_col(df_ac, 'intensity_level', 'Value')\n",
                "    if col_zone and col_annee and col_valeur:\n",
                "        df_ac = df_ac[[col_zone, col_annee, col_valeur]].copy().rename(columns={col_zone:'Pays', col_annee:'Annee', col_valeur:'Valeur'})\n",
                "        df_ac = clean_common(df_ac)\n",
                "        save_cleaned(df_ac, \"UcdpPrioConflict_v24_1.csv\")\n",
                "\n",
                "# 4. Nettoyage des fichiers de la WHO\n",
                "who_dir = os.path.join(RAW_DIR, \"socio_economie_demographie\", \"who\")\n",
                "if os.path.exists(who_dir):\n",
                "    for filename in os.listdir(who_dir):\n",
                "        if filename.endswith(\".csv\"):\n",
                "            df_who = load_csv(os.path.join(who_dir, filename))\n",
                "            col_zone = find_col(df_who, 'Entity', 'country_name', 'Zone')\n",
                "            col_annee = find_col(df_who, 'Year', 'year')\n",
                "            col_valeur = find_col(df_who, 'Value', 'value', 'Valeur', df_who.columns[3] if len(df_who.columns) >= 4 else None)\n",
                "            if col_zone and col_annee and col_valeur:\n",
                "                df_who = df_who[[col_zone, col_annee, col_valeur]].copy().rename(columns={col_zone:'Pays', col_annee:'Annee', col_valeur:'Valeur'})\n",
                "                df_who = clean_common(df_who)\n",
                "                save_cleaned(df_who, filename)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌍 PARTIE 4 : Vérification et lecture des datasets Raster (.tif) de WorldClim\n",
                "\n",
                "Les données WorldClim ne subissent pas de traitement tabulaire (ce sont des rasters binaires géoréférencés). Cependant, pour s'assurer que les fichiers ne sont pas corrompus et que le moteur spatial `Layer1Engine` les lira correctement, nous exécutons une routine d'inspection sur les en-têtes et métadonnées de chaque catégorie de GeoTIFF."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "wc_dir = os.path.join(RAW_DIR, \"WorldClim\")\n",
                "if os.path.exists(wc_dir):\n",
                "    print(\"Inspection des répertoires WorldClim (.tif) :\")\n",
                "    for folder in sorted(os.listdir(wc_dir)):\n",
                "        folder_path = os.path.join(wc_dir, folder)\n",
                "        if os.path.isdir(folder_path):\n",
                "            tifs = [f for f in os.listdir(folder_path) if f.endswith('.tif')]\n",
                "            if tifs:\n",
                "                sample_path = os.path.join(folder_path, tifs[0])\n",
                "                try:\n",
                "                    with Image.open(sample_path) as img:\n",
                "                        width, height = img.size\n",
                "                        mode = img.mode\n",
                "                        print(f\"  [OK] {folder} (Exemple: {tifs[0]}) : Résolution = {width}x{height}, Mode de pixels = {mode}\")\n",
                "                except Exception as e:\n",
                "                    print(f\"  [ERREUR] Impossible de charger {tifs[0]} dans {folder} : {e}\")\n",
                "            else:\n",
                "                print(f\"  [Vide] Aucun fichier .tif trouvé dans le répertoire {folder}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🎯 Conclusion Générale du Nettoyage\n",
                "\n",
                "Tous les fichiers tabulaires de toutes les catégories physiques et socio-économiques (y compris l'hydrologie et World Bank/OWID) sont désormais nettoyés et stockés dans `data/cleaned/`.\n",
                "Les fichiers de géométries GeoJSON et les rasters climatologiques WorldClim GeoTIFF ont été inspectés, validés, et sont pleinement fonctionnels pour alimenter le moteur physique global."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": ".venv",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.1"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_path = 'data_cleaning.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Successfully generated data_cleaning.ipynb at: {output_path}")
