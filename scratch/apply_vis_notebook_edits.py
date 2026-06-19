import json

vis_markdown_12 = [
    "## 12. \U0001f30b Risques Physiques & Infrastructures Énergétiques\n",
    "\n",
    "*   **Volcans Actifs** : Cartographie des volcans mondiaux selon leur type et leur élévation.\n",
    "*   **Séismes Majeurs** : Événements sismiques de magnitude supérieure à 0, cartographiés par magnitude (taille) et profondeur (couleur).\n",
    "*   **Centrales Électriques** : Répartition mondiale et capacité cumulée en mégawatts (MW) selon le type de combustible principal.\n"
]

vis_code_12 = [
    "# Chargement des risques physiques\n",
    "df_v = pd.read_csv(\"data/cleaned/volcanoes_cleaned.csv\")\n",
    "df_eq = pd.read_csv(\"data/cleaned/earthquakes_cleaned.csv\")\n",
    "df_pp = pd.read_csv(\"data/cleaned/global_power_plants_cleaned.csv\")\n",
    "\n",
    "fig, axes = plt.subplots(2, 2, figsize=(18, 14))\n",
    "fig.suptitle('\u2699\ufe0f Section 12 : Risques Physiques et Centrales \u00c9lectriques', fontsize=18, weight='bold')\n",
    "\n",
    "# 1. Volcans actifs\n",
    "sns.scatterplot(data=df_v, x='longitude', y='latitude', hue='primary_volcano_type', size='elevation', \n",
    "                sizes=(10, 200), alpha=0.7, palette='Set1', ax=axes[0,0])\n",
    "axes[0,0].set_title(\"Volcans Actifs Mondiaux par Type et Altitude\", fontsize=12, weight='bold')\n",
    "axes[0,0].set_xlabel(\"Longitude\")\n",
    "axes[0,0].set_ylabel(\"Latitude\")\n",
    "axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')\n",
    "\n",
    "# 2. S\u00e9ismes Majeurs (Couleur = Profondeur, Taille = Magnitude)\n",
    "sc = axes[0,1].scatter(df_eq['longitude'], df_eq['latitude'], c=df_eq['depth'], s=df_eq['mag']**2 * 2, \n",
    "                       cmap='magma', alpha=0.5)\n",
    "cbar = fig.colorbar(sc, ax=axes[0,1])\n",
    "cbar.set_label(\"Profondeur de l'hypocentre (km)\")\n",
    "axes[0,1].set_title(\"S\u00e9ismes Majeurs (Taille = Magnitude, Couleur = Profondeur)\", fontsize=12, weight='bold')\n",
    "axes[0,1].set_xlabel(\"Longitude\")\n",
    "axes[0,1].set_ylabel(\"Latitude\")\n",
    "\n",
    "# 3. Capacit\u00e9 des Centrales \u00c9lectriques par Combustible\n",
    "pp_cap = df_pp.groupby('primary_fuel')['capacity_mw'].sum().reset_index().sort_values('capacity_mw', ascending=False)\n",
    "sns.barplot(data=pp_cap, x='capacity_mw', y='primary_fuel', palette='viridis', ax=axes[1,0])\n",
    "axes[1,0].set_title(\"Capacit\u00e9 \u00c9nerg\u00e9tique Globale par Type de Combustible (MW)\", fontsize=12, weight='bold')\n",
    "axes[1,0].set_xlabel(\"Capacit\u00e9 Cumul\u00e9e (MW)\")\n",
    "axes[1,0].set_ylabel(\"Combustible Principal\")\n",
    "\n",
    "# 4. Distribution G\u00e9ographique des Centrales \u00c9lectriques\n",
    "sns.scatterplot(data=df_pp, x='longitude', y='latitude', hue='primary_fuel', size='capacity_mw',\n",
    "                sizes=(5, 300), alpha=0.5, palette='tab20', ax=axes[1,1])\n",
    "axes[1,1].set_title(\"Cartographie Globale des Centrales \u00c9lectriques\", fontsize=12, weight='bold')\n",
    "axes[1,1].set_xlabel(\"Longitude\")\n",
    "axes[1,1].set_ylabel(\"Latitude\")\n",
    "axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

vis_markdown_13 = [
    "## 13. \U0001f48e Ressources Minérales & Énergétiques Fossiles\n",
    "\n",
    "*   **Minerais Globaux (MRDS)** : Cartographie des gisements de métaux et ressources minérales de l'USGS.\n",
    "*   **Tracker Charbon / Fer** : Emplacement des grandes exploitations de charbon actif et de minerai de fer.\n",
    "*   **Pétrole & Gaz** : Répartition spatiale des champs pétrolifères et gaziers en cours d'exploitation.\n"
]

vis_code_13 = [
    "# Chargement des ressources\n",
    "df_mrds = pd.read_csv(\"data/cleaned/mrds_cleaned.csv\", low_memory=False)\n",
    "df_coal = pd.read_csv(\"data/cleaned/coal_mines_cleaned.csv\")\n",
    "df_iron = pd.read_csv(\"data/cleaned/iron_mines_cleaned.csv\")\n",
    "df_oil = pd.read_csv(\"data/cleaned/oil_gas_cleaned.csv\")\n",
    "\n",
    "fig, axes = plt.subplots(2, 2, figsize=(18, 14))\n",
    "fig.suptitle('\ud83d\udc8e Section 13 : Ressources Min\u00e9rales & \u00c9nerg\u00e9tiques', fontsize=18, weight='bold')\n",
    "\n",
    "# 1. Top Commodit\u00e9s MRDS\n",
    "top_comm = df_mrds['Commodite'].value_counts().head(10).reset_index()\n",
    "sns.barplot(data=top_comm, x='count', y='Commodite', palette='rocket', ax=axes[0,0])\n",
    "axes[0,0].set_title(\"Top 10 des Commodit\u00e9s Min\u00e9rales R\u00e9pertori\u00e9es (MRDS)\", fontsize=12, weight='bold')\n",
    "axes[0,0].set_xlabel(\"Nombre de Sites\")\n",
    "axes[0,0].set_ylabel(\"Commodit\u00e9\")\n",
    "\n",
    "# 2. Cartographie des gisements de Fer et Charbon vs MRDS\n",
    "df_mrds_sample = df_mrds.dropna(subset=['Latitude', 'Longitude'])\n",
    "df_mrds_sample = df_mrds_sample.sample(min(15000, len(df_mrds_sample)), random_state=42)\n",
    "sns.scatterplot(data=df_mrds_sample, x='Longitude', y='Latitude', \n",
    "                color='gray', alpha=0.1, s=1, label='Gisements MRDS', ax=axes[0,1])\n",
    "sns.scatterplot(data=df_iron, x='Longitude', y='Latitude', color='red', alpha=0.8, s=15, label='Mines de Fer (Tracker)', ax=axes[0,1])\n",
    "sns.scatterplot(data=df_coal, x='Longitude', y='Latitude', color='black', alpha=0.6, s=10, label='Mines de Charbon (Tracker)', ax=axes[0,1])\n",
    "axes[0,1].set_title(\"Cartographie des Gisements de Fer et Charbon vs MRDS\", fontsize=12, weight='bold')\n",
    "axes[0,1].set_xlabel(\"Longitude\")\n",
    "axes[0,1].set_ylabel(\"Latitude\")\n",
    "axes[0,1].legend(loc='lower left')\n",
    "\n",
    "# 3. Capacit\u00e9 des Mines de Charbon par Statut\n",
    "coal_status = df_coal.groupby('Statut')['Capacite_Mtpa'].sum().reset_index().sort_values('Capacite_Mtpa', ascending=False)\n",
    "sns.barplot(data=coal_status, x='Capacite_Mtpa', y='Statut', palette='copper', ax=axes[1,0])\n",
    "axes[1,0].set_title(\"Capacit\u00e9 Totale des Mines de Charbon par Statut (Mtpa)\", fontsize=12, weight='bold')\n",
    "axes[1,0].set_xlabel(\"Capacit\u00e9 Cumul\u00e9e (Mtpa)\")\n",
    "axes[1,0].set_ylabel(\"Statut de la Mine\")\n",
    "\n",
    "# 4. Cartographie des Champs de P\u00e9trole et Gaz\n",
    "sns.scatterplot(data=df_oil, x='Longitude', y='Latitude', hue='Type_Combustible', style='Status',\n",
    "                alpha=0.6, s=15, palette='plasma', ax=axes[1,1])\n",
    "axes[1,1].set_title(\"R\u00e9partition des Gisements de P\u00e9trole & Gaz\", fontsize=12, weight='bold')\n",
    "axes[1,1].set_xlabel(\"Longitude\")\n",
    "axes[1,1].set_ylabel(\"Latitude\")\n",
    "axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

vis_markdown_14 = [
    "## 14. \U0001f33f Données Écologiques & Hydrologie\n",
    "\n",
    "*   **Densité de Bois Écologique** : Analyse de la répartition des mesures de densité de bois par type morphologique de plantes (arbre, buisson, etc.).\n",
    "*   **Hydrologie (Banque Mondiale)** : Évolution temporelle des prélèvements annuels d'eau douce renouvelable pour le top 10 des pays préleveurs.\n"
]

vis_code_14 = [
    "# Chargement \u00e9cologie et pr\u00e9l\u00e8vements d'eau\n",
    "df_wd = pd.read_csv(\"data/cleaned/wood_density_cleaned.csv\")\n",
    "df_hydro = pd.read_csv(\"data/cleaned/wb_freshwater_withdrawal_pct.csv\")\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
    "fig.suptitle('\ud83c\udf3f Section 14 : \u00c9cologie de la Biomasse & Hydrologie', fontsize=16, weight='bold')\n",
    "\n",
    "# 1. Distribution de la Densit\u00e9 de Bois\n",
    "sns.histplot(data=df_wd, x='Wood Density', hue='Plant Growth Form', kde=True, multiple='stack', palette='crest', ax=axes[0])\n",
    "axes[0].set_title(\"Distribution de la Densit\u00e9 de Bois par Forme de Croissance\", fontsize=12, weight='bold')\n",
    "axes[0].set_xlabel(\"Densit\u00e9 de Bois (g/cm\u00b3)\")\n",
    "axes[0].set_ylabel(\"Nombre d'esp\u00e8ces\")\n",
    "\n",
    "# 2. Pr\u00e9l\u00e8vements d'eau douce - \u00c9volution temporelle pour les 10 pays consommant le plus\n",
    "top_hydro_pays = df_hydro[df_hydro['Annee'] == 2020].sort_values('Valeur', ascending=False).head(10)['Pays'].tolist()\n",
    "df_hydro_top = df_hydro[df_hydro['Pays'].isin(top_hydro_pays)]\n",
    "sns.lineplot(data=df_hydro_top, x='Annee', y='Valeur', hue='Pays', marker='o', palette='tab10', ax=axes[1])\n",
    "axes[1].set_title(\"\u00c9volution des Pr\u00e9l\u00e8vements d'Eau Douce (% des ressources)\", fontsize=12, weight='bold')\n",
    "axes[1].set_xlabel(\"Ann\u00e9e\")\n",
    "axes[1].set_ylabel(\"Pr\u00e9l\u00e8vement (% de l'eau renouvelable)\")\n",
    "axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

vis_markdown_15 = [
    "## 15. \U0001f4c8 Relations Socio-Démographiques & Indicateurs de Développement\n",
    "\n",
    "Visualisation croisée des indicateurs de développement humain et de santé :\n",
    "*   **Espérance de Vie vs Richesse (PIB/habitant)** : Corrélation non-linéaire (échelle logarithmique) pondérée par le taux de survie infantile et l'indice de développement humain (IDH).\n",
    "*   **Matrice de Corrélation** : Relations directes entre richesse (PIB), santé (espérance de vie, mortalité infantile) et incidence de maladies (paludisme/malaria).\n"
]

vis_code_15 = [
    "# Chargement et fusion des donn\u00e9es de d\u00e9veloppement\n",
    "df_gdp = pd.read_csv(\"data/cleaned/wb_gdp_per_capita.csv\")\n",
    "df_life = pd.read_csv(\"data/cleaned/wb_life_expectancy.csv\")\n",
    "df_child = pd.read_csv(\"data/cleaned/wb_child_mortality.csv\")\n",
    "df_malaria = pd.read_csv(\"data/cleaned/wb_malaria_incidence.csv\")\n",
    "df_hdi = pd.read_csv(\"data/cleaned/owid_hdi.csv\")\n",
    "\n",
    "# Fusion master des indicateurs de d\u00e9veloppement\n",
    "df_dev = pd.merge(df_gdp, df_life, on=['Pays', 'Annee'], suffixes=('_gdp', '_life')).rename(columns={'Valeur_gdp': 'GDP_pc', 'Valeur_life': 'Life_Exp'})\n",
    "df_dev = pd.merge(df_dev, df_child, on=['Pays', 'Annee']).rename(columns={'Valeur': 'Child_Mort'})\n",
    "df_dev = pd.merge(df_dev, df_hdi, on=['Pays', 'Annee']).rename(columns={'Valeur': 'HDI'})\n",
    "df_dev = pd.merge(df_dev, df_malaria, on=['Pays', 'Annee'], how='left').rename(columns={'Valeur': 'Malaria_Incidence'})\n",
    "\n",
    "# Donn\u00e9es pour l'ann\u00e9e 2015\n",
    "df_dev_2015 = df_dev[df_dev['Annee'] == 2015].dropna(subset=['GDP_pc', 'Life_Exp', 'Child_Mort', 'HDI'])\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
    "fig.suptitle('\ud83d\udcc8 Section 15 : Corr\u00e9lations Socio-D\u00e9mographiques & Sant\u00e9 (2015)', fontsize=16, weight='bold')\n",
    "\n",
    "# 1. PIB vs Esp\u00e9rance de vie color\u00e9 par l'IDH (IDH) et taille invers\u00e9e par la mortalit\u00e9 infantile\n",
    "sc = axes[0].scatter(df_dev_2015['GDP_pc'], df_dev_2015['Life_Exp'], c=df_dev_2015['HDI'], \n",
    "                     s=150 - df_dev_2015['Child_Mort'], cmap='viridis', alpha=0.8, edgecolor='w')\n",
    "axes[0].set_xscale('log')\n",
    "axes[0].set_title(\"Esp\u00e9rance de Vie vs PIB/habitant (Taille invers\u00e9e = Mort. Infantile)\", fontsize=12, weight='bold')\n",
    "axes[0].set_xlabel(\"PIB par Habitant (USD, \u00c9chelle Log)\")\n",
    "axes[0].set_ylabel(\"Esp\u00e9rance de Vie (Ann\u00e9es)\")\n",
    "cbar = fig.colorbar(sc, ax=axes[0])\n",
    "cbar.set_label(\"Index de D\u00e9veloppement Humain (HDI)\")\n",
    "\n",
    "# 2. Matrice de corr\u00e9lation d\u00e9veloppement et sant\u00e9\n",
    "corr_dev = df_dev_2015[['GDP_pc', 'Life_Exp', 'Child_Mort', 'HDI', 'Malaria_Incidence']].corr()\n",
    "sns.heatmap(corr_dev, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, ax=axes[1])\n",
    "axes[1].set_title(\"Matrice de Corr\u00e9lation - D\u00e9veloppement et Sant\u00e9\", fontsize=12, weight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

notebook_path = 'data_visualization.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Insert the cells at index 30
cells_to_insert = [
    {"cell_type": "markdown", "metadata": {}, "source": vis_markdown_12},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": vis_code_12},
    {"cell_type": "markdown", "metadata": {}, "source": vis_markdown_13},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": vis_code_13},
    {"cell_type": "markdown", "metadata": {}, "source": vis_markdown_14},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": vis_code_14},
    {"cell_type": "markdown", "metadata": {}, "source": vis_markdown_15},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": vis_code_15}
]

for offset, cell in enumerate(cells_to_insert):
    nb['cells'].insert(30 + offset, cell)

# Save notebook safely with ensure_ascii=True to avoid Windows surrogate errors
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print(f"Successfully inserted {len(cells_to_insert)} new cells into data_visualization.ipynb.")
