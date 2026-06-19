# 🌍 Swarm_Prediction — De la terre aux gens

> **La question du projet :** *dans quelle mesure l'**environnement agricole** d'un pays
> (sa terre, son climat, ses récoltes) détermine-t-il la **santé et la démographie** de
> sa population (natalité, mortalité, malnutrition) ?*

Le projet relie **deux jeux de données réels** par une clé commune — **`(Pays, Année)`** —
pour raconter une seule histoire : **la terre nourrit les gens**. On ne génère aucun
chiffre au hasard : chaque variable vient d'une source publique (FAO, World Bank,
WorldClim, Project Tycho), et chaque modèle est évalué honnêtement.

```
   COUCHE 1 — LA TERRE                         COUCHE 2 — LES GENS
   environnement & agriculture                 démographie & santé
   FAO · WorldClim · géologie                  World Bank · FAO · Project Tycho
   rendement, sols, climat, biomes             natalité, mortalité, malnutrition
            └──────────────┐         clé        ┌──────────────┘
                           ▼   (Pays, Année)    ▼
                    couche1_x_couche2.ipynb  ←  LE FIL NARRATIF
              « le rendement agricole explique la malnutrition »
```

---

## 🎯 Le résultat principal

Une fois les deux couches jointes (183 pays, 2010–2024) :

- **Corrélation rendement agricole ↔ malnutrition = −0.45** : plus un pays produit,
  moins sa population souffre de la faim.
- Un modèle prédisant la **malnutrition** uniquement à partir de variables
  **agro-environnementales** (Couche 1) obtient **R² ≈ 0.17 sur des pays jamais vus** —
  la terre explique une part réelle de la santé des populations.

---

## 🧱 Couche 1 — la terre (`modelisation_9_cibles.ipynb`)

9 variables, **deux natures différentes** (c'est la clé de la rigueur du projet) :

| Famille | Cibles | Origine | Traitement |
|---|---|---|---|
| **① Mesurées** | `yield`, `water_stress`, `soil_degradation`, `thermal_anomaly` | datasets **réels** (FAO, World Bank) | **modèle ML** (on *prédit*) |
| **② Dérivées** | `npp`, `fauna_density`, `parasite_vsi`, `flood_risk`, `erosion_prob` | **formules physiques** (`build_dataset.py`) | **calcul direct** (on ne prédit pas) |

**R² honnêtes** (évaluation **par pays** — voir Méthodologie) : `yield` 0.69 · `soil` 0.53 ·
`thermal` 0.33 · `water_stress` −0.09. Les cibles dérivées ne reçoivent pas de score ML
mais un *contrôle de cohérence* (un modèle proxy retrouve-t-il la formule ?).

Pipeline : `clean_data.py` → `build_dataset.py` (+ `layer1_engine.py`) → `dataset_final_modelisation.csv` → notebook.

## 🩸 Couche 2 — les gens (`couche2/`)

Démographie & santé, keyé sur `(Pays, Année)` :
`download_data.py` → `clean_merge.py` → `dataset_couche2.csv` (≈ 14 000 lignes, 217 pays,
1960–2025 : natalité, fécondité, mortalité infantile/<5/brute, densité, population,
malnutrition FAO, migration). Détails : `couche2/README.md` et `couche2/SOURCES_DONNEES.txt`.

## 🧵 Le lien (`couche1_x_couche2.ipynb`)

Jointure des deux couches sur `(Pays, Année)` (conversion ISO alpha-2 → alpha-3), puis :
heatmap des corrélations Couche 1 × Couche 2, le scatter *rendement → malnutrition*, et
le modèle inter-couches. **C'est le notebook qui « relie tout ».**

---

## 🔬 Méthodologie (ce qui rend les résultats fiables)

- **Split par pays** (`GroupShuffleSplit` sur l'ISO) et **non temporel** : les variables
  physiques étant constantes par pays, un split par année laisse le modèle *mémoriser*
  chaque pays (faux R² = 1.0). En testant sur des **pays jamais vus**, le R² reflète la
  vraie capacité de généralisation.
- **Mesuré vs calculé** : on ne « prédit » pas une variable qui est déjà une formule.
- **Données reproductibles, hors Git** (`data/` est ignoré) : tout se régénère via les
  scripts.

## 🗂️ Structure

```
build_dataset.py / layer1_engine.py / clean_data.py   Couche 1 : collecte → dataset
modelisation_9_cibles.ipynb                           Couche 1 : modèles (4 ML + 5 dérivées)
couche2/                                              Couche 2 : pipeline démographie/santé
couche1_x_couche2.ipynb                               LE LIEN : jointure + modèle inter-couches
models/                                               4 modèles ML entraînés (Couche 1)
reports/                                              figures + tableau de résultats
EDA.ipynb / data_*.ipynb                              exploration / nettoyage (Couche 1)
```

## ▶️ Lancer

```bash
pip install -r requirements.txt
# Couche 1
python build_dataset.py                       # construit data/cleaned/dataset_final_modelisation.csv
jupyter notebook modelisation_9_cibles.ipynb  # modèles + tableau de résultats
# Couche 2
python couche2/download_data.py && python couche2/clean_merge.py
# Le lien
jupyter notebook couche1_x_couche2.ipynb      # l'histoire : terre → gens
```

---

*Données 100 % réelles et publiques. Sources : FAO (FAOSTAT), World Bank, WorldClim,
Project Tycho, OWID. Voir `couche2/SOURCES_DONNEES.txt`.*
