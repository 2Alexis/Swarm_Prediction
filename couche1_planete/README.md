# 🌍 Couche 1 — La Planète (V12)

> Environnement physique + Agriculture (par culture spécifique) + Élevage + Écologie + Énergie + Émissions.
> Les sorties de cette couche servent d'inputs à la **Couche 2 — Le Sang**.

## 📊 62 cibles dans 10 sous-couches

### 1A. Céréales & racines (2)
| Cible | R² |
|---|---|
| Rendement racines/tubercules | 0.54 |
| Rendement céréales | 0.59 |

### 1B. Oléagineux par culture (8) — *remplace l'agrégat R²=0.09*
| Cible | R² |
|---|---|
| **Colza** | **0.58** |
| Cotton | 0.28 |
| Arachide | 0.22 |
| Coco | 0.15 |
| Soja | 0.06 |
| Sésame | 0.11 |
| Olives | 0.02 |
| Tournesol | -0.32 |

### 1C. Fruits par culture (16) — *remplace l'agrégat R²=0.14*
| Cible | R² |
|---|---|
| **Pomme** | **0.34** |
| Fraise | 0.30 |
| Citron | 0.27 |
| Raisin | 0.21 |
| Banane | 0.15 |
| Ananas | 0.11 |
| Watermelon | 0.09 |
| Orange | 0.04 |
| Pêche | 0.02 |
| Avocat | -0.02 |
| Date | -0.11 |
| Apricot | -0.15 |
| Pear | -0.20 |
| Plum | -0.08 |
| Cherry | -0.35 |
| Mango | -1.32 |

### 1D. Légumes par culture (9) — *remplace l'agrégat R²=0.44*
| Cible | R² |
|---|---|
| **Concombre** | **0.59** |
| **Tomate** | **0.53** |
| Pomme de terre | 0.43 |
| Aubergine | 0.34 |
| Laitue | 0.22 |
| Carotte | 0.10 |
| Cauliflower | 0.04 |
| Cabbage | 0.00 |
| Oignon | -0.11 |

### 1E. Légumineuses par culture (3) — *remplace l'agrégat R²=-0.08*
| Cible | R² |
|---|---|
| **Pois sec** | **0.45** |
| Haricot sec | 0.22 |
| Pois chiche | -0.54 |

### 1F. Élevage (8)
| Cible | R² |
|---|---|
| **Production œufs** | **0.85** |
| **Rendement lait** | **0.83** |
| Aquaculture | 0.62 |
| Rendement œufs | 0.48 |
| Carcasse poulet | 0.34 |
| Carcasse bovine | 0.32 |
| Carcasse porc | 0.21 |
| Carcasse ovin/caprin | 0.19 |

### 1G. Environnement physique (5)
| Cible | R² |
|---|---|
| **Anomalie thermique** | **0.83** |
| **Accès eau potable** | **0.81** |
| Stress hydrique | 0.66 |
| Dégradation du sol | 0.61 |
| Humidité sol racinaire | 0.51 |

### 1H. Écologie (2)
| Cible | R² |
|---|---|
| **% forêt national** | **0.88** |
| Perte couvert arboré | 0.59 |

### 1I. Émissions atmosphériques (3)
| Cible | R² |
|---|---|
| **Émissions CH4** | **0.95** |
| **Émissions N2O** | **0.96** |
| **Émissions CO2** | **0.73** |

### 1J. Énergie (6)
| Cible | R² |
|---|---|
| **Production charbon** | **0.90** |
| **Production pétrole** | **0.85** |
| **Production gaz** | **0.78** |
| **Hydroélectrique** | **0.78** |
| **Génération éolienne** | **0.75** |
| **Solaire** | **0.73** |

## 🏆 Top 15 cibles Couche 1 V12 (R² ≥ 0.7)

| # | Cible | R² | Sous-couche |
|---|---|---|---|
| 1 | Émissions N2O | 0.958 | Émissions |
| 2 | Émissions CH4 | 0.950 | Émissions |
| 3 | Production charbon | 0.895 | Énergie |
| 4 | % forêt national | 0.883 | Écologie |
| 5 | Production œufs | 0.853 | Élevage |
| 6 | Production pétrole | 0.853 | Énergie |
| 7 | Rendement lait | 0.827 | Élevage |
| 8 | Anomalie thermique | 0.828 | Environnement |
| 9 | Accès eau potable | 0.806 | Environnement |
| 10 | Hydroélectrique | 0.781 | Énergie |
| 11 | Production gaz | 0.779 | Énergie |
| 12 | Génération éolienne | 0.753 | Énergie |
| 13 | Émissions CO2 | 0.732 | Émissions |
| 14 | Solaire | 0.728 | Énergie |

## 📁 Structure
```
couche1_planete/
├── README.md
├── config.py                     # 62 cibles + anti-leak strict
├── fetch_couche1_extra.py        # téléchargement initial OWID/FAO
├── fetch_religion_meat.py        # Pew + meat consumption per type
├── religion_static.py            # Données Pew 2010 statiques (163 pays)
├── enrich_couche1.py             # V8 → V9 (élevage + énergie + émissions)
├── enrich_couche1_v2.py          # V9 → V10 (cultures spé + religion + meat)
├── compute_suitability.py        # V10 → V11 (EcoCrop suitability scores)
├── enrich_v12.py                 # V11 → V12 (suitability adoucie + stress climat)
├── train.py                      # entraînement (PerCluster désactivé)
├── modelisation_couche1.ipynb    # notebook interactif
├── models/                       # 62 modèles best_target_*.joblib
└── reports/results.csv           # tableau complet
```

## 🚀 Utilisation

```bash
# Rebuild complet (~20 min)
python couche1_planete/fetch_couche1_extra.py
python couche1_planete/enrich_couche1.py
python couche1_planete/fetch_religion_meat.py
python couche1_planete/enrich_couche1_v2.py
python couche1_planete/compute_suitability.py
python couche1_planete/enrich_v12.py

# Train (~10 min)
python couche1_planete/train.py

# Ou via notebook
jupyter notebook couche1_planete/modelisation_couche1.ipynb
```

## 💡 Charger un modèle
```python
import joblib
data = joblib.load('couche1_planete/models/best_target_yield_tomato.joblib')
pipe = data['pipe']
features = data['features']
prediction = pipe.predict(new_data[features])
```

## 📚 Sources V12
- **WorldClim BIO 1-19** : T°, précipitations, saisonnalité
- **NASA POWER** : T°, précipitations, humidité sol annuelles 1990-2024
- **Berkeley Earth** : anomalies T° par pays 1743-2020
- **FAO** : rendements végétaux par culture, élevage (lait, viandes, œufs), intrants
- **FAO EcoCrop** : T_opt/P_opt par culture pour suitability scores
- **OWID** : émissions, énergie, aquaculture, eau, meat consumption per type
- **Pew Research 2010** : composition religieuse par pays
- **Hansen/OWID** : couverture forestière, déforestation
- **World Bank** : surface arable, irrigation, eau potable
- **NOAA** : ENSO, NAO, AMO, PDO, AO, CO2 Mauna Loa

## 🚧 Limites connues
- **Mango, Pois chiche, Cerise, Pear, Sunflower** : R² négatif → ces cultures dépendent trop des variétés/techniques spécifiques pour être prédites par climat seul
- **Carcasses animales** : R² 0.2-0.3, plafond gratuit (manque données race + intensification)
- **Migration nette** : déplacée en Couche 4 (politique)
- **Marine protégée** : déplacée en Couche 4 (politique)
