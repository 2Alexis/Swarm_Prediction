# 🩸 Couche 2 — Le Sang

> Démographie + Épidémiologie + Catastrophes humaines.
> **Principe central** : prédire les gens à partir UNIQUEMENT de la terre (Couche 1).
> Les variables socio-éco (PIB, HDI, urbanisation...) sont blacklistées pour rester rigoureux.

## Sous-couches

### 2A. Démographie (7 cibles)
| Cible | R² |
|---|---|
| Natalité (‰) | **0.95** |
| Taux de fécondité | **0.95** |
| Mortalité infantile (<5 ans) | **0.82** |
| Espérance de vie | 0.75 |
| Croissance démographique | 0.69 |
| Mortalité brut (‰) | 0.58 |
| Migration nette | 0.08 |

### 2B. Santé / Épidémiologie (1 cible)
| Cible | R² |
|---|---|
| Stunting (retard de croissance) | 0.67 |

### 2C. Catastrophes humaines (2 cibles)
| Cible | R² |
|---|---|
| Décès dus aux catastrophes (log) | 0.54 |
| Personnes affectées (log) | 0.53 |

## La grande victoire scientifique
🌍 **Natalité prédite à R²=0.95 sans aucune variable socio-éco** — uniquement à partir
du climat, du sol, de la géographie, des rendements agricoles et du cluster climatique.

Ça démontre que **la démographie est largement structurée par l'environnement physique** :
- Plus on monte vers les tropiques humides → plus la natalité est forte
- Plus le climat est tempéré et l'agriculture productive → plus la mortalité infantile baisse
- L'humidité du sol racinaire (NASA GWETROOT) prédit la fécondité

## Fichiers
```
couche2_sang/
├── config.py                      # Cibles + anti-leak strict env→démo
├── train.py                       # Entraînement complet
├── modelisation_couche2.ipynb     # Notebook interactif
├── models/
│   └── best_target_*.joblib       # 10 modèles principaux + 60 par cluster
└── reports/
    └── results.csv                # R² par stratégie
```

## Utilisation
```bash
# Entraîner toute la couche (~3 min)
python couche2_sang/train.py

# Notebook interactif
jupyter notebook couche2_sang/modelisation_couche2.ipynb
```

## Chargement d'un modèle
```python
import joblib
data = joblib.load('couche2_sang/models/best_target_birth_rate.joblib')
pipe = data['pipe']
features = data['features']

# Prédire :
prediction = pipe.predict(new_data[features])
```

## Pourquoi pas la migration nette ?
R² = 0.08 — quasiment inmodélisable depuis l'environnement.
La migration est dominée par :
- Politiques d'immigration (Couche 4 — Le Chaos)
- Conflits récents (Couche 4)
- Différentiels économiques (Couche 3 — Le Moteur)

Cette cible attendra les couches supérieures pour être prédite correctement.

## Sources de données utilisées
- **World Bank** : Birth_Rate, Death_Rate, Child_Mort, Life_Exp, Pop_Growth, Net_Migration, Fertility, stunting
- **OWID** : décès et personnes affectées par catastrophes (mirror EM-DAT)
- **OWID** : fertility rate, undernourishment
- **FAO** : global hunger index
- **WHO** : air quality
