# 🩸 Couche 2 — Le Sang : Démographie & Épidémiologie (pipeline de données)

> Collecter → cleaner → **fusionner** des données **réelles** de démographie et d'épidémiologie
> en un seul dataset keyé sur `(Pays, Année)`, puis l'explorer.
> Même logique que la Couche 1 (FAO) → les deux datasets se joindront sur `(Pays, Année)`.

## Pipeline

```
download_data.py          clean_merge.py              exploration.ipynb
   COLLECTE        →      NETTOYAGE + FUSION     →        EXPLORATION (EDA)
data/couche2/*.csv     data/couche2/dataset_couche2.csv     graphiques
```

| Étape | Commande (depuis la racine) | Sortie |
|-------|------------------------------|--------|
| 1. Collecte | `python couche2/download_data.py` | `data/couche2/*.csv` (World Bank, FAO, Project Tycho) |
| 2. Clean + fusion | `python couche2/clean_merge.py` | `data/couche2/dataset_couche2.csv` |
| 3. Exploration | `jupyter notebook couche2/exploration.ipynb` | graphiques EDA |

## Fichiers du dossier `couche2/`

| Fichier | Rôle |
|---------|------|
| `download_data.py` | Télécharge les vraies données depuis les API officielles. |
| `clean_merge.py` | Nettoie chaque indicateur et les fusionne en un seul dataset. |
| `exploration.ipynb` | Analyse exploratoire (complétude, distributions, corrélations). |
| `SOURCES_DONNEES.txt` | Toutes les sources réelles, une par mécanique (codes, liens). |
| `README.md` | Ce document. |

## Le dataset fusionné — `data/couche2/dataset_couche2.csv`

**~14 000 lignes × 13 colonnes · 217 pays · 1960–2025** (hors Git, reproductible).

| Colonne | Indicateur réel | Source | Code |
|---------|------------------|--------|------|
| `Code_Pays`, `Pays`, `Region`, `Annee` | clé + région | World Bank | — |
| `Natalite_pour1000` | Taux de natalité brut | World Bank | `SP.DYN.CBRT.IN` |
| `Fecondite_enf_par_femme` | Fécondité totale | World Bank | `SP.DYN.TFRT.IN` |
| `MortInfantile_pour1000` | Mortalité infantile (<1 an) | World Bank | `SP.DYN.IMRT.IN` |
| `MortMoins5ans_pour1000` | Mortalité <5 ans | World Bank | `SH.DYN.MORT` |
| `MortBrute_pour1000` | Mortalité brute | World Bank | `SP.DYN.CDRT.IN` |
| `Densite_hab_km2` | Densité de population | World Bank | `EN.POP.DNST` |
| `Population` | Population totale | World Bank | `SP.POP.TOTL` |
| `Carence_pct` | Sous-alimentation (% pop.) | FAO | `SN.ITK.DEFC.ZS` |
| `MigrationNette` | Migration nette | World Bank | `SM.POP.NETM` |

## Ce que dit l'exploration

- ✅ **La fusion marche** : 9 indicateurs se joignent proprement sur `(Pays, Année)`.
- ⚠️ **Complétude** : tout > 82 % sauf `Carence_pct` (FAO, **2001→** seulement, ~27 %).
- 📉 **Signal clair** : transition démographique (fécondité ↓, mortalité infantile ↓) et forte
  corrélation fécondité ↔ mortalité infantile ↔ carence.
- 🦠 **Project Tycho** (épidémies, `epidemies_rougeole_tycho.csv`) reste **séparé** : granularité
  état US / semaine, pas de clé `(Pays, Année)` → analysé à part.

Sources complètes : voir **`SOURCES_DONNEES.txt`**.
