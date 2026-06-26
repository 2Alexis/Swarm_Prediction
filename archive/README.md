# 📦 archive/ — itérations précédentes (conservées, non utilisées)

Ce dossier regroupe le **code des générations précédentes** du projet, déplacé ici lors
du rangement de l'arborescence. **Rien n'est supprimé** : tout reste accessible et
l'historique git est intact. La structure de référence actuelle est la **V3 thématique**
(`couche1_planete/` + `couche2_sang/`).

Pour restaurer un fichier : `git mv archive/<dossier>/<fichier> .`

| Sous-dossier | Contenu | Pourquoi archivé |
|---|---|---|
| `notebooks_9cibles/` | EDA, data_*, modelisation_9_cibles, couche1_histoire, couche1_x_couche2… | Narration de l'ancienne version « 9 cibles », remplacée par les notebooks V3 (`couche1_planete/modelisation_couche1.ipynb`, `couche2_sang/modelisation_couche2.ipynb`) |
| `train_iterations/` | `train_v2` → `train_v8_*`, `train_by_cluster/crop/stacking` | Itérations d'entraînement de l'ancienne version. La V3 entraîne via `couche*/train.py` |
| `data_pipeline_old/` | `enrich_v2..v6`, `fetch_*`, `download_*`, `impute_*`, `build_dataset_v2`… | Ancien pipeline de collecte/enrichissement. La V3 a sa propre chaîne dans `couche1_planete/` (`build_v13`, `fix_fra_and_build_v14`…) |
| `scaffolding/` | `create_notebook*.py`, `create_organize_notebooks.py` | Générateurs de notebooks (one-shot), les notebooks existent déjà |
| `models_old/` | `models/`, `models_v2..v8`, `models_cluster/crop/stack` | Anciens modèles entraînés. Les modèles V3 sont dans `couche1_planete/models/` et `couche2_sang/models/` |
| `reports_old/` | ancien dossier `reports/` (tableaux v2..v7, figures) | Résultats de l'ancienne version. Résultats V3 dans `couche*/reports/results.csv` |

## Fichiers racine conservés (nécessaires à la V3)

`config_shared.py` · `build_dataset.py` · `layer1_engine.py` · `clean_data.py` — importés par
les notebooks et scripts de `couche1_planete/` et `couche2_sang/`.
