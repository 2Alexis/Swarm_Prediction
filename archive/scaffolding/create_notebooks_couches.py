"""
create_notebooks_couches.py — Génère un notebook par couche :
  - couche1_planete/modelisation_couche1.ipynb
  - couche2_sang/modelisation_couche2.ipynb

Chaque notebook charge le dataset partagé V8 honnête + sa propre config + entraîne ses modèles.
"""
import json
import os


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}


def make_layer_notebook(layer_name, layer_emoji, layer_title, layer_subtitle,
                        config_module, layer_dir, sublayer_descriptions):
    """Génère un notebook pour une couche."""
    return nb([
md(f"""# {layer_emoji} Notebook — Modélisation de la COUCHE {layer_name}

## {layer_title}
{layer_subtitle}

### Sous-couches modélisées
{sublayer_descriptions}

### Méthodologie
- Données : `data/cleaned/dataset_final_v12_couche1.csv` (8 400 × 741, 240 pays, 8 clusters)
  *(fallback automatique sur v11/v10/v9/v8 si non disponible)*
- Anti-leak strict via `{config_module}.py`
- 2 stratégies comparées par cible :
  - **A.** Modèle GLOBAL (toutes données mondiales)
  - **B.** GLOBAL + feature `cluster` (numéro de cluster climatique en input)
  - *(PerCluster désactivé — jamais utile en V11, gain de temps × 3)*
- Split : `GroupShuffleSplit` par pays (test = pays jamais vus)
- Modèle : XGBoost (500 estimateurs, depth=6, lr=0.05)

### Reproduction
```bash
python {layer_dir}/train.py
```
Charge `couche1_planete/models/best_*.joblib` ensuite."""),

md("## 1. Imports & configuration partagée"),
code(f"""import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Imports projet (config partagée + spécifique couche)
sys.path.insert(0, '..')  # racine du projet depuis sous-dossier
from config_shared import (
    load_dataset, make_preprocessor, make_xgb, select_top_features,
    build_blacklist, split_train_test, TOP_K
)
from config import (
    TARGETS, SUBLAYERS, TARGET_SOURCE, EXTRA_LEAKS,
    YIELD_TARGETS, SOCIO_TARGETS, DISASTER_TARGETS
)

from sklearn.metrics import r2_score, mean_absolute_error
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

print(f'Cibles {layer_name} : {{len(TARGETS)}} dans {{len(SUBLAYERS)}} sous-couches')
for sub, t in SUBLAYERS.items():
    print(f'  {{sub:30s}} : {{len(t)}} cibles')"""),

md("## 2. Chargement du dataset (V12 → V8 fallback)"),
code("""# Chargement avec fallback sur la version la plus récente disponible
for path, version in [('../data/cleaned/dataset_final_v12_couche1.csv', 'V12 (suitability + stress + religion + meat)'),
                       ('../data/cleaned/dataset_final_v11_couche1.csv', 'V11 (suitability EcoCrop)'),
                       ('../data/cleaned/dataset_final_v10_couche1.csv', 'V10 (cultures spé + religion + meat)'),
                       ('../data/cleaned/dataset_final_v9_couche1.csv',  'V9 (élevage + énergie + émissions)'),
                       (None, 'V8 default')]:
    if path is None:
        df = load_dataset()
        print(f'✓ Dataset {version} chargé : {df.shape}'); break
    if os.path.exists(path):
        df = load_dataset(path)
        print(f'✓ Dataset {version} chargé : {df.shape}'); break

print(f'   {df["ISO"].nunique()} pays, {df["cluster"].nunique()} clusters')

# Cibles dérivées (si pas déjà créées)
for c in ['disaster_deaths', 'disaster_affected']:
    if c in df.columns and f'target_{c}' not in df.columns:
        df[f'target_{c}'] = np.log1p(df[c].clip(lower=0))
if 'stunting_pct' in df.columns and 'target_stunting' not in df.columns:
    df['target_stunting'] = df['stunting_pct']
if 'Fertility_Rate' in df.columns and 'target_fertility' not in df.columns:
    df['target_fertility'] = df['Fertility_Rate']

# Vérification : toutes les cibles de la couche sont présentes
missing = [t for t in TARGETS if t not in df.columns]
print(f'\\nCibles présentes dans df : {len(TARGETS)-len(missing)}/{len(TARGETS)}')
if missing:
    print(f'⚠️  Cibles absentes : {missing}')"""),

md(f"## 3. Fonctions d'entraînement (héritées de config_shared)"),
code("""def get_bl(target, keep_cluster=False):
    return build_blacklist(
        df, target,
        target_source=TARGET_SOURCE.get(target),
        extra_leaks=EXTRA_LEAKS.get(target, []),
        yield_targets=YIELD_TARGETS,
        socio_targets=SOCIO_TARGETS,
        disaster_targets=DISASTER_TARGETS,
        keep_cluster=keep_cluster,
    )

def train_global(target, keep_cluster=False):
    d = df.dropna(subset=[target]).copy()
    if len(d) < 200: return None
    bl = get_bl(target, keep_cluster=keep_cluster)
    feats = [c for c in d.columns if c not in bl and d[c].dtype != object]
    feats = [c for c in feats if d[c].notna().sum() > 0]
    tr, te = split_train_test(d)
    Xtr_full, Xte_full = tr[feats], te[feats]
    ytr, yte = tr[target], te[target]
    sel = select_top_features(Xtr_full, ytr)
    if keep_cluster and 'cluster' in feats and 'cluster' not in sel:
        sel = ['cluster'] + sel[:-1]
    Xtr, Xte = Xtr_full[sel], Xte_full[sel]
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('pre', make_preprocessor()), ('model', make_xgb())])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    return {'r2': r2_score(yte, pred), 'mae': mean_absolute_error(yte, pred),
            'pipe': pipe, 'features': sel, 'n_obs': len(d)}"""),

md("## 4. Entraînement par sous-couche (Global + Global+Cluster)"),
code("""results = []

for sublayer, tgts in SUBLAYERS.items():
    print(f'━━━ {sublayer} ━━━')
    for tgt in tgts:
        if tgt not in df.columns: continue
        res_g  = train_global(tgt, keep_cluster=False)
        if res_g is None: continue
        res_gc = train_global(tgt, keep_cluster=True)
        # Sauvegarde meilleur
        joblib.dump({'pipe': res_gc['pipe'], 'features': res_gc['features']},
                    f'models/best_{tgt}.joblib')
        label = TARGETS[tgt]
        print(f'  🎯 {label:35s} Glob={res_g["r2"]:+.3f}  Glob+C={res_gc["r2"]:+.3f}')
        results.append({
            'Sous-couche': sublayer, 'Cible': label, 'Technique': tgt,
            'R² Global': round(res_g['r2'], 4),
            'R² Global+Cluster': round(res_gc['r2'], 4),
            'MAE': round(res_g['mae'], 3),
            'N obs': res_g['n_obs'],
        })
    print()

out = pd.DataFrame(results)
out.to_csv('reports/results_notebook.csv', index=False)
print('═' * 70)
print(f'📊 RÉSULTATS NOTEBOOK')
print('═' * 70)
print(out.to_string(index=False))"""),

md("## 5. Visualisation des résultats — performance par cible"),
code("""fig, ax = plt.subplots(figsize=(11, max(5, len(out)*0.5)))
out_sorted = out.sort_values('R² Global+Cluster')

# Couleurs par sous-couche
sublayer_colors = {sub: plt.cm.tab10(i) for i, sub in enumerate(SUBLAYERS)}
colors = [sublayer_colors[s] for s in out_sorted['Sous-couche']]

ax.barh(out_sorted['Cible'], out_sorted['R² Global+Cluster'], color=colors, alpha=0.85)
ax.set_xlim(-0.1, 1.0)
ax.axvline(0.5, color='gray', ls='--', alpha=0.5, label='R²=0.5')
ax.axvline(0.7, color='green', ls='--', alpha=0.5, label='R²=0.7')
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel('R² sur pays jamais vus')

# Légende sous-couches
from matplotlib.patches import Patch
patches = [Patch(color=c, label=s) for s, c in sublayer_colors.items()]
ax.legend(handles=patches + [ax.get_legend_handles_labels()[0][0]],
          labels=list(sublayer_colors.keys()) + ['R²=0.5'], loc='lower right')
ax.set_title(f'Performance par cible — Couche {layer_name}', weight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('reports/r2_par_cible.png', dpi=120, bbox_inches='tight')
plt.show()
""".replace("{layer_name}", layer_name)),

md("## 6. Feature importance — top 10 par cible"),
code("""TOP_CIBLES = out.sort_values('R² Global+Cluster', ascending=False).head(6)['Technique'].tolist()
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for ax, tgt in zip(axes.flatten(), TOP_CIBLES):
    path = f'models/best_{tgt}.joblib'
    if not os.path.exists(path): continue
    data = joblib.load(path)
    pipe = data['pipe']
    feats = data['features']
    model = pipe.named_steps['model']
    if not hasattr(model, 'feature_importances_'): continue
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=True).tail(10)
    colors = ['red' if f=='cluster' else 'steelblue' for f in imp.index]
    ax.barh(imp.index, imp.values, color=colors, alpha=0.85)
    label = TARGETS[tgt]
    ax.set_title(label, weight='bold', fontsize=10)
    ax.tick_params(axis='y', labelsize=8)
plt.suptitle(f'Top 10 features — Couche {layer_name} (rouge = cluster)',
             weight='bold', fontsize=14, y=1.01).replace("{layer_name}","")
plt.tight_layout()
plt.savefig('reports/feature_importance.png', dpi=120, bbox_inches='tight')
plt.show()
""".replace("{layer_name}", layer_name)),

md("## 7. Diagnostic prédit vs réel — 4 cibles fortes"),
code("""DIAG_TARGETS = out.sort_values('R² Global+Cluster', ascending=False).head(4)['Technique'].tolist()
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, tgt in zip(axes, DIAG_TARGETS):
    path = f'models/best_{tgt}.joblib'
    if not os.path.exists(path): continue
    data = joblib.load(path)
    pipe = data['pipe']
    feats = data['features']
    d = df.dropna(subset=[tgt]).copy()
    tr, te = split_train_test(d)
    X_te, y_te = te[feats], te[tgt]
    y_pred = pipe.predict(X_te)
    r2 = r2_score(y_te, y_pred)
    scatter = ax.scatter(y_te, y_pred, c=te['cluster'], cmap='tab10', alpha=0.4, s=10)
    lims = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', lw=1.5)
    ax.set_xlabel('Réel')
    ax.set_ylabel('Prédit')
    ax.set_title(f'{TARGETS[tgt]}\\nR² = {r2:.3f}', weight='bold', fontsize=10)
plt.suptitle(f'Prédit vs Réel — Couche {layer_name} (couleur = cluster)',
             weight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('reports/diagnostic.png', dpi=120, bbox_inches='tight')
plt.show()
""".replace("{layer_name}", layer_name)),

md(f"""## 8. Synthèse — Couche {layer_name}

Tous les modèles sont sauvegardés dans `couche{layer_name.split()[0].lower()}_*/models/best_*.joblib`.

### Pour charger un modèle :
```python
data = joblib.load('models/best_target_NAME.joblib')
pipe = data['pipe']           # pipeline sklearn complet
features = data['features']   # liste des features utilisées

# Prédire sur de nouvelles données :
prediction = pipe.predict(new_data[features])
```

### Pour la simulation
Ce modèle s'utilise dans le moteur de simulation pour estimer les sorties de cette couche
à partir des features environnementales d'entrée.

> Voir `couche{layer_name.split()[0].lower()}_*/reports/results.csv` pour le tableau complet."""),
])


# ════════════════════════════════════════════════════════════════════════
# NOTEBOOK COUCHE 1 — La Planète
# ════════════════════════════════════════════════════════════════════════

notebook_c1 = make_layer_notebook(
    layer_name="1",
    layer_emoji="🌍",
    layer_title="La Planète (V12 — cultures spécifiques + suitability + religion + meat)",
    layer_subtitle=("Domaine : environnement physique + agriculture (par culture spécifique) + élevage + écologie + énergie + émissions. "
                    "Dataset : `dataset_final_v12_couche1.csv` (8 400 × 741 colonnes). "
                    "Les sorties de cette couche servent d'inputs à la Couche 2 (Le Sang)."),
    config_module="couche1_planete/config",
    layer_dir="couche1_planete",
    sublayer_descriptions=(
        "1. **Céréales & racines** — 2 cibles : rendement céréales, racines/tubercules\n"
        "2. **Oléagineux par culture** — 8 : soja, colza, tournesol, arachide, olives, sésame, coco, coton\n"
        "3. **Fruits par culture** — 16 : pomme, banane, orange, raisin, fraise, ananas, mangue, avocat, citron, pêche, poire, pastèque, datte, abricot, cerise, prune\n"
        "4. **Légumes par culture** — 9 : tomate, pomme de terre, oignon, chou, carotte, concombre, aubergine, chou-fleur, laitue\n"
        "5. **Légumineuses par culture** — 3 : pois chiche, haricot sec, pois sec\n"
        "6. **Élevage** — 8 : lait, carcasses bovine/poulet/ovin/porc, œufs (rendement + production), aquaculture\n"
        "7. **Environnement physique** — 5 : stress hydrique, dégradation sol, anomalie thermique, humidité sol, accès eau\n"
        "8. **Écologie** — 2 : % forêt, perte couvert arboré\n"
        "9. **Émissions atmosphériques** — 3 : CO2, CH4, N2O\n"
        "10. **Énergie** — 6 : solaire, éolien, hydro, charbon, pétrole, gaz\n\n"
        "**Total : 62 cibles** dans 10 sous-couches.\n\n"
        "### Nouveautés V12\n"
        "- **EcoCrop suitability scores** (39 cultures) — score 0-1 basé sur T_opt/P_opt/frost/latitude\n"
        "- **Indices de stress climatique** : heat_stress, frost_risk, aridity, PET, growing_season, continentalité\n"
        "- **Pew Religion 2010** : % muslim/christian/hindu/buddhist par pays\n"
        "- **OWID Meat consumption per type** : beef, pig, poultry, sheep/goat, milk, eggs\n"
        "- **PerCluster désactivé** (jamais utile, gain de temps × 3)"
    ),
)


# ════════════════════════════════════════════════════════════════════════
# NOTEBOOK COUCHE 2 — Le Sang
# ════════════════════════════════════════════════════════════════════════

notebook_c2 = make_layer_notebook(
    layer_name="2",
    layer_emoji="🩸",
    layer_title="Le Sang",
    layer_subtitle=(
        "Domaine : démographie + épidémiologie + catastrophes humaines.\n\n"
        "**Principe central du projet :** prédire la démographie à partir UNIQUEMENT des features "
        "environnementales (Couche 1) — montrer que la terre détermine les gens. "
        "Les variables socio-éco sont blacklistées pour rester rigoureux."
    ),
    config_module="couche2_sang/config",
    layer_dir="couche2_sang",
    sublayer_descriptions=(
        "1. **Démographie** — 7 cibles : natalité, mortalité brut, mortalité infantile, espérance de vie, "
        "croissance démo, migration nette, fécondité\n"
        "2. **Santé / Épidémiologie** — 1 cible : retard de croissance (stunting)\n"
        "3. **Catastrophes humaines** — 2 cibles : décès et affectés par catastrophes"
    ),
)


# ════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════
for path, nb_data in [
    ("couche1_planete/modelisation_couche1.ipynb", notebook_c1),
    ("couche2_sang/modelisation_couche2.ipynb", notebook_c2),
]:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, ensure_ascii=False, indent=1)
    print(f"[OK] {path}  ({len(nb_data['cells'])} cellules)")

print("\n[DONE] Notebooks par couche générés.")
