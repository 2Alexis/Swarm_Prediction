"""
couche2_sang/train.py — Entraînement de la COUCHE 2 : Le Sang.

Démographie + Épidémiologie + Catastrophes humaines.

PRINCIPE : on prédit la démographie à partir de l'environnement uniquement
(SOCIO_VARS blacklistés) — pour montrer que la terre détermine les gens.

Sortie :
  - couche2_sang/models/best_{target}.joblib
  - couche2_sang/reports/results.csv
"""
import os, sys, io, warnings
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_shared import (load_dataset, make_preprocessor, make_xgb,
                            select_top_features, build_blacklist, split_train_test,
                            TOP_K)
from couche2_sang.config import (TARGETS, SUBLAYERS, TARGET_SOURCE, EXTRA_LEAKS,
                                  YIELD_TARGETS, SOCIO_TARGETS, DISASTER_TARGETS)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

LAYER_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(LAYER_DIR, "models")
REPORTS_DIR = os.path.join(LAYER_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("══════════════════════════════════════════════════════════════")
print("🩸 COUCHE 2 — LE SANG")
print("    Démographie + Épidémiologie + Catastrophes humaines")
print("    (env → démo : on prédit les gens à partir de la terre)")
print("══════════════════════════════════════════════════════════════\n")

df = load_dataset()

# Cibles dérivées (créées dynamiquement)
for c in ["disaster_deaths", "disaster_affected"]:
    if c in df.columns and f"target_{c}" not in df.columns:
        df[f"target_{c}"] = np.log1p(df[c].clip(lower=0))
if "stunting_pct" in df.columns and "target_stunting" not in df.columns:
    df["target_stunting"] = df["stunting_pct"]
if "Fertility_Rate" in df.columns and "target_fertility" not in df.columns:
    df["target_fertility"] = df["Fertility_Rate"]

print(f"Dataset : {df.shape}, {df['ISO'].nunique()} pays, {df['cluster'].nunique()} clusters\n")


def get_bl(target, keep_cluster=False):
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
    if keep_cluster and "cluster" in feats and "cluster" not in sel:
        sel = ["cluster"] + sel[:-1]
    Xtr, Xte = Xtr_full[sel], Xte_full[sel]
    pipe = Pipeline([("pre", make_preprocessor()), ("model", make_xgb())])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    return {"r2": r2_score(yte, pred), "mae": mean_absolute_error(yte, pred),
            "pipe": pipe, "features": sel, "n_obs": len(d),
            "n_test_countries": te["ISO"].nunique()}


def train_per_cluster(target):
    weighted_r2, total_n = 0, 0
    per_c = {}
    for c in sorted(df["cluster"].unique()):
        d_c = df[df["cluster"] == c].dropna(subset=[target])
        if len(d_c) < 100 or d_c["ISO"].nunique() < 4:
            per_c[c] = None; continue
        bl = get_bl(target)
        feats = [col for col in d_c.columns if col not in bl and d_c[col].dtype != object]
        feats = [col for col in feats if d_c[col].notna().sum() > 0]
        if not feats: continue
        try:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
            tr_idx, te_idx = next(splitter.split(d_c, groups=d_c["ISO"]))
        except Exception: continue
        tr, te = d_c.iloc[tr_idx], d_c.iloc[te_idx]
        if len(te) < 30: continue
        Xtr, Xte = tr[feats], te[feats]
        ytr, yte = tr[target], te[target]
        sel = select_top_features(Xtr, ytr, k=min(40, len(feats)))
        Xtr, Xte = Xtr[sel], Xte[sel]
        try:
            pipe = Pipeline([("pre", make_preprocessor()),
                             ("model", XGBRegressor(n_estimators=300, max_depth=5,
                                                    learning_rate=0.05, random_state=42,
                                                    n_jobs=-1, verbosity=0))])
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            r2 = r2_score(yte, pred)
            per_c[c] = r2
            joblib.dump({"pipe": pipe, "features": sel},
                        os.path.join(MODELS_DIR, f"{target}_c{c}.joblib"))
            weighted_r2 += r2 * len(te)
            total_n += len(te)
        except Exception: continue
    avg_r2 = weighted_r2 / total_n if total_n > 0 else np.nan
    return {"r2": avg_r2, "per_cluster": per_c}


# ── Entraînement par sous-couche ──────────────────────────────────────────
results = []
for sublayer, tgts in SUBLAYERS.items():
    print(f"━━━ Sous-couche : {sublayer} ━━━")
    for tgt in tgts:
        if tgt not in df.columns: continue
        label = TARGETS[tgt]
        res_g  = train_global(tgt, keep_cluster=False)
        if res_g is None:
            print(f"  ⏭  {label} (n<200)")
            continue
        res_gc = train_global(tgt, keep_cluster=True)
        res_pc = train_per_cluster(tgt)

        # Sauvegarder le meilleur (Global+Cluster par défaut)
        joblib.dump({"pipe": res_gc["pipe"], "features": res_gc["features"]},
                    os.path.join(MODELS_DIR, f"best_{tgt}.joblib"))

        g  = res_g["r2"]
        gc = res_gc["r2"]
        pc = res_pc["r2"] if pd.notna(res_pc["r2"]) else np.nan
        print(f"  🎯 {label:38s} Glob={g:+.3f}  Glob+C={gc:+.3f}  PerC={pc:+.3f}")
        results.append({
            "Sous-couche": sublayer, "Cible": label, "Technique": tgt,
            "R² Global": round(g, 4),
            "R² Global+Cluster": round(gc, 4) if pd.notna(gc) else None,
            "R² PerCluster": round(pc, 4) if pd.notna(pc) else None,
            "MAE": round(res_g["mae"], 3),
            "N obs": res_g["n_obs"],
            "N pays test": res_g["n_test_countries"],
        })
    print()

out = pd.DataFrame(results)
out_path = os.path.join(REPORTS_DIR, "results.csv")
out.to_csv(out_path, index=False)
print("══════════════════════════════════════════════════════════════")
print("📊 RÉSULTATS COUCHE 2")
print("══════════════════════════════════════════════════════════════")
print(out.to_string(index=False))
print(f"\n→ {out_path}")
print(f"→ {len(os.listdir(MODELS_DIR))} modèles sauvegardés dans {MODELS_DIR}/")
