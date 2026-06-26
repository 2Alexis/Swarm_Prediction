"""Diagnostic des 10 cibles faibles de la Couche 1."""
import os, sys, io
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

df = pd.read_csv("data/cleaned/dataset_final_v9_couche1.csv", low_memory=False)
df["cluster"] = df["cluster"].astype(int)
df["Annee"] = df["Annee"].astype(int)
df["ISO"] = df["ISO"].astype(str)

WEAK_TARGETS = {
    "target_eggs_yield":           ("Rendement oeufs (kg/poule)",       "livestock_eggs_yield"),
    "target_yield_vegetables":     ("Rendement legumes",                "yield_vegetables_kgha"),
    "target_chicken_carcass":      ("Poids carcasse poulet (g)",        "livestock_chicken_carcass_g"),
    "target_cattle_carcass":       ("Poids carcasse bovine (kg)",       "livestock_cattle_carcass_kg"),
    "target_sheepgoat_carcass":    ("Poids carcasse ovin/caprin (kg)",  "livestock_sheepgoat_carcass_kg"),
    "target_pig_carcass":          ("Poids carcasse porc (kg)",         "livestock_pig_carcass_kg"),
    "target_yield_fruits":         ("Rendement fruits",                 "yield_fruits_kgha"),
    "target_yield_oilcrops":       ("Rendement oleagineux",             "yield_oilcrops_kgha"),
    "target_yield_pulses":         ("Rendement legumineuses",           "yield_pulses_kgha"),
    "target_marine_protected":     ("% aires marines protegees",        "marine_protected_pct"),
}

target_cols = [c for c in df.columns if c.startswith("target_")]
EXCL_BASE = set(target_cols) | {"ISO","Annee","cluster","T_ref","P_ref"}

for tgt, (label, source) in WEAK_TARGETS.items():
    print("\n" + "="*78)
    print(f"== {label}")
    print(f"   ({tgt})")
    print("="*78)
    sub = df.dropna(subset=[tgt])
    n = len(sub)
    n_iso = sub["ISO"].nunique()
    s = sub[tgt]
    print(f"\n  Distribution : n={n:,} | pays={n_iso} | annees={sub['Annee'].min()}-{sub['Annee'].max()}")
    print(f"                 min={s.min():.2f} max={s.max():.2f} mean={s.mean():.2f} std={s.std():.2f}")

    intra_var = sub.groupby("ISO")[tgt].var().mean()
    inter_var = sub.groupby("ISO")[tgt].mean().var()
    icc = inter_var / (intra_var + inter_var) if (intra_var + inter_var) > 0 else 0
    print(f"\n  Variabilite :")
    print(f"    intra-pays (entre annees)  : var = {intra_var:.3f}")
    print(f"    inter-pays                 : var = {inter_var:.3f}")
    print(f"    ICC                        : {icc:.2f}")
    if icc > 0.7:
        print(f"    -> signal DOMINE PAR LES DIFFERENCES PAYS (peu de variation temporelle)")
    elif icc > 0.3:
        print(f"    -> signal partage entre pays et annees")
    else:
        print(f"    -> signal DOMINE PAR VARIATIONS TEMPORELLES")

    excl = EXCL_BASE.copy()
    if source:
        excl |= {c for c in df.columns if c == source or c.startswith(source + "_")}
    feats = [c for c in sub.select_dtypes(include="number").columns
             if c not in excl and sub[c].notna().sum() > 100]
    cors = sub[feats].corrwith(sub[tgt]).dropna()
    cors_abs = cors.abs().sort_values(ascending=False)
    top = cors_abs.head(10)
    max_r = top.iloc[0] if len(top) else 0
    print(f"\n  Plafond theorique (max |r| feature) : {max_r:.3f}")
    print(f"  Top 10 correlations :")
    for f in top.index:
        r = cors[f]
        print(f"    {f:50s} r={r:+.3f}")

print("\n[DONE]")
