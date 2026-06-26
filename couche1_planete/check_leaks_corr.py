"""
check_leaks_corr.py — Matrice de corrélation cibles suspectes × features ajoutées.

Pour chaque cible "suspecte" en V14, calcule la corrélation absolue avec :
  - Toutes les colonnes ajoutées en V14 (FAO yields top30, FRA, FAO Fertilizers by Product...)

Classification :
  - r > 0.95 → LEAK CERTAIN (cible déguisée) → à blacklister
  - 0.85 ≤ r ≤ 0.95 → leak probable, à investiguer
  - 0.50 ≤ r < 0.85 → feature utile (info partielle)
  - r < 0.50 → ok / décorrélée
"""
import os, sys, io
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CLN = "data/cleaned"

print("[1] Chargement v14…")
df = pd.read_csv(f"{CLN}/shared/dataset_final_v14_couche1.csv", low_memory=False)
print(f"   Shape : {df.shape}")

# Cibles suspectes (gain ≥ 0.40 en V14)
SUSPECTS = {
    "target_forest_share":       0.89,
    "target_tree_cover_loss":    0.99,
    "target_pig_carcass":        0.95,
    "target_sheepgoat_carcass":  0.88,
    "target_cattle_carcass":     0.85,
    "target_yield_cereals":      0.85,
    "target_chicken_carcass":    0.82,
}

# Identifier les features ajoutées en V14 vs V13
v13_cols = set(pd.read_csv(f"{CLN}/shared/dataset_final_v13_couche1.csv", nrows=0).columns)
v14_new_cols = [c for c in df.columns if c not in v13_cols
                 and not c.startswith("target_")
                 and df[c].dtype in ("float64","int64")]
print(f"   {len(v14_new_cols)} nouvelles features V14 à tester")

# Pour chaque cible suspecte, calcul des corrs
print("\n══════════════════════════════════════════════════════════════")
print("🔍 ANALYSE DE LEAKS — corr(cible, features V14)")
print("══════════════════════════════════════════════════════════════")

leak_report = []
for tgt, r2_v14 in SUSPECTS.items():
    if tgt not in df.columns:
        print(f"\n⏭ {tgt} absent"); continue

    sub = df.dropna(subset=[tgt])
    y = sub[tgt]
    print(f"\n━━ {tgt}  (R² V14 = {r2_v14:+.2f})  ━━")

    # Compute correlations
    rows = []
    for c in v14_new_cols:
        x = sub[c]
        # Filtrer NaN
        m = x.notna() & y.notna()
        if m.sum() < 100: continue
        r = x[m].corr(y[m])
        if pd.notna(r):
            rows.append({"feature": c, "abs_corr": abs(r), "corr": r,
                          "n_obs": int(m.sum())})

    if not rows: continue
    corr_df = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)

    # Affichage top 10
    top10 = corr_df.head(10)
    print(f"  Top 10 corrélations avec features V14 :")
    for _, r in top10.iterrows():
        flag = ""
        if r["abs_corr"] > 0.95: flag = "🚨 LEAK CERTAIN"
        elif r["abs_corr"] > 0.85: flag = "⚠️ leak probable"
        elif r["abs_corr"] > 0.50: flag = "✓ utile"
        print(f"    {r['feature']:50s} |r|={r['abs_corr']:.3f}  n={r['n_obs']:5d}  {flag}")

    # Verdict
    n_leaks = (corr_df["abs_corr"] > 0.95).sum()
    n_suspicious = ((corr_df["abs_corr"] > 0.85) & (corr_df["abs_corr"] <= 0.95)).sum()
    n_useful = ((corr_df["abs_corr"] > 0.50) & (corr_df["abs_corr"] <= 0.85)).sum()

    print(f"\n  📊 Verdict : {n_leaks} leaks certains, {n_suspicious} suspects, {n_useful} utiles")

    leak_report.append({
        "Cible": tgt, "R²V14": r2_v14,
        "N leaks (r>0.95)": n_leaks,
        "N suspects (0.85-0.95)": n_suspicious,
        "N utiles (0.50-0.85)": n_useful,
        "Top feature": top10.iloc[0]["feature"],
        "Top |r|": round(top10.iloc[0]["abs_corr"], 3),
    })


# ── BILAN GLOBAL ──────────────────────────────────────────────────────────
print("\n\n══════════════════════════════════════════════════════════════")
print("📊 BILAN GLOBAL")
print("══════════════════════════════════════════════════════════════")
report_df = pd.DataFrame(leak_report)
print(report_df.to_string(index=False))
report_df.to_csv("couche1_planete/reports/leaks_correlation_v14.csv", index=False)
print(f"\n→ couche1_planete/reports/leaks_correlation_v14.csv")
