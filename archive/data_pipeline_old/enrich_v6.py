"""
enrich_v6.py — dataset_final_v6.csv → dataset_final_v7.csv

  + Berkeley Earth : anomalies T par pays 1743-2020 (multi-décennal)
    → anomalies 10y, 30y, 60y, 100y
    → tendance T décennale
    → variance climatique historique
"""
import os, sys, io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = "data/cleaned"

print("[1] Chargement v6…")
df = pd.read_csv(f"{D}/dataset_final_v6.csv")
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
print(f"   départ : {df.shape}")

print("[2] Berkeley Earth (anomalies T 1743-2020)…")
be = pd.read_csv(f"{D}/berkeley_earth_yearly.csv")
be["ISO"] = be["ISO"].astype(str)
# Merger anomalies annuelles
df = df.merge(be, on=["ISO", "Annee"], how="left")
print(f"   + 4 colonnes (be_t_anom_*)")

# Anomalies multi-décennales : T(année courante) - moyenne historique
df = df.sort_values(["ISO", "Annee"])

# Calculer baseline pré-industrielle (1850-1900) par pays
be_pre = be[(be["Annee"] >= 1850) & (be["Annee"] <= 1900)].groupby("ISO")["be_t_anom_annual"].mean()
df["be_t_baseline_1850_1900"] = df["ISO"].map(be_pre)
df["be_t_anom_vs_preindustrial"] = df["be_t_anom_annual"] - df["be_t_baseline_1850_1900"]

# Anomalies cumulées sur fenêtres glissantes
print("[3] Anomalies multi-décennales…")
for window, name in [(10, "10y"), (30, "30y"), (60, "60y"), (100, "100y")]:
    df[f"be_t_anom_roll{name}"] = df.groupby("ISO")["be_t_anom_annual"].transform(
        lambda s: s.rolling(window, min_periods=max(3, window // 5)).mean())

# Tendance T décennale = slope sur fenêtre glissante 10y
def trend_slope(s):
    arr = s.dropna().values
    if len(arr) < 5:
        return np.nan
    n = len(arr)
    x = np.arange(n)
    return np.polyfit(x, arr, 1)[0]

print("[4] Tendances T décennales…")
df["be_t_trend_10y"] = df.groupby("ISO")["be_t_anom_annual"].transform(
    lambda s: s.rolling(10, min_periods=5).apply(trend_slope, raw=False))
df["be_t_trend_30y"] = df.groupby("ISO")["be_t_anom_annual"].transform(
    lambda s: s.rolling(30, min_periods=10).apply(trend_slope, raw=False))

# Variance climatique historique
df["be_t_volatility_30y"] = df.groupby("ISO")["be_t_anom_annual"].transform(
    lambda s: s.rolling(30, min_periods=5).std())

# Saut décennal (différence avec moyenne 30 ans précédentes)
print("[5] Saut décennal…")
df["be_t_decade_shift"] = (
    df.groupby("ISO")["be_t_anom_roll10y"].transform("first") if False
    else df["be_t_anom_roll10y"] - df["be_t_anom_roll30y"]
)

# Lags Berkeley
for k in (1, 3, 5, 10):
    df[f"be_t_anom_lag{k}"] = df.groupby("ISO")["be_t_anom_annual"].shift(k)

print("[6] Sauvegarde…")
out = f"{D}/dataset_final_v7.csv"
df.to_csv(out, index=False)
print(f"\n[OK] {out}")
print(f"     shape = {df.shape}  (+{df.shape[1] - 566} cols vs v6)")
print(f"     Berkeley couverture : {df['be_t_anom_annual'].notna().sum()} lignes non-nulles")
