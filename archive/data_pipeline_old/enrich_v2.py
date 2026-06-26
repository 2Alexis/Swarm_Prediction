"""
enrich_v2.py — Enrichit dataset_final_v2.csv avec :
  - Indices climatiques mondiaux (ONI, SOI, NAO, AMO, PDO, AO) + leurs lags
  - CO2 atmosphérique mondial (Mauna Loa) + lags
  - OWID catastrophes (deaths, events, affected, damages_usd) + cumul 5 ans
  - Séismes yearly (count, max mag, mean depth)
  - Power generation yearly (gen_gwh, renewable share)
  - Volcans pays (count, dernière éruption, pop dans 100km)
  - OWID extras (fertility, life_exp_extra, internet)
Sortie : data/cleaned/dataset_final_v3.csv
"""
import os, sys, io
import pandas as pd
import numpy as np
import pycountry, babel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = "data/cleaned"

from build_dataset import custom_mappings, get_english_iso, get_iso_from_alpha3

# ── Helpers ──────────────────────────────────────────────────────────────
def to_iso(pays_series):
    """Best-effort Pays → ISO."""
    out = []
    for p in pays_series:
        if pd.isna(p):
            out.append(None); continue
        s = str(p).strip().lower()
        code = custom_mappings.get(s)
        if not code:
            code = get_english_iso(p)
        out.append(code)
    return out

def add_global_index(df, csv_path, colname, lags=(1, 3, 5)):
    if not os.path.exists(csv_path):
        print(f"  ✗ {csv_path} manquant")
        return df
    idx = pd.read_csv(csv_path)
    idx = idx.dropna().rename(columns={"Valeur": colname})
    idx = idx.groupby("Annee", as_index=False)[colname].mean()
    # Sort + lags
    idx = idx.sort_values("Annee")
    for k in lags:
        idx[f"{colname}_lag{k}"] = idx[colname].shift(k)
    idx[f"{colname}_roll5"] = idx[colname].rolling(5, min_periods=2).mean()
    df = df.merge(idx, on="Annee", how="left")
    print(f"  + {colname} (+lags) couvre {idx['Annee'].min()}-{idx['Annee'].max()}")
    return df

def add_owid_country(df, csv_path, colname, fillna=None):
    if not os.path.exists(csv_path):
        print(f"  ✗ {csv_path} manquant"); return df
    sub = pd.read_csv(csv_path)
    sub["ISO"] = to_iso(sub["Pays"])
    sub = sub.dropna(subset=["ISO"])
    sub["Annee"] = pd.to_numeric(sub["Annee"], errors="coerce")
    sub = sub.dropna(subset=["Annee"])
    sub["Annee"] = sub["Annee"].astype(int)
    sub["Valeur"] = pd.to_numeric(sub["Valeur"], errors="coerce")
    sub = sub.dropna(subset=["Valeur"])
    sub = sub.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": colname})
    df = df.merge(sub, on=["ISO", "Annee"], how="left")
    if fillna is not None:
        df[colname] = df[colname].fillna(fillna)
    print(f"  + {colname} ({sub[colname].notna().sum()} valeurs)")
    return df

def add_iso_year_csv(df, csv_path):
    if not os.path.exists(csv_path):
        print(f"  ✗ {csv_path} manquant"); return df
    sub = pd.read_csv(csv_path)
    df = df.merge(sub, on=["ISO", "Annee"], how="left")
    cols = [c for c in sub.columns if c not in ("ISO", "Annee")]
    print(f"  + {cols} de {csv_path}")
    return df

def add_iso_static_csv(df, csv_path):
    if not os.path.exists(csv_path):
        print(f"  ✗ {csv_path} manquant"); return df
    sub = pd.read_csv(csv_path)
    df = df.merge(sub, on=["ISO"], how="left")
    cols = [c for c in sub.columns if c != "ISO"]
    print(f"  + {cols} (statique)")
    return df


# ── 1. Charger v2 ──────────────────────────────────────────────────────────
print("[1] Chargement de dataset_final_v2.csv…")
df = pd.read_csv(f"{D}/dataset_final_v2.csv")
df = df.dropna(subset=["ISO"]).copy()
df["ISO"] = df["ISO"].astype(str)
df["Annee"] = df["Annee"].astype(int)
print(f"   shape de départ : {df.shape}")

# ── 2. Indices climatiques mondiaux ────────────────────────────────────────
print("\n[2] Indices climatiques mondiaux (ONI/SOI/NAO/AMO/PDO/AO)…")
for short, longname in [("oni", "enso_oni"), ("soi", "enso_soi"),
                         ("nao", "nao"), ("amo", "amo"),
                         ("pdo", "pdo"), ("ao", "ao")]:
    df = add_global_index(df, f"{D}/climate_index_{short}.csv", longname)

# ── 3. CO2 mondial ─────────────────────────────────────────────────────────
print("\n[3] CO2 atmosphérique mondial…")
df = add_global_index(df, f"{D}/global_co2_atmospheric.csv", "co2_ppm_global")

# ── 4. OWID catastrophes ───────────────────────────────────────────────────
print("\n[4] OWID — catastrophes naturelles…")
df = add_owid_country(df, f"{D}/owid_disaster_deaths.csv",      "disaster_deaths", fillna=0)
df = add_owid_country(df, f"{D}/owid_disaster_events.csv",      "disaster_events", fillna=0)
df = add_owid_country(df, f"{D}/owid_disaster_affected.csv",    "disaster_affected", fillna=0)
df = add_owid_country(df, f"{D}/owid_disaster_damages_usd.csv", "disaster_damages_usd", fillna=0)

# Cumul 5 ans (mémoire des catastrophes)
df = df.sort_values(["ISO", "Annee"])
for c in ["disaster_deaths", "disaster_events", "disaster_affected", "disaster_damages_usd"]:
    if c in df.columns:
        df[f"{c}_cumul5y"] = df.groupby("ISO")[c].transform(lambda s: s.rolling(5, min_periods=1).sum())

# ── 5. OWID extras ────────────────────────────────────────────────────────
print("\n[5] OWID extras…")
df = add_owid_country(df, f"{D}/owid_life_exp_extra.csv", "Life_Exp_OWID")
df = add_owid_country(df, f"{D}/owid_fertility.csv",      "Fertility_Rate")
df = add_owid_country(df, f"{D}/owid_internet.csv",       "Internet_OWID")

# ── 6. Séismes yearly ─────────────────────────────────────────────────────
print("\n[6] Séismes yearly…")
df = add_iso_year_csv(df, f"{D}/earthquakes_yearly.csv")
# Fillna pour pays sans séismes
for c in ["earthquake_count", "earthquake_max_mag", "earthquake_mean_mag", "earthquake_mean_depth"]:
    if c in df.columns:
        if c == "earthquake_count":
            df[c] = df[c].fillna(0)
        else:
            df[c] = df[c].fillna(df[c].median())

# ── 7. Power generation yearly ────────────────────────────────────────────
print("\n[7] Power generation yearly…")
df = add_iso_year_csv(df, f"{D}/power_generation_yearly.csv")

# ── 8. Volcans pays (statique enrichi) ────────────────────────────────────
print("\n[8] Volcans pays…")
df = add_iso_static_csv(df, f"{D}/volcanoes_country.csv")
# Récency dernière éruption = Annee - last_eruption_year (dynamique !)
if "last_eruption_year" in df.columns:
    df["years_since_last_eruption"] = df["Annee"] - df["last_eruption_year"]
    df["years_since_last_eruption"] = df["years_since_last_eruption"].clip(lower=0).fillna(9999)

# ── 9. Sauvegarde ─────────────────────────────────────────────────────────
out = f"{D}/dataset_final_v3.csv"
df.to_csv(out, index=False)
print(f"\n[OK] {out}")
print(f"     shape = {df.shape}")
print(f"     +{df.shape[1] - 159} nouvelles colonnes vs v2")

# Récap features dynamiques
num = df.select_dtypes(include="number").columns.tolist()
static = []
dynamic = []
for c in num:
    s = df.groupby("ISO")[c].std().mean()
    if pd.notna(s) and s < 0.01:
        static.append(c)
    else:
        dynamic.append(c)
print(f"     DYNAMIC: {len(dynamic)} / {len(num)} (était 133/158 en v2)")
