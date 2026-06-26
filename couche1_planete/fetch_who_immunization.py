"""
fetch_who_immunization.py — Récupère les données WHO Immunization via GHO API.

API gratuite, no-auth : https://ghoapi.azureedge.net/api/{INDICATOR_CODE}

Indicateurs couverts (WUENIC = WHO/UNICEF Estimates) :
  - BCG : Tuberculose
  - DTP1 / DTP3 : Diphtérie-Tétanos-Pertussis (1ère et 3e dose)
  - HepB3 : Hépatite B
  - Hib3 : Haemophilus influenzae type b
  - MCV1 / MCV2 : Rougeole 1ère et 2e dose
  - PCV3 : Pneumocoque
  - Pol3 : Polio
  - RotaC : Rotavirus
  - IPV1 : Polio inactivé
"""
import os, sys, io, urllib.request, json, time
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pycountry
def iso3_to_iso2(c):
    try:
        co = pycountry.countries.get(alpha_3=c)
        return co.alpha_2 if co else None
    except: return None


def fetch_who_gho(indicator):
    """Récupère un indicateur GHO via API JSON."""
    url = f"https://ghoapi.azureedge.net/api/{indicator}?$format=json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8", errors="ignore"))
        rows = []
        for r_data in data.get("value", []):
            yr = r_data.get("TimeDim") or r_data.get("YearCode")
            spat = r_data.get("SpatialDimType")
            iso = r_data.get("SpatialDim") if spat == "COUNTRY" else None
            val = r_data.get("NumericValue")
            if val is None: val = r_data.get("Value")
            if iso is None or yr is None or val is None: continue
            try:
                rows.append({"ISO3": iso, "Annee": int(yr), "Valeur": float(val)})
            except: continue
        return pd.DataFrame(rows)
    except Exception as e:
        return None


# Codes GHO pour vaccins WUENIC
INDICATORS = {
    "WHS4_544":  "who_pol3_pct",      # Polio 3e dose
    "WHS4_543":  "who_mcv1_pct",      # Rougeole 1ère dose
    "WHS4_117":  "who_bcg_pct",       # BCG
    "WHS4_125":  "who_hepb3_pct",     # HepB 3e dose
    "WHS4_100":  "who_hib3_pct",      # Hib 3e dose
    "WHS4_128":  "who_dtp1_pct",      # DTP 1ère dose
    "WHS4_129":  "who_dtp3_pct",      # DTP 3e dose
    "WHS8_110":  "who_mcv2_pct",      # Rougeole 2e dose
    "WHS4_543v2":"who_pcv3_pct",      # Pneumocoque 3e (alt)
    "WHS5_543":  "who_rotac_pct",     # Rotavirus
    "WHS4_544v2":"who_ipv1_pct",      # Polio inactivé 1ère
}

# Alternative noms en cas d'échec
ALT_INDICATORS = {
    "WSH_IMM_BCG":     "who_bcg_alt",
    "WSH_IMM_DTP3":    "who_dtp3_alt",
    "WSH_IMM_HEPB3":   "who_hepb3_alt",
    "WSH_IMM_HIB3":    "who_hib3_alt",
    "WSH_IMM_MCV1":    "who_mcv1_alt",
    "WSH_IMM_PCV3":    "who_pcv3_alt",
    "WSH_IMM_POL3":    "who_pol3_alt",
    "WSH_IMM_ROTAC":   "who_rotac_alt",
    "WSH_IMM_IPV1":    "who_ipv1_alt",
    "IMM_DTP3_DSTRC":  "who_dtp3_districts",
}


print("══════════════════════════════════════════════════════════════")
print("📡 Téléchargement WHO Immunization via GHO API")
print("══════════════════════════════════════════════════════════════\n")

all_dfs = []
successful = []
failed = []

# Premier batch : WHS4_*
for code, name in INDICATORS.items():
    print(f"  → {code:15s} ({name})...", end=" ", flush=True)
    df = fetch_who_gho(code)
    if df is not None and len(df) > 50:
        df["ISO"] = df["ISO3"].apply(iso3_to_iso2)
        df = df.dropna(subset=["ISO"])
        df = df.groupby(["ISO","Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur":name})
        all_dfs.append(df)
        successful.append((code, name, len(df)))
        print(f"✓ {len(df)} obs")
    else:
        failed.append(code)
        print(f"✗ vide ou erreur")
    time.sleep(0.3)


# Deuxième batch : noms alternatifs
print("\n  Tentative noms alternatifs (WSH_IMM_*)…")
for code, name in ALT_INDICATORS.items():
    if name.replace("_alt","") in [s[1].replace("_pct","") for s in successful]: continue
    print(f"  → {code:18s} ({name})...", end=" ", flush=True)
    df = fetch_who_gho(code)
    if df is not None and len(df) > 50:
        df["ISO"] = df["ISO3"].apply(iso3_to_iso2)
        df = df.dropna(subset=["ISO"])
        df = df.groupby(["ISO","Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur":name})
        all_dfs.append(df)
        successful.append((code, name, len(df)))
        print(f"✓ {len(df)} obs")
    else:
        failed.append(code)
        print(f"✗")
    time.sleep(0.3)


# Fusion finale
print(f"\n📊 BILAN")
print(f"  ✓ Réussis : {len(successful)}/{len(INDICATORS)+len(ALT_INDICATORS)}")
for code, name, n in successful:
    print(f"     {code:18s} {name:25s} {n:,} obs")

if all_dfs:
    # Outer merge sur (ISO, Annee)
    merged = all_dfs[0]
    for d in all_dfs[1:]:
        merged = merged.merge(d, on=["ISO","Annee"], how="outer")

    out = "data/cleaned/demographie/who_immunization_coverage.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"\n  → {out} ({len(merged)} lignes × {merged.shape[1]-2} indicateurs)")
    print(f"  Couverture : {merged['ISO'].nunique()} pays, {merged['Annee'].min()}-{merged['Annee'].max()}")
else:
    print("\n  ✗ Aucun indicateur récupéré — API potentiellement HS")
