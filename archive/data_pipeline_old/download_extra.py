"""
download_extra.py — Télécharge tous les datasets bonus (ENSO, NAO, AMO, PDO, OWID disasters)
Tout est public, pas d'auth. Sortie : data/cleaned/*.csv au format (Pays, Annee, Valeur) ou (Annee, ...).
"""
import os
import io
import sys
import urllib.request
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

OUT = "data/cleaned"
os.makedirs(OUT, exist_ok=True)

def fetch(url, timeout=60):
    print(f"  → {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


def parse_noaa_table(text, name):
    """Format NOAA standard : YEAR  JAN  FEB  ...  DEC  (séparé par espaces)."""
    rows = []
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts or not parts[0].isdigit():
            continue
        if len(parts) < 13:
            continue
        year = int(parts[0])
        try:
            vals = [float(x) for x in parts[1:13]]
            # NOAA missing values often -99.99 or 999.9
            vals = [v for v in vals if abs(v) < 90]
            if len(vals) >= 6:
                rows.append({"Annee": year, "Valeur": float(np.mean(vals))})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    print(f"    {name}: {len(df)} années")
    return df


# ── 1. ONI (El Niño / La Niña) — l'indice climatique le plus important ─────
def get_oni():
    # ONI = Oceanic Niño Index, ERSSTv5
    txt = fetch("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt")
    rows = []
    for line in txt.splitlines()[1:]:
        parts = line.strip().split()
        if len(parts) >= 4:
            try:
                yr = int(parts[1])
                anom = float(parts[3])
                rows.append({"Annee": yr, "Valeur": anom})
            except Exception:
                continue
    df = pd.DataFrame(rows).groupby("Annee", as_index=False)["Valeur"].mean()
    df.to_csv(f"{OUT}/climate_index_oni.csv", index=False)
    print(f"  ONI: {len(df)} années -> climate_index_oni.csv")
    return df


# ── 2. SOI (Southern Oscillation Index) ────────────────────────────────────
def get_soi():
    txt = fetch("https://www.cpc.ncep.noaa.gov/data/indices/soi")
    # Format : ANOMALY puis bloc "STANDARDIZED DATA"
    lines = txt.splitlines()
    start = None
    for i, l in enumerate(lines):
        if "STANDARDIZED" in l.upper() and "DATA" in l.upper():
            start = i
            break
    if start is None:
        start = 0
    sub = "\n".join(lines[start:])
    df = parse_noaa_table(sub, "SOI")
    df.to_csv(f"{OUT}/climate_index_soi.csv", index=False)
    return df


# ── 3. NAO (North Atlantic Oscillation) ────────────────────────────────────
def get_nao():
    txt = fetch("https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table")
    df = parse_noaa_table(txt, "NAO")
    df.to_csv(f"{OUT}/climate_index_nao.csv", index=False)
    return df


# ── 4. AMO (Atlantic Multi-decadal Oscillation) ────────────────────────────
def get_amo():
    txt = fetch("https://psl.noaa.gov/data/correlation/amon.us.long.data")
    df = parse_noaa_table(txt, "AMO")
    df.to_csv(f"{OUT}/climate_index_amo.csv", index=False)
    return df


# ── 5. PDO (Pacific Decadal Oscillation) ───────────────────────────────────
def get_pdo():
    txt = fetch("https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat")
    df = parse_noaa_table(txt, "PDO")
    df.to_csv(f"{OUT}/climate_index_pdo.csv", index=False)
    return df


# ── 6. AO (Arctic Oscillation) ────────────────────────────────────────────
def get_ao():
    txt = fetch("https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table")
    df = parse_noaa_table(txt, "AO")
    df.to_csv(f"{OUT}/climate_index_ao.csv", index=False)
    return df


# ── 7. OWID Natural disasters (mirror EM-DAT) ─────────────────────────────
def get_owid_disasters():
    # OWID a un dataset annuel "Number of deaths from natural disasters"
    candidates = [
        ("number-of-deaths-from-natural-disasters",   "disaster_deaths"),
        ("number-of-natural-disaster-events",         "disaster_events"),
        ("total-affected-by-natural-disasters",       "disaster_affected"),
        ("natural-disasters-economic-damages",        "disaster_damages_usd"),
    ]
    for slug, colname in candidates:
        try:
            url = f"https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=true"
            txt = fetch(url, timeout=90)
            df = pd.read_csv(io.StringIO(txt))
            df.columns = [c.strip() for c in df.columns]
            # Standard schema OWID : Entity, Code, Year, <value column>
            ent = "Entity" if "Entity" in df.columns else df.columns[0]
            yr  = "Year"   if "Year"   in df.columns else df.columns[2]
            val_cols = [c for c in df.columns if c not in (ent, "Code", yr)]
            # Aggrégation toutes catégories de catastrophes (somme)
            if val_cols:
                df["Valeur"] = df[val_cols].select_dtypes(include="number").sum(axis=1, min_count=1)
            else:
                continue
            out = df[[ent, yr, "Valeur"]].rename(columns={ent: "Pays", yr: "Annee"})
            out = out.dropna(subset=["Pays", "Annee", "Valeur"])
            out.to_csv(f"{OUT}/owid_{colname}.csv", index=False)
            print(f"  OWID {colname}: {len(out)} lignes")
        except Exception as e:
            print(f"  ✗ {slug}: {e}")


# ── 8. OWID variables additionnelles utiles ───────────────────────────────
def get_owid_extras():
    extras = [
        ("life-expectancy",                         "owid_life_exp_extra"),
        ("fertility-rate-complete-gapminder",       "owid_fertility"),
        ("share-of-the-population-living-in-urban-agglomerations-of-more-than-1-million-by-country", "owid_urban_meta"),
        ("crop-yields",                             "owid_crop_yields"),
        ("undernourishment-fao",                    "owid_undernourishment"),
        ("share-of-population-undernourished",      "owid_undernourishment_share"),
        ("share-of-the-population-suffering-from-an-eating-disorder", None),  # skip
        ("share-of-individuals-using-the-internet", "owid_internet"),
    ]
    for slug, colname in extras:
        if not colname:
            continue
        try:
            url = f"https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=true"
            txt = fetch(url, timeout=90)
            df = pd.read_csv(io.StringIO(txt))
            df.columns = [c.strip() for c in df.columns]
            ent = "Entity" if "Entity" in df.columns else df.columns[0]
            yr  = "Year"   if "Year"   in df.columns else df.columns[2]
            val_cols = [c for c in df.columns if c not in (ent, "Code", yr)]
            num_cols = [c for c in val_cols if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                continue
            df["Valeur"] = df[num_cols[0]]
            out = df[[ent, yr, "Valeur"]].rename(columns={ent: "Pays", yr: "Annee"}).dropna()
            out.to_csv(f"{OUT}/{colname}.csv", index=False)
            print(f"  OWID {colname}: {len(out)} lignes (colonne {num_cols[0]})")
        except Exception as e:
            print(f"  ✗ {slug}: {e}")


# ── 9. Volcans : éruptions par année (Smithsonian GVP) ─────────────────────
def get_eruptions():
    # Smithsonian GVP — accessible direct
    try:
        url = "https://volcano.si.edu/database/list_volcano_holocene_excel.cfm"
        # API CSV : on essaie
        url2 = "https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/volcanoes/eruptions?download=true"
        txt = fetch(url2, timeout=120)
        df = pd.read_csv(io.StringIO(txt))
        # Colonnes typiques: 'Year', 'Country', 'Vei', 'Name'
        cols_lower = {c.lower(): c for c in df.columns}
        yr = cols_lower.get("year")
        country = cols_lower.get("country")
        vei = cols_lower.get("vei")
        if yr and country:
            out = df[[country, yr] + ([vei] if vei else [])].copy()
            out = out.rename(columns={country: "Pays", yr: "Annee"})
            if vei:
                out = out.rename(columns={vei: "Vei"})
                out["Valeur"] = pd.to_numeric(out["Vei"], errors="coerce").fillna(0)
            else:
                out["Valeur"] = 1
            out = out.dropna(subset=["Pays", "Annee"])
            out["Annee"] = pd.to_numeric(out["Annee"], errors="coerce")
            out = out.dropna(subset=["Annee"])
            out["Annee"] = out["Annee"].astype(int)
            out = out[(out["Annee"] >= 1900) & (out["Annee"] <= 2025)]
            agg = out.groupby(["Pays", "Annee"], as_index=False).agg(
                eruption_count=("Valeur", "count"),
                eruption_vei_max=("Valeur", "max"),
            )
            agg.to_csv(f"{OUT}/eruptions_yearly.csv", index=False)
            print(f"  Éruptions: {len(agg)} (Pays, Année)")
            return
    except Exception as e:
        print(f"  ✗ éruptions: {e}")


# ── 10. CO2 atmosphérique (Mauna Loa, dynamique mondiale) ─────────────────
def get_mauna_loa():
    try:
        url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.csv"
        txt = fetch(url)
        lines = [l for l in txt.splitlines() if l and not l.startswith("#")]
        rows = []
        for l in lines[1:]:
            parts = l.split(",")
            if len(parts) >= 2:
                try:
                    rows.append({"Annee": int(parts[0]), "Valeur": float(parts[1])})
                except Exception:
                    continue
        df = pd.DataFrame(rows)
        df.to_csv(f"{OUT}/global_co2_atmospheric.csv", index=False)
        print(f"  CO2 Mauna Loa: {len(df)} années")
    except Exception as e:
        print(f"  ✗ Mauna Loa: {e}")


# ── 11. Sea level rise mondial (CSIRO/NOAA) ──────────────────────────────
def get_sea_level():
    try:
        url = "https://datahub.io/core/sea-level-rise/r/csiro_alt_gmsl_yr_2015.csv"
        txt = fetch(url)
        df = pd.read_csv(io.StringIO(txt))
        df["Annee"] = pd.to_datetime(df.iloc[:, 0]).dt.year
        df = df.groupby("Annee", as_index=False).mean(numeric_only=True)
        df = df.rename(columns={df.columns[1]: "Valeur"})
        df[["Annee", "Valeur"]].to_csv(f"{OUT}/global_sea_level.csv", index=False)
        print(f"  Sea level: {len(df)} années")
    except Exception as e:
        print(f"  ✗ sea level: {e}")


if __name__ == "__main__":
    print("[1] Indices climatiques NOAA…")
    for fn in (get_oni, get_soi, get_nao, get_amo, get_pdo, get_ao):
        try:
            fn()
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")

    print("[2] OWID — catastrophes naturelles…")
    get_owid_disasters()

    print("[3] OWID — variables additionnelles…")
    get_owid_extras()

    print("[4] Éruptions volcaniques NOAA…")
    get_eruptions()

    print("[5] CO2 atmosphérique mondial…")
    get_mauna_loa()

    print("[6] Niveau des mers global…")
    get_sea_level()

    print("\n[OK] Tous les datasets téléchargeables récupérés dans data/cleaned/")
