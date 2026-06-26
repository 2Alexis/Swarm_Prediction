"""
fetch_more_datasets.py — WHO Air Quality + GFW Hansen + WorldPop pre-aggregates.
"""
import os, sys, io, urllib.request, urllib.parse, json
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = "data/cleaned"

def fetch_text(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


# ── 1. WHO Air Quality via GHO OData API (gratuit, no-auth) ──────────────
print("[1] WHO Air Quality DB…")
WHO_INDICATORS = {
    # Code WHO GHO -> nom court
    "SDGPM25":                    "who_pm25_mean",
    "AIR_5":                      "who_pm10_urban",
    "AIR_31":                     "who_pm25_urban",
    "AIR_32":                     "who_pm10_urban_alt",
    "WSH_AIR_HOUSEHOLD":          "who_household_air_solid_fuels",
    "AIR_10":                     "who_air_pollution_deaths",
    "AIR_41":                     "who_ambient_air_deaths_rate",
}

def fetch_who(code):
    try:
        url = f"https://ghoapi.azureedge.net/api/{code}?$format=json"
        data = json.loads(fetch_text(url))
        rows = []
        for r in data.get("value", []):
            yr = r.get("TimeDim") or r.get("YearCode")
            spat = r.get("SpatialDimType")
            iso = r.get("SpatialDim") if spat == "COUNTRY" else None
            val = r.get("NumericValue") or r.get("Value")
            if not iso or yr is None or val in (None, ""): continue
            try:
                rows.append({"ISO3": iso, "Annee": int(yr), "Valeur": float(val)})
            except: continue
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"   ✗ {code}: {e}")
        return None

import pycountry
def iso3_to_iso2(c):
    try:
        co = pycountry.countries.get(alpha_3=c)
        return co.alpha_2 if co else None
    except: return None

for code, name in WHO_INDICATORS.items():
    df = fetch_who(code)
    if df is None or df.empty:
        print(f"   ✗ {code}")
        continue
    df["ISO"] = df["ISO3"].apply(iso3_to_iso2)
    df = df.dropna(subset=["ISO"])
    df = df.groupby(["ISO", "Annee"], as_index=False)["Valeur"].mean().rename(columns={"Valeur": name})
    df.to_csv(f"{D}/{name}.csv", index=False)
    print(f"   + {name}: {len(df)} (ISO,Année)")


# ── 2. GFW Hansen tree cover loss — direct CSV ──────────────────────────
print("\n[2] GFW Hansen tree cover loss…")
# GFW publie un CSV national : https://data.globalforestwatch.org/
# Endpoint API : https://production-api.globalforestwatch.org/
# Approche directe : utiliser leur "datapi" pour les chiffres pays
try:
    # Hansen lossyear par pays via Global Forest Watch datasets v20240114 ou similaire
    # Tentative via Datamart CSV
    url = "https://data-api.globalforestwatch.org/dataset/umd_tree_cover_loss/v1.10/download/csv?sql=SELECT iso, treecoverloss__year, SUM(area__ha) FROM data WHERE umd_tree_cover_density__threshold = '30' GROUP BY iso, treecoverloss__year"
    txt = fetch_text(url, timeout=300)
    df = pd.read_csv(io.StringIO(txt))
    df.columns = [c.strip() for c in df.columns]
    print(f"   GFW DataAPI: {df.shape}")
    df = df.rename(columns={"iso": "ISO3", "treecoverloss__year": "Annee",
                            "sum": "tree_cover_loss_ha"})
    df["ISO"] = df["ISO3"].apply(iso3_to_iso2)
    df = df.dropna(subset=["ISO"])
    df = df[["ISO", "Annee", "tree_cover_loss_ha"]]
    df.to_csv(f"{D}/gfw_tree_cover_loss.csv", index=False)
    print(f"   + Hansen tree loss: {len(df)} (ISO,Année)")
except Exception as e:
    print(f"   ✗ GFW DataAPI: {e}")
    # Plan B : Notre-World-in-Data mirror du Hansen
    try:
        url = "https://ourworldindata.org/grapher/annual-deforestation.csv?v=1&csvType=full&useColumnShortNames=true"
        txt = fetch_text(url)
        df = pd.read_csv(io.StringIO(txt))
        df.columns = [c.strip() for c in df.columns]
        from build_dataset import get_english_iso
        df["ISO"] = df["Entity"].apply(get_english_iso)
        df = df.dropna(subset=["ISO"])
        # Find value column
        val_col = [c for c in df.columns if c not in ("Entity", "Code", "Year", "ISO")][0]
        df = df[["ISO", "Year", val_col]].rename(columns={"Year": "Annee", val_col: "deforestation_ha"})
        df.to_csv(f"{D}/owid_deforestation.csv", index=False)
        print(f"   + OWID deforestation: {len(df)} (ISO,Année)")
    except Exception as e2:
        print(f"   ✗ OWID alt: {e2}")


# ── 3. WorldPop pre-aggregated country totals ─────────────────────────────
print("\n[3] WorldPop pre-aggregates…")
# WorldPop a un CSV national "Population estimates by age 2000-2020"
# URL : https://hub.worldpop.org/doi/10.5258/SOTON/WP00660
# Plus simple : WorldPop API endpoint
try:
    # Try WorldPop REST API
    url = "https://api.worldpop.org/v1/services/stats?dataset=wpgpas&year=2020&iso3=USA"
    txt = fetch_text(url, timeout=30)
    print(f"   WorldPop API OK: {txt[:200]}")
    # This is per-country, async — would be 240 calls × multiple years. Skip for now.
    print("   WorldPop API per-country/per-year trop volumineux — utilisation de WB Population uniquement (déjà en V4)")
except Exception as e:
    print(f"   ✗ WorldPop API: {e}")

# Plan B : OWID demographic detailed (age structure)
OWID_DEMO = [
    ("population-by-age-group", "owid_pop_by_age"),
    ("rural-population-share-of-total-population", "owid_rural_pop_pct"),
    ("urbanization-vs-gdp", None),
    ("population-density", "owid_pop_density"),
    ("number-of-births-per-year", "owid_births"),
    ("number-of-deaths-per-year", "owid_deaths"),
    ("annual-number-of-deaths-by-cause", "owid_deaths_by_cause"),
    ("crude-birth-rate", "owid_crude_birth_rate"),
    ("crude-death-rate", "owid_crude_death_rate"),
    ("net-migration-rate", "owid_net_migration_rate"),
]

def owid_csv(slug):
    url = f"https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=true"
    txt = fetch_text(url, timeout=120)
    df = pd.read_csv(io.StringIO(txt))
    df.columns = [c.strip() for c in df.columns]
    return df

ok = 0
for slug, name in OWID_DEMO:
    if not name: continue
    try:
        df = owid_csv(slug)
        ent = "Entity" if "Entity" in df.columns else df.columns[0]
        yr  = "Year"   if "Year"   in df.columns else df.columns[2]
        val_cols = [c for c in df.columns if c not in (ent, "Code", yr)]
        num_cols = [c for c in val_cols if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols: continue
        out = df[[ent, yr, num_cols[0]]].rename(columns={ent: "Pays", yr: "Annee", num_cols[0]: "Valeur"})
        out = out.dropna(subset=["Pays", "Annee", "Valeur"])
        out.to_csv(f"{D}/{name}.csv", index=False)
        print(f"   + {name}: {len(out)} lignes")
        ok += 1
    except Exception as e:
        print(f"   ✗ {slug}: {e}")
print(f"   → {ok}/{len(OWID_DEMO)} OWID demo OK")


# ── 4. Additional OWID for low-R² targets ────────────────────────────────
print("\n[4] Datasets supplémentaires pour fruits/légumineuses…")
OWID_AGRI = [
    ("vegetable-production", "owid_vegetable_production"),
    ("fruit-production", "owid_fruit_production"),
    ("legume-production-tonnes", "owid_legume_production"),
    ("yields-of-crops", "owid_crop_yields_all"),
    ("share-of-land-area-used-for-agriculture", "owid_agri_land_share"),
    ("pesticide-use-per-hectare", "owid_pesticide_per_ha"),
    ("fertilizer-application-per-hectare", "owid_fertilizer_per_ha"),
    ("share-of-individuals-with-electricity-access", "owid_elec_access_extra"),
    ("food-emissions-by-country", "owid_food_emissions"),
    ("crop-area-vs-yield", None),
]
ok = 0
for slug, name in OWID_AGRI:
    if not name: continue
    try:
        df = owid_csv(slug)
        ent = "Entity" if "Entity" in df.columns else df.columns[0]
        yr  = "Year"   if "Year"   in df.columns else df.columns[2]
        val_cols = [c for c in df.columns if c not in (ent, "Code", yr)]
        num_cols = [c for c in val_cols if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols: continue
        out = df[[ent, yr, num_cols[0]]].rename(columns={ent: "Pays", yr: "Annee", num_cols[0]: "Valeur"})
        out = out.dropna(subset=["Pays", "Annee", "Valeur"])
        out.to_csv(f"{D}/{name}.csv", index=False)
        print(f"   + {name}: {len(out)} lignes")
        ok += 1
    except Exception as e:
        print(f"   ✗ {slug}: {e}")
print(f"   → {ok}/{len(OWID_AGRI)} OK")

print("\n[DONE] WHO + GFW + WorldPop/OWID extras finis.")
