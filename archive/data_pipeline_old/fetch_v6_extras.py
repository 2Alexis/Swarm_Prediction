"""
fetch_v6_extras.py — Téléchargements V6 (CRU TS via OWID + catastrophes par type via OWID).
"""
import os, sys, io, urllib.request
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = "data/cleaned"

def fetch(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")

def owid_csv(slug, name):
    try:
        url = f"https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=true"
        txt = fetch(url, timeout=120)
        df = pd.read_csv(io.StringIO(txt))
        df.columns = [c.strip() for c in df.columns]
        ent = "Entity" if "Entity" in df.columns else df.columns[0]
        yr  = "Year"   if "Year"   in df.columns else df.columns[2]
        val_cols = [c for c in df.columns if c not in (ent, "Code", yr)]
        num_cols = [c for c in val_cols if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            return False
        # If multiple numeric cols (per disaster type) save WIDE
        if len(num_cols) > 1:
            out = df[[ent, yr] + num_cols].rename(columns={ent: "Pays", yr: "Annee"})
        else:
            out = df[[ent, yr, num_cols[0]]].rename(columns={ent: "Pays", yr: "Annee", num_cols[0]: "Valeur"})
        out = out.dropna(subset=["Pays", "Annee"])
        out.to_csv(f"{D}/{name}.csv", index=False)
        cols_str = ",".join(num_cols)
        print(f"  + {name}: {len(out)} lignes / cols={cols_str}")
        return True
    except Exception as e:
        print(f"  ✗ {slug}: {e}")
        return False


# ── 1. CRU TS via OWID (températures historiques par pays) ─────────────────
print("[1] CRU TS / temperature anomalies via OWID…")
CRU_SLUGS = [
    ("average-monthly-surface-temperature",   "owid_avg_monthly_temp"),
    ("annual-temperature-anomalies",          "owid_annual_temp_anomaly"),
    ("annual-temperature-anomalies-by-decade","owid_annual_temp_anomaly_decade"),
    ("monthly-temperature-anomalies",         "owid_monthly_temp_anomaly"),
    ("annual-precipitation-cru",              "owid_annual_precip_cru"),
    ("co2-emissions-per-capita",              "owid_co2_per_capita"),
    ("ghg-emissions-by-sector",               "owid_ghg_by_sector"),
    ("methane-emissions",                     "owid_methane"),
    ("nitrous-oxide-emissions",               "owid_nitrous_oxide"),
    ("water-stress-by-country",               "owid_water_stress_by_country"),
    ("share-of-water-stress",                 None),
]
for slug, name in CRU_SLUGS:
    if not name: continue
    owid_csv(slug, name)


# ── 2. Catastrophes par type (équivalent EM-DAT) via OWID ──────────────────
print("\n[2] Catastrophes ventilées par type via OWID…")
DISASTER_TYPES = [
    ("number-of-deaths-from-drought",                  "disaster_deaths_drought"),
    ("number-of-deaths-from-flood",                    "disaster_deaths_flood"),
    ("number-of-deaths-from-earthquakes",              "disaster_deaths_earthquake"),
    ("number-of-deaths-from-storms",                   "disaster_deaths_storm"),
    ("number-of-deaths-from-wildfires",                "disaster_deaths_wildfire"),
    ("number-of-deaths-from-volcanic-activity",        "disaster_deaths_volcano"),
    ("number-of-deaths-from-extreme-temperatures",     "disaster_deaths_extreme_temp"),
    ("number-of-deaths-from-landslides",               "disaster_deaths_landslide"),
    ("number-affected-by-drought",                     "disaster_affected_drought"),
    ("number-affected-by-floods",                      "disaster_affected_flood"),
    ("number-affected-by-storms",                      "disaster_affected_storm"),
    ("number-affected-by-wildfires",                   "disaster_affected_wildfire"),
    ("number-affected-by-earthquakes",                 "disaster_affected_earthquake"),
    ("number-affected-by-extreme-temperatures",        "disaster_affected_extreme_temp"),
    ("damage-cost-from-drought",                       "disaster_damage_drought"),
    ("damage-cost-from-flooding",                      "disaster_damage_flood"),
    ("damage-cost-from-storms",                        "disaster_damage_storm"),
]
ok = 0
for slug, name in DISASTER_TYPES:
    if owid_csv(slug, name): ok += 1
print(f"  → {ok}/{len(DISASTER_TYPES)} OK")


# ── 3. Plus de variables démo/santé/économie utiles ─────────────────────────
print("\n[3] Variables additionnelles…")
EXTRAS = [
    ("share-of-deaths-from-pollution",        "owid_deaths_pollution_pct"),
    ("agricultural-output-dollars",           "owid_agri_output_usd"),
    ("share-of-economy-in-services",          "owid_services_share"),
    ("global-hunger-index",                   "owid_ghi"),
    ("undernourishment-share-of-population",  "owid_undernourishment_share"),
    ("vegetable-and-fruit-consumption-per-capita-fao", "owid_fruit_veg_consumption"),
    ("share-of-children-of-school-age-who-are-out-of-school", "owid_oos_children"),
    ("life-expectancy-vs-healthcare-expenditure", None),
    ("share-of-the-population-with-mental-health-disorders", "owid_mental_health"),
    ("share-of-the-rural-population", "owid_rural_share"),
    ("internally-displaced-persons-from-conflict",     "owid_idps_conflict"),
    ("refugees-by-country-of-origin",                   "owid_refugees_origin"),
    ("share-of-population-with-access-to-electricity-rural", "owid_elec_rural"),
    ("cropland-allocation-grains",            None),
    ("population-density-vs-land-use",        None),
]
for slug, name in EXTRAS:
    if name: owid_csv(slug, name)

print("\n[DONE]")
