"""
fetch_nasa_power.py — NASA POWER annual API par pays/année.

Variables sub-annuelles ESSENTIELLES non-disponibles ailleurs gratuitement :
  - GWETROOT : humidité zone racinaire (THE soil moisture variable for crops)
  - GWETTOP  : humidité top sol
  - GWETPROF : profil d'humidité sol
  - T2M_MEAN/MAX/MIN/RANGE : T quotidiennes agrégées en annuel
  - PRECTOTCORR : précip corrigées
  - ALLSKY_SFC_SW_DWN : radiation solaire
  - RH2M : humidité relative
  - PS : pression surface
  - WS10M : vent
"""
import os, sys, io, json, time, urllib.request
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = "data/cleaned"

# Récupérer les coords de chaque ISO depuis v4
df_v4 = pd.read_csv(f"{D}/dataset_final_v4.csv", usecols=["ISO", "latitude", "longitude"])
df_v4 = df_v4.dropna(subset=["ISO", "latitude", "longitude"]).copy()
df_v4["ISO"] = df_v4["ISO"].astype(str)
coords = df_v4.groupby("ISO").agg(latitude=("latitude", "first"),
                                  longitude=("longitude", "first")).reset_index()
print(f"Pays à traiter : {len(coords)}")

PARAMS = [
    "T2M", "T2M_MAX", "T2M_MIN", "T2M_RANGE",
    "PRECTOTCORR",
    "GWETROOT", "GWETTOP", "GWETPROF",
    "ALLSKY_SFC_SW_DWN", "RH2M", "PS", "WS10M",
]

def fetch_country(lat, lon, start=1990, end=2024, timeout=180):
    """NASA POWER monthly endpoint. Renvoie {param: {YYYYMM: value}}."""
    url = ("https://power.larc.nasa.gov/api/temporal/monthly/point"
           f"?parameters={','.join(PARAMS)}&community=AG"
           f"&longitude={lon:.4f}&latitude={lat:.4f}"
           f"&start={start}&end={end}&format=JSON")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8", errors="ignore"))
    out = data.get("properties", {}).get("parameter", {})
    return out

OUT_CSV = f"{D}/nasa_power_yearly.csv"
done_iso = set()
if os.path.exists(OUT_CSV):
    prev = pd.read_csv(OUT_CSV)
    done_iso = set(prev["ISO"].unique())
    print(f"Déjà téléchargé : {len(done_iso)} pays — reprise.")

rows = []
errors = 0
t0 = time.time()
for i, row in coords.iterrows():
    iso = row["ISO"]
    if iso in done_iso: continue
    try:
        data = fetch_country(row["latitude"], row["longitude"])
        if not data:
            errors += 1; continue
        # data[param][YYYYMM] = value → agréger par année
        # Format clé : "199001", "199002" ... ou "1990" pour le total annuel
        years_data = {}  # {year: {param: [monthly_values]}}
        for p in PARAMS:
            if p not in data: continue
            for k, v in data[p].items():
                if v is None or v <= -900: continue
                # Filtrer entrées annuelles (clé = "ANN" ou 4-digit year)
                if len(str(k)) == 4:
                    year_int = int(k)
                elif len(str(k)) == 6:
                    year_int = int(str(k)[:4])
                else:
                    continue
                years_data.setdefault(year_int, {}).setdefault(p, []).append(float(v))
        for yr_int, pdata in years_data.items():
            rec = {"ISO": iso, "Annee": yr_int}
            for p in PARAMS:
                vals = pdata.get(p, [])
                if vals:
                    if p == "PRECTOTCORR":
                        # précip annuelle = somme mensuelle (mm/jour → mm/an)
                        rec[f"nasa_{p.lower()}"] = float(np.sum(vals)) * 30.4
                    else:
                        rec[f"nasa_{p.lower()}"] = float(np.mean(vals))
            if len(rec) > 2:
                rows.append(rec)
    except Exception as e:
        errors += 1
        print(f"   ✗ {iso}: {e}")
    if i % 10 == 0:
        elapsed = time.time() - t0
        rate = (i+1) / elapsed if elapsed > 0 else 0
        eta = (len(coords) - i - 1) / rate if rate > 0 else 0
        print(f"  {i+1}/{len(coords)} ({iso})  errs={errors}  ETA={eta/60:.1f}min")
        # Save partial every 10 countries
        if rows:
            partial = pd.DataFrame(rows)
            if os.path.exists(OUT_CSV):
                prev = pd.read_csv(OUT_CSV)
                partial = pd.concat([prev, partial], ignore_index=True)
            partial.drop_duplicates(["ISO", "Annee"], keep="last").to_csv(OUT_CSV, index=False)
            rows = []

# Final save
if rows:
    partial = pd.DataFrame(rows)
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        partial = pd.concat([prev, partial], ignore_index=True)
    partial.drop_duplicates(["ISO", "Annee"], keep="last").to_csv(OUT_CSV, index=False)

final = pd.read_csv(OUT_CSV)
print(f"\n[OK] NASA POWER : {len(final)} (ISO,Année), {final['ISO'].nunique()} pays")
print(f"Variables : {[c for c in final.columns if c.startswith('nasa_')]}")
