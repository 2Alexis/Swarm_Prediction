"""
fetch_berkeley_earth.py — Berkeley Earth country temperature anomalies (CSV direct).

Format pour chaque pays :
  http://berkeleyearth.lbl.gov/auto/Regional/TAVG/Text/{country-slug}-TAVG-Trend.txt

Données : anomalie mensuelle 1750-présent par rapport à 1951-1980.
Agrégation : moyennes annuelles + anomalies 10y/30y/60y + tendances décennales.
"""
import os, sys, io, urllib.request, time, re
import pandas as pd
import numpy as np
import pycountry

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = "data/cleaned"

# Récupérer ISO du dataset
df_v6 = pd.read_csv(f"{D}/dataset_final_v6.csv", usecols=["ISO"], low_memory=False)
isos = sorted(df_v6["ISO"].dropna().unique())
print(f"Pays cibles : {len(isos)}")

def country_name_to_slug(name):
    """Berkeley Earth slug = nom anglais en minuscules, espaces → tirets."""
    s = name.lower()
    s = re.sub(r"[^a-z\s\-]", "", s)
    s = s.replace(" ", "-")
    return s

def iso2_to_slug(iso2):
    try:
        co = pycountry.countries.get(alpha_2=iso2)
        if not co:
            return None
        name = co.name
        # Convertir noms spéciaux
        SPECIALS = {
            "United States": "united-states",
            "United Kingdom": "united-kingdom",
            "Russian Federation": "russia",
            "Korea, Republic of": "south-korea",
            "Korea, Democratic People's Republic of": "north-korea",
            "Iran, Islamic Republic of": "iran",
            "Viet Nam": "vietnam",
            "Tanzania, United Republic of": "tanzania",
            "Bolivia, Plurinational State of": "bolivia",
            "Venezuela, Bolivarian Republic of": "venezuela",
            "Czechia": "czech-republic",
            "Türkiye": "turkey",
            "Lao People's Democratic Republic": "laos",
            "Syrian Arab Republic": "syria",
            "Moldova, Republic of": "moldova",
            "Macedonia, the former Yugoslav Republic of": "north-macedonia",
            "Congo, The Democratic Republic of the": "congo-democratic-republic-of-the",
            "Congo": "congo",
            "Côte d'Ivoire": "cote-d-ivoire",
            "Cabo Verde": "cape-verde",
            "Eswatini": "swaziland",
            "Palestine, State of": "palestine",
            "Myanmar": "burma-myanmar",
        }
        return SPECIALS.get(name, country_name_to_slug(name))
    except Exception:
        return None


def fetch_country(slug, timeout=60):
    url = f"https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Regional/TAVG/{slug}-TAVG-Trend.txt"
    # Mirroir AWS si dispo. Sinon retour LBL.
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception:
        url2 = f"http://berkeleyearth.lbl.gov/auto/Regional/TAVG/Text/{slug}-TAVG-Trend.txt"
        req = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="ignore")


def parse_berkeley(text):
    """Parse Berkeley Earth format. Colonnes : Year, Month, Anomaly, Unc, Annual_Anom, Annual_Unc, ..."""
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("%"):
            continue
        parts = line.split()
        if len(parts) < 3: continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            anom = float(parts[2])  # anomalie mensuelle vs 1951-1980
            if abs(anom) > 50:  # missing
                continue
            rows.append({"Year": year, "Month": month, "anom_monthly": anom})
        except Exception:
            continue
    if not rows: return None
    df = pd.DataFrame(rows)
    # Agrégation annuelle
    annual = df.groupby("Year").agg(
        be_t_anom_annual=("anom_monthly", "mean"),
        be_t_anom_min=("anom_monthly", "min"),
        be_t_anom_max=("anom_monthly", "max"),
        be_t_anom_std=("anom_monthly", "std"),
    ).reset_index()
    return annual


OUT_CSV = f"{D}/berkeley_earth_yearly.csv"
done = set()
if os.path.exists(OUT_CSV):
    prev = pd.read_csv(OUT_CSV)
    done = set(prev["ISO"].unique())
    print(f"Déjà fait : {len(done)}")

all_rows = []
t0 = time.time()
errors = 0

def save_partial(rows):
    if not rows: return
    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"Year": "Annee"})
    out = out[["ISO", "Annee", "be_t_anom_annual", "be_t_anom_min", "be_t_anom_max", "be_t_anom_std"]]
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        out = pd.concat([prev, out], ignore_index=True)
    out = out.drop_duplicates(["ISO", "Annee"], keep="last")
    out.to_csv(OUT_CSV, index=False)

for i, iso in enumerate(isos, 1):
    if iso in done: continue
    slug = iso2_to_slug(iso)
    if not slug:
        errors += 1; continue
    try:
        text = fetch_country(slug)
        annual = parse_berkeley(text)
        if annual is None or annual.empty:
            errors += 1; continue
        annual["ISO"] = iso
        all_rows.append(annual)
    except Exception as e:
        errors += 1
        if errors < 20:
            print(f"   ✗ {iso} ({slug}): {e}", flush=True)
    if i % 10 == 0:
        elapsed = time.time() - t0
        eta = (len(isos) - i) * elapsed / max(i, 1) / 60
        print(f"  {i}/{len(isos)}  errs={errors}  ETA={eta:.1f}min", flush=True)
        save_partial(all_rows)
        all_rows = []  # vidé après save

save_partial(all_rows)
if os.path.exists(OUT_CSV):
    final = pd.read_csv(OUT_CSV)
    print(f"\n[OK] {OUT_CSV} : {len(final)} (ISO,Année), {final['ISO'].nunique()} pays, {final['Annee'].min()}-{final['Annee'].max()}")
else:
    print(f"\n[!] Aucun pays récupéré ({errors} erreurs)")
