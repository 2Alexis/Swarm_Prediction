"""
optional_loaders.py — Loaders qui s'activent si les fichiers sont présents dans data/raw/.

Datasets supportés :
  - SPAM 2020 (IFPRI) — rendements 10km par culture
  - FAO GAEZ 4.0 — potentiel agro-écologique par culture
  - ESA CCI Soil Moisture — humidité du sol annuelle
  - EM-DAT (full) — base catastrophes complète (login requis)

Lancement :
    python optional_loaders.py
"""
import os, sys, io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
RAW = "data/raw"
OUT = "data/cleaned"

# ── SPAM 2020 (IFPRI) ──────────────────────────────────────────────────────
# URL : https://www.mapspam.info/data/
# Format attendu : data/raw/SPAM/spam2020_v1r0_global_Y_*.tif (rendements GeoTIFF par culture)
def load_spam():
    spam_dir = f"{RAW}/SPAM"
    if not os.path.isdir(spam_dir):
        print("  SPAM : dossier absent, skip. Pour activer :")
        print("    1. Télécharger sur https://www.mapspam.info/data/")
        print("    2. Décompresser dans data/raw/SPAM/")
        print("    3. Relancer optional_loaders.py")
        return
    tifs = [f for f in os.listdir(spam_dir) if f.endswith(".tif") and "Y_" in f]
    print(f"  SPAM : {len(tifs)} rasters de rendement détectés")
    # TODO : zonal stats par pays via shapefile
    print("  (intégration zonal-stats nécessite rasterio + geopandas)")


# ── FAO GAEZ 4.0 ───────────────────────────────────────────────────────────
def load_gaez():
    gaez_dir = f"{RAW}/GAEZ"
    if not os.path.isdir(gaez_dir):
        print("  GAEZ : dossier absent, skip. Pour activer :")
        print("    1. Télécharger sur https://gaez.fao.org/pages/data-access-download")
        print("    2. Décompresser dans data/raw/GAEZ/")
        return
    print(f"  GAEZ : {len(os.listdir(gaez_dir))} fichiers détectés")


# ── ESA CCI Soil Moisture ──────────────────────────────────────────────────
def load_smap():
    sm_dir = f"{RAW}/ESA_CCI_SM"
    if not os.path.isdir(sm_dir):
        print("  ESA CCI SM : dossier absent, skip. Pour activer :")
        print("    1. S'inscrire sur https://esa-soilmoisture-cci.org/")
        print("    2. Télécharger les NetCDF annuels dans data/raw/ESA_CCI_SM/")
        return
    ncs = [f for f in os.listdir(sm_dir) if f.endswith(".nc")]
    print(f"  ESA CCI SM : {len(ncs)} NetCDF détectés")
    if not ncs:
        return
    import xarray as xr
    from scipy.spatial import cKDTree
    centroids = pd.read_csv(f"{OUT}/country_centroids.csv", keep_default_na=False)
    centroids = centroids.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    tree = cKDTree(centroids[["latitude", "longitude"]].values)
    iso = centroids["ISO"].values
    rows = []
    for nc in sorted(ncs):
        try:
            yr = int("".join(c for c in nc if c.isdigit())[:4])
            ds = xr.open_dataset(f"{sm_dir}/{nc}")
            # Variable cible standard : 'sm' (volumetric soil moisture)
            var = "sm" if "sm" in ds.data_vars else list(ds.data_vars)[0]
            # Moyenne annuelle par centroïde (nearest grid cell)
            for i, (lat, lon) in enumerate(centroids[["latitude", "longitude"]].values):
                try:
                    val = float(ds[var].sel(lat=lat, lon=lon, method="nearest").mean().values)
                    rows.append({"ISO": iso[i], "Annee": yr, "soil_moisture_m3m3": val})
                except Exception:
                    pass
            ds.close()
        except Exception as e:
            print(f"   ✗ {nc}: {e}")
    if rows:
        sm = pd.DataFrame(rows).groupby(["ISO", "Annee"], as_index=False)["soil_moisture_m3m3"].mean()
        sm.to_csv(f"{OUT}/soil_moisture_yearly.csv", index=False)
        print(f"   → soil_moisture_yearly.csv ({len(sm)} (ISO, Année))")


# ── EM-DAT (full) ──────────────────────────────────────────────────────────
def load_emdat():
    path = f"{RAW}/emdat_public.xlsx"
    if not os.path.exists(path):
        print("  EM-DAT : fichier absent. Pour activer :")
        print("    1. Créer un compte gratuit sur https://www.emdat.be/")
        print("    2. Télécharger 'EM-DAT public xlsx' → data/raw/emdat_public.xlsx")
        return
    df = pd.read_excel(path, sheet_name=0, skiprows=6)
    df.columns = [c.strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}
    yr  = cols.get("year") or cols.get("start year")
    iso = cols.get("iso") or cols.get("country iso")
    typ = cols.get("disaster type") or cols.get("type")
    dead = cols.get("total deaths") or cols.get("deaths")
    aff = cols.get("total affected") or cols.get("affected")
    dmg = cols.get("total damages, adjusted ('000 us$)") or cols.get("damages")
    sub = df[[iso, yr, typ, dead, aff, dmg]].rename(columns={
        iso: "ISO", yr: "Annee", typ: "Type",
        dead: "deaths", aff: "affected", dmg: "damages_k_usd"})
    sub["Annee"] = pd.to_numeric(sub["Annee"], errors="coerce").astype("Int64")
    sub = sub.dropna(subset=["ISO", "Annee"])
    sub["Annee"] = sub["Annee"].astype(int)
    agg = sub.groupby(["ISO", "Annee", "Type"]).agg(
        deaths=("deaths", "sum"),
        affected=("affected", "sum"),
        damages_k_usd=("damages_k_usd", "sum"),
        events=("deaths", "count"),
    ).reset_index()
    # Pivot par type
    piv = agg.pivot_table(index=["ISO", "Annee"], columns="Type",
                          values=["deaths", "affected", "events"],
                          aggfunc="sum", fill_value=0)
    piv.columns = [f"emdat_{m}_{t}".lower().replace(" ", "_") for m, t in piv.columns]
    piv = piv.reset_index()
    piv.to_csv(f"{OUT}/emdat_yearly_full.csv", index=False)
    print(f"  EM-DAT : {len(piv)} (ISO, Année), {piv.shape[1]-2} colonnes par type")


if __name__ == "__main__":
    print("[Optional loaders] Vérification des datasets bonus…\n")
    load_spam()
    print()
    load_gaez()
    print()
    load_smap()
    print()
    load_emdat()
    print("\n[Fin]")
