"""
add_worldclim_bio.py — Échantillonne BIO 2..4, 7..11, 13..14, 16..19 (les manquants)
sur la bbox de chaque pays et sauvegarde en CSV (statique par ISO).

Variables ajoutées (signal saisonnier important) :
  BIO2  : Mean Diurnal Range
  BIO3  : Isothermality
  BIO4  : Temperature Seasonality
  BIO7  : Temperature Annual Range
  BIO8  : Mean T of Wettest Quarter
  BIO9  : Mean T of Driest Quarter
  BIO10 : Mean T of Warmest Quarter (= growing season)
  BIO11 : Mean T of Coldest Quarter
  BIO13 : Precip of Wettest Month
  BIO14 : Precip of Driest Month (drought signal !)
  BIO16 : Precip of Wettest Quarter
  BIO17 : Precip of Driest Quarter (water stress !)
  BIO18 : Precip of Warmest Quarter (growing season precip)
  BIO19 : Precip of Coldest Quarter
"""
import os, sys, io
import numpy as np
import pandas as pd
from PIL import Image
import xarray as xr
import math

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

D = "data/cleaned"
WC = "data/raw/WorldClim/biovar"
RELIEF = "data/raw/topographie_relief"

from build_dataset import custom_mappings, agricultural_centroids

# ── Charger le dataset v3 pour récupérer (ISO, area, lat, lon) déjà résolus
df_v3 = pd.read_csv(f"{D}/dataset_final_v3.csv")
df_v3 = df_v3.dropna(subset=["ISO"]).copy()
df_v3["ISO"] = df_v3["ISO"].astype(str)

iso_lat = df_v3.groupby("ISO")["latitude"].first().to_dict()
iso_lon = df_v3.groupby("ISO")["longitude"].first().to_dict()
iso_area = df_v3.groupby("ISO")["area_km2"].first().to_dict()

# ETOPO pour le land mask
ETOPO = xr.open_dataset(f"{RELIEF}/earth-topography-10arcmin.nc")


def bbox(lat, lon, area_km2):
    side = math.sqrt(max(area_km2, 1.0))
    half_lat = min(25.0, (side / 2) / 111.0)
    cos_lat = max(0.2, math.cos(math.radians(lat)))
    half_lon = min(35.0, (side / 2) / (111.0 * cos_lat))
    return lat - half_lat, lat + half_lat, lon - half_lon, lon + half_lon


def read_bbox(tif_path, lat_min, lat_max, lon_min, lon_max):
    with Image.open(tif_path) as img:
        w, h = img.size
        c_min = max(0, int((lon_min + 180) / 360 * w))
        c_max = min(w, int((lon_max + 180) / 360 * w) + 1)
        r_min = max(0, int((90 - lat_max) / 180 * h))
        r_max = min(h, int((90 - lat_min) / 180 * h) + 1)
        if c_max <= c_min or r_max <= r_min:
            return None
        crop = img.crop((c_min, r_min, c_max, r_max))
        arr = np.array(crop, dtype=np.float32)
    arr = np.where((arr < -10000) | (arr > 1e10) | np.isnan(arr), np.nan, arr)
    return arr


BIO_VARS = {
    "bio_2.tif":  "bio2_mean_diurnal_range",
    "bio_3.tif":  "bio3_isothermality",
    "bio_4.tif":  "bio4_temp_seasonality",
    "bio_7.tif":  "bio7_temp_annual_range",
    "bio_8.tif":  "bio8_temp_wettest_quarter",
    "bio_9.tif":  "bio9_temp_driest_quarter",
    "bio_10.tif": "bio10_temp_warmest_quarter",
    "bio_11.tif": "bio11_temp_coldest_quarter",
    "bio_13.tif": "bio13_precip_wettest_month",
    "bio_14.tif": "bio14_precip_driest_month",
    "bio_16.tif": "bio16_precip_wettest_quarter",
    "bio_17.tif": "bio17_precip_driest_quarter",
    "bio_18.tif": "bio18_precip_warmest_quarter",
    "bio_19.tif": "bio19_precip_coldest_quarter",
}

print("Échantillonnage des BIO 2-19 manquants pour chaque pays…")
rows = []
isos = sorted(iso_lat.keys())
for i, iso in enumerate(isos, 1):
    lat = iso_lat[iso]
    lon = iso_lon[iso]
    area = iso_area.get(iso) or 100_000
    if pd.isna(lat) or pd.isna(lon):
        continue
    lat_min, lat_max, lon_min, lon_max = bbox(lat, lon, area)

    row = {"ISO": iso}
    for tif_name, colname in BIO_VARS.items():
        # Cherche le fichier dans biovar/
        path = None
        for f in os.listdir(WC):
            if f.endswith(".tif") and tif_name.replace(".tif", "") in f:
                path = f"{WC}/{f}"
                break
        if path is None or not os.path.exists(path):
            row[colname] = None
            continue
        arr = read_bbox(path, lat_min, lat_max, lon_min, lon_max)
        if arr is None or np.all(np.isnan(arr)):
            row[colname] = None
        else:
            row[colname] = float(np.nanmean(arr))
    rows.append(row)
    if i % 30 == 0 or i == len(isos):
        print(f"  {i}/{len(isos)} pays")

bio_df = pd.DataFrame(rows)
bio_df.to_csv(f"{D}/worldclim_bio_extra.csv", index=False)
print(f"\n[OK] {len(bio_df)} pays × {len(BIO_VARS)} variables BIO -> worldclim_bio_extra.csv")
