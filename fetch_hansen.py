"""Fetch GFW Hansen tree cover loss avec URL encodée + plans B."""
import os, sys, io, urllib.request, urllib.parse, json
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = "data/cleaned"

def fetch_text(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")

# Plan A : GFW DataAPI (URL encodée)
print("Plan A : GFW Data API…")
try:
    sql = "SELECT iso, treecoverloss__year, SUM(area__ha) FROM data WHERE umd_tree_cover_density__threshold = '30' GROUP BY iso, treecoverloss__year"
    url = "https://data-api.globalforestwatch.org/dataset/umd_tree_cover_loss/v1.10/download/csv?sql=" + urllib.parse.quote(sql)
    txt = fetch_text(url, timeout=300)
    df = pd.read_csv(io.StringIO(txt))
    print(f"  cols: {list(df.columns)}, rows: {len(df)}")
    df.to_csv(f"{D}/gfw_tree_cover_loss.csv", index=False)
except Exception as e:
    print(f"  ✗ Plan A: {e}")

# Plan B : OWID grapher "share of forest area" + déforestation FAO
print("\nPlan B : OWID forest datasets…")
for slug, name in [
    ("forest-area-as-share-of-land-area",     "owid_forest_share"),
    ("annual-change-forest-area",             "owid_forest_change"),
    ("net-forest-conversion-fao",             "owid_forest_conversion"),
    ("share-global-forest-loss-since-2000",   "owid_global_forest_loss_share"),
    ("tree-cover-loss",                       "owid_tree_cover_loss"),
    ("tree-cover-loss-by-driver",             "owid_tree_loss_driver"),
    ("annual-deforestation",                  "owid_annual_deforestation"),
    ("primary-forest-loss",                   "owid_primary_forest_loss"),
    ("forest-area-km",                        "owid_forest_area_km"),
]:
    try:
        url = f"https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=true"
        txt = fetch_text(url, timeout=120)
        df = pd.read_csv(io.StringIO(txt))
        df.columns = [c.strip() for c in df.columns]
        ent = "Entity" if "Entity" in df.columns else df.columns[0]
        yr  = "Year"   if "Year"   in df.columns else df.columns[2]
        val_cols = [c for c in df.columns if c not in (ent, "Code", yr)]
        num_cols = [c for c in val_cols if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols: continue
        out = df[[ent, yr, num_cols[0]]].rename(columns={ent: "Pays", yr: "Annee", num_cols[0]: "Valeur"})
        out = out.dropna(subset=["Pays", "Annee", "Valeur"])
        out.to_csv(f"{D}/{name}.csv", index=False)
        print(f"  + {name}: {len(out)} lignes ({num_cols[0]})")
    except Exception as e:
        print(f"  ✗ {slug}: {e}")
