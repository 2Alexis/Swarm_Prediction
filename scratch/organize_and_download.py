import os
import shutil
import urllib.request
import sys

raw_dir = r"c:\Users\alexi\Desktop\Swarm_Prediction\data\raw"

# Define category folders
categories = {
    "climat": os.path.join(raw_dir, "atmosphere_climat"),
    "relief": os.path.join(raw_dir, "topographie_relief"),
    "eau": os.path.join(raw_dir, "hydrologie_eau"),
    "sols": os.path.join(raw_dir, "pedologie_sols"),
    "geologie": os.path.join(raw_dir, "geologie_risques"),
    "ecologie": os.path.join(raw_dir, "ecologie_biomasse"),
    "socio_economie": os.path.join(raw_dir, "socio_economie_demographie")
}

# 1. Create directories
print("--- Step 1: Creating category directories ---")
for cat_name, cat_path in categories.items():
    os.makedirs(cat_path, exist_ok=True)
    print(f"Created: {cat_path}")

# 2. Reorganize existing folders/files in data/raw
print("\n--- Step 2: Sorting existing datasets ---")

def move_folder(src_rel, dest_cat):
    src_path = os.path.join(raw_dir, src_rel)
    if os.path.exists(src_path):
        dest_path = os.path.join(categories[dest_cat], os.path.basename(src_rel))
        if os.path.exists(dest_path):
            if os.path.isdir(dest_path):
                shutil.rmtree(dest_path)
            else:
                os.remove(dest_path)
        shutil.move(src_path, categories[dest_cat])
        print(f"  Moved folder: {src_rel} -> {dest_cat}/")
    else:
        print(f"  Folder not found (skipping): {src_rel}")

def move_file(src_rel, dest_cat):
    src_path = os.path.join(raw_dir, src_rel)
    if os.path.exists(src_path):
        dest_path = os.path.join(categories[dest_cat], os.path.basename(src_rel))
        if os.path.exists(dest_path):
            os.remove(dest_path)
        shutil.move(src_path, categories[dest_cat])
        print(f"  Moved file: {src_rel} -> {dest_cat}/")
    else:
        print(f"  File not found (skipping): {src_rel}")

# Move climat datasets
move_folder(r"faostat\temperature", "climat")
move_folder(r"faostat\climat_indicateurs", "climat")
move_folder(r"faostat\emissions_ges", "climat")

# Move sols datasets
move_folder(r"faostat\sol_nutritif", "sols")

# Move ecologie datasets
move_folder(r"faostat\forets", "ecologie")
move_file(r"worldbank\wb_forest_area_pct.csv", "ecologie")

# Move eau datasets
move_file(r"worldbank\wb_freshwater_withdrawal_pct.csv", "eau")

# Move all other files/folders under faostat, worldbank, who, owid, acled, un_population to socio_economie
for item in os.listdir(raw_dir):
    item_path = os.path.join(raw_dir, item)
    # Don't move the category folders themselves!
    if item in categories.keys() or item_path in categories.values():
        continue
    
    # Check if folder or file
    if os.path.isdir(item_path):
        dest_path = os.path.join(categories["socio_economie"], item)
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.move(item_path, categories["socio_economie"])
        print(f"  Moved folder: {item} -> socio_economie/")
    else:
        dest_path = os.path.join(categories["socio_economie"], item)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        shutil.move(item_path, categories["socio_economie"])
        print(f"  Moved file: {item} -> socio_economie/")

# 3. Download new datasets
print("\n--- Step 3: Downloading new physical datasets ---")

downloads = [
    {
        "url": "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-05-12/volcano.csv",
        "dest": os.path.join(categories["geologie"], "volcanoes.csv"),
        "name": "Active Volcanoes Database (Smithsonian GVP)"
    },
    {
        "url": "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&minmagnitude=5&starttime=2015-01-01",
        "dest": os.path.join(categories["geologie"], "earthquakes_usgs.csv"),
        "name": "Earthquakes M5.0+ Since 2015 (USGS Live)"
    },
    {
        "url": "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_boundaries.json",
        "dest": os.path.join(categories["geologie"], "tectonic_plates.geojson"),
        "name": "Tectonic Plate Boundaries (Fraxen/USGS)"
    },
    {
        "url": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_rivers_lake_centerlines.geojson",
        "dest": os.path.join(categories["eau"], "rivers.geojson"),
        "name": "Major Rivers & Lakes Centerlines (Natural Earth)"
    },
    {
        "url": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_lakes.geojson",
        "dest": os.path.join(categories["eau"], "lakes.geojson"),
        "name": "Major Lakes (Natural Earth)"
    },
    {
        "url": "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_coastline.geojson",
        "dest": os.path.join(categories["relief"], "coastlines.geojson"),
        "name": "Global Coastlines (Natural Earth)"
    },
    {
        "url": "https://raw.githubusercontent.com/ajkellerstein/wood-density/master/wood_density.csv",
        "dest": os.path.join(categories["ecologie"], "wood_density.csv"),
        "name": "Global Wood Density Database"
    },
    {
        "url": "https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv",
        "dest": os.path.join(categories["geologie"], "global_power_plants.csv"),
        "name": "Global Power Plants Database (WRI)"
    },
    {
        "url": "https://zenodo.org/records/5882203/files/earth-topography-10arcmin.nc",
        "dest": os.path.join(categories["relief"], "earth-topography-10arcmin.nc"),
        "name": "Earth Topography 10-arcminute Grid (ETOPO1/Zenodo)"
    }
]

for dl in downloads:
    print(f"Downloading {dl['name']}...")
    try:
        req = urllib.request.Request(
            dl["url"], 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            with open(dl["dest"], 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        size_mb = os.path.getsize(dl["dest"]) / (1024 * 1024)
        print(f"  Saved to: {dl['dest']} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"  Error downloading {dl['name']}: {str(e)}")

print("\n Reorganization and download complete!")
