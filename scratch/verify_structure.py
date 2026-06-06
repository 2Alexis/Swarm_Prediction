import os

raw_dir = r"c:\Users\alexi\Desktop\Swarm_Prediction\data\raw"
print("Scanning final raw dataset structure:")

categories = [
    "atmosphere_climat",
    "topographie_relief",
    "hydrologie_eau",
    "pedologie_sols",
    "geologie_risques",
    "ecologie_biomasse",
    "socio_economie_demographie"
]

for cat in categories:
    cat_path = os.path.join(raw_dir, cat)
    print("=" * 80)
    print(f"Category: {cat} (Path: data/raw/{cat})")
    if os.path.exists(cat_path):
        contents = os.listdir(cat_path)
        print(f"  Total items: {len(contents)}")
        for item in sorted(contents):
            item_path = os.path.join(cat_path, item)
            if os.path.isdir(item_path):
                sub_contents = os.listdir(item_path)
                print(f"    - [DIR] {item}/ ({len(sub_contents)} items)")
            else:
                size_mb = os.path.getsize(item_path) / (1024 * 1024)
                print(f"    - [FILE] {item} ({size_mb:.2f} MB)")
    else:
        print("  Directory does not exist!")
