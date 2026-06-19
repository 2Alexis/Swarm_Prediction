import os

wc_dir = r"c:\Users\alexi\Desktop\Swarm_Prediction\data\raw\WorldClim"
print("Scanning WorldClim directory:", wc_dir)

tif_files = []
for root, dirs, files in os.walk(wc_dir):
    for f in files:
        if f.endswith('.tif'):
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, wc_dir)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            tif_files.append((rel_path, size_mb))

print(f"Found {len(tif_files)} GeoTIFF files.")
for rel_path, size in sorted(tif_files)[:20]:
    print(f"  - {rel_path} ({size:.2f} MB)")
if len(tif_files) > 20:
    print(f"  ... and {len(tif_files) - 20} more.")
