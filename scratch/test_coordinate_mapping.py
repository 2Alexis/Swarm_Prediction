from PIL import Image
import numpy as np

tif_path = r"c:\Users\alexi\Desktop\Swarm_Prediction\data\raw\WorldClim\biovar\wc2.1_10m_bio_1.tif"

def get_value_at(lat, lon):
    img = Image.open(tif_path)
    arr = np.array(img)
    
    # 2160 cols, 1080 rows
    # Lon: -180 to 180 -> 0 to 2160
    # Lat: 90 to -90 -> 0 to 1080
    col = int((lon + 180.0) / 360.0 * 2160)
    row = int((90.0 - lat) / 180.0 * 1080)
    
    # Clip to bounds
    col = max(0, min(col, 2159))
    row = max(0, min(row, 1079))
    
    val = arr[row, col]
    return val, row, col

# Paris coordinates
lat, lon = 48.8566, 2.3522
val, r, c = get_value_at(lat, lon)
print(f"Paris (lat={lat}, lon={lon}): val={val:.2f} (row={r}, col={c})")

# Cairo coordinates (hotter)
lat, lon = 30.0444, 31.2357
val, r, c = get_value_at(lat, lon)
print(f"Cairo (lat={lat}, lon={lon}): val={val:.2f} (row={r}, col={c})")

# Montreal coordinates (colder)
lat, lon = 45.5017, -73.5673
val, r, c = get_value_at(lat, lon)
print(f"Montreal (lat={lat}, lon={lon}): val={val:.2f} (row={r}, col={c})")
