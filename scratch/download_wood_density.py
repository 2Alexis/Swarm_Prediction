import urllib.request
import os

dest = r"c:\Users\alexi\Desktop\Swarm_Prediction\data\raw\ecologie_biomasse\wood_density.csv"
url = "https://raw.githubusercontent.com/higuchip/FT_database/master/wood_density.csv"

try:
    print(f"Downloading wood density from: {url}")
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response:
        with open(dest, 'wb') as out_file:
            out_file.write(response.read())
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"Success! Saved to {dest} ({size_mb:.2f} MB)")
    
    # Print the first few lines to verify
    with open(dest, 'r', encoding='utf-8', errors='ignore') as f:
        print("First 3 lines:")
        for _ in range(3):
            print("  ", f.readline().strip())
except Exception as e:
    print(f"Error: {str(e)}")
