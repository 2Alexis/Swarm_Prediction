import sys
import os
import pandas as pd
import numpy as np

# Import the engine
from layer1_engine import Layer1Engine

engine = Layer1Engine(raw_dir='data/raw')

test_locations = [
    {"name": "Paris (Zone Temperée)", "lat": 48.8566, "lon": 2.3522},
    {"name": "Manaus (Zone Equatoriale)", "lat": -3.1190, "lon": -60.0217},
    {"name": "Le Caire (Zone Aride)", "lat": 30.0444, "lon": 31.2357}
]

results = []
for loc in test_locations:
    res = engine.get_physical_features(loc["lat"], loc["lon"])
    res["Name"] = loc["name"]
    results.append(res)

df_sim = pd.DataFrame(results)
print("Columns in df_sim:", list(df_sim.columns))
