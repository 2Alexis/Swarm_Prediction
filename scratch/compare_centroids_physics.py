import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from layer1_engine import Layer1Engine

# Define geometric vs agricultural centroids for large countries
centroids_comparison = {
    'Canada': {
        'geo': (57.5505, -98.4168),
        'agro': (50.5, -105.5) # Southern Plains (Saskatchewan/Manitoba)
    },
    'Algeria': {
        'geo': (28.3510, 2.6558),
        'agro': (36.5, 3.0) # Northern Fertile Coast
    },
    'Australia': {
        'geo': (-25.6973, 134.0228),
        'agro': (-34.5, 143.5) # Southeast Agricultural Region (Murray-Darling)
    },
    'Egypt': {
        'geo': (26.6052, 30.2401),
        'agro': (30.1, 31.2) # Nile Delta / Cairo region
    },
    'Russian Federation': {
        'geo': (59.0394, 98.6705),
        'agro': (52.0, 45.0) # Southwest European Russia
    },
    'China': {
        'geo': (38.0733, 104.6911),
        'agro': (34.5, 114.5) # North China Plain
    },
    'United States': {
        'geo': (38.8208, -96.3316),
        'agro': (41.5, -93.5) # Midwest Corn Belt (Iowa)
    },
    'Brazil': {
        'geo': (-11.5246, -54.3552),
        'agro': (-20.5, -48.5) # Southeast / Center-West agricultural regions
    }
}

engine = Layer1Engine(raw_dir='data/raw')

for country, coords in centroids_comparison.items():
    print(f"\n==================== {country} ====================")
    geo_lat, geo_lon = coords['geo']
    agro_lat, agro_lon = coords['agro']
    
    geo_features = engine.get_physical_features(geo_lat, geo_lon)
    agro_features = engine.get_physical_features(agro_lat, agro_lon)
    
    print(f"Geometric Centroid (Lat: {geo_lat:.2f}, Lon: {geo_lon:.2f}):")
    print(f"  Soil pH: {geo_features['soil_pH']} | Clay: {geo_features['clay_pct']}% | Sand: {geo_features['sand_pct']}%")
    print(f"  Mean Temp: {geo_features['temp_mean']} °C | Precip: {geo_features['precip_mean']} mm")
    print(f"  Dist to Coast: {geo_features['dist_to_coast_km']:.1f} km | Elevation: {geo_features['elevation']:.1f} m")
    
    print(f"Agricultural Centroid (Lat: {agro_lat:.2f}, Lon: {agro_lon:.2f}):")
    print(f"  Soil pH: {agro_features['soil_pH']} | Clay: {agro_features['clay_pct']}% | Sand: {agro_features['sand_pct']}%")
    print(f"  Mean Temp: {agro_features['temp_mean']} °C | Precip: {agro_features['precip_mean']} mm")
    print(f"  Dist to Coast: {agro_features['dist_to_coast_km']:.1f} km | Elevation: {agro_features['elevation']:.1f} m")
