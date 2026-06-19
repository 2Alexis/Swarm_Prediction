import pandas as pd

centroids = pd.read_csv('data/cleaned/country_centroids.csv')
features = pd.read_csv('data/cleaned/country_physical_features.csv')

print("=== country_centroids.csv ===")
print("Shape:", centroids.shape)
print("Columns:", centroids.columns.tolist())
print(centroids.head(3))

print("\n=== country_physical_features.csv ===")
print("Shape:", features.shape)
print("Columns:", features.columns.tolist()[:10], "... (total columns:", len(features.columns), ")")
print("Last column name:", features.columns[-1])
print(features[['Pays', 'latitude', 'longitude']].head(5))

# Check intersection of country names
centroids_countries = set(centroids['COUNTRY'].unique())
features_countries = set(features['Pays'].unique())
print(f"\nCentroids countries count: {len(centroids_countries)}")
print(f"Features countries count: {len(features_countries)}")
print(f"Common countries count: {len(centroids_countries.intersection(features_countries))}")

# Check mismatches
print("\nCountries in features but not centroids:")
print(features_countries - centroids_countries)
