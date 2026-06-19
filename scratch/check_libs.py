libs = ['babel', 'countryinfo', 'pycountry', 'geopandas', 'geopy', 'fiona', 'shapely']
for lib in libs:
    try:
        __import__(lib)
        print(f"  [YES] {lib} is installed.")
    except ImportError:
        print(f"  [NO] {lib} is NOT installed.")
