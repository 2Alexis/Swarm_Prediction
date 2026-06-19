import os
import json
import math
import numpy as np
import pandas as pd
import xarray as xr
import requests
from PIL import Image

class Layer1Engine:
    def __init__(self, raw_dir=r"c:\Users\alexi\Desktop\Swarm_Prediction\data\raw"):
        self.raw_dir = raw_dir
        self.climat_dir = os.path.join(raw_dir, "atmosphere_climat")
        self.relief_dir = os.path.join(raw_dir, "topographie_relief")
        self.eau_dir = os.path.join(raw_dir, "hydrologie_eau")
        self.sols_dir = os.path.join(raw_dir, "pedologie_sols")
        self.geologie_dir = os.path.join(raw_dir, "geologie")
        self.ecologie_dir = os.path.join(raw_dir, "ecologie_biomasse")
        self.wc_dir = os.path.join(raw_dir, "WorldClim")
        self.cleaned_dir = os.path.join(os.path.dirname(raw_dir), "cleaned")
        
        # Lazy load large datasets to avoid initialization overhead
        self._etopo_ds = None
        self._rivers = None
        self._lakes = None
        self._coastlines = None
        self._plates = None
        self._volcanoes = None
        self._earthquakes = None
        self._power_plants = None
        self._wood_density = None
        self._minerals_df = None
        self._minerals_tree = None
        self._fossils_df = None
        self._fossils_tree = None

    # --- GEOGRAPHIC UTILITIES ---
    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """Computes Haversine distance in kilometers between two points."""
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def _get_min_distance_to_geojson(self, lat, lon, geojson_data):
        """Finds minimum distance in km from lat/lon to any vertex in GeoJSON."""
        min_dist = float('inf')
        
        def recurse_coords(coords):
            nonlocal min_dist
            if not coords:
                return
            if isinstance(coords[0], (int, float)):
                # Coordinate point [lon, lat]
                dist = self.haversine(lat, lon, coords[1], coords[0])
                if dist < min_dist:
                    min_dist = dist
            else:
                for item in coords:
                    recurse_coords(item)

        if "features" in geojson_data:
            for feature in geojson_data["features"]:
                geom = feature.get("geometry", {})
                if geom:
                    recurse_coords(geom.get("coordinates", []))
        elif "coordinates" in geojson_data:
            recurse_coords(geojson_data["coordinates"])
            
        return min_dist if min_dist != float('inf') else None

    # --- LAZY LOADERS ---
    def get_etopo_ds(self):
        if self._etopo_ds is None:
            nc_path = os.path.join(self.relief_dir, "earth-topography-10arcmin.nc")
            if os.path.exists(nc_path):
                self._etopo_ds = xr.open_dataset(nc_path)
            else:
                raise FileNotFoundError(f"Topography file not found at: {nc_path}")
        return self._etopo_ds

    def get_rivers(self):
        if self._rivers is None:
            path = os.path.join(self.eau_dir, "rivers.geojson")
            with open(path, 'r', encoding='utf-8') as f:
                self._rivers = json.load(f)
        return self._rivers

    def get_lakes(self):
        if self._lakes is None:
            path = os.path.join(self.eau_dir, "lakes.geojson")
            with open(path, 'r', encoding='utf-8') as f:
                self._lakes = json.load(f)
        return self._lakes

    def get_coastlines(self):
        if self._coastlines is None:
            path = os.path.join(self.relief_dir, "coastlines.geojson")
            with open(path, 'r', encoding='utf-8') as f:
                self._coastlines = json.load(f)
        return self._coastlines

    def get_plates(self):
        if self._plates is None:
            path = os.path.join(self.geologie_dir, "tectonic_plates.geojson")
            with open(path, 'r', encoding='utf-8') as f:
                self._plates = json.load(f)
        return self._plates

    def get_volcanoes(self):
        if self._volcanoes is None:
            cleaned_path = os.path.join(self.cleaned_dir, "volcanoes_cleaned.csv")
            if os.path.exists(cleaned_path):
                self._volcanoes = pd.read_csv(cleaned_path)
            else:
                path = os.path.join(self.geologie_dir, "volcanoes.csv")
                self._volcanoes = pd.read_csv(path)
        return self._volcanoes

    def get_earthquakes(self):
        if self._earthquakes is None:
            cleaned_path = os.path.join(self.cleaned_dir, "earthquakes_cleaned.csv")
            if os.path.exists(cleaned_path):
                self._earthquakes = pd.read_csv(cleaned_path)
            else:
                path = os.path.join(self.geologie_dir, "earthquakes_usgs.csv")
                self._earthquakes = pd.read_csv(path)
        return self._earthquakes

    def get_power_plants(self):
        if self._power_plants is None:
            cleaned_path = os.path.join(self.cleaned_dir, "global_power_plants_cleaned.csv")
            if os.path.exists(cleaned_path):
                self._power_plants = pd.read_csv(cleaned_path)
            else:
                path = os.path.join(self.geologie_dir, "global_power_plants.csv")
                self._power_plants = pd.read_csv(path)
        return self._power_plants

    def get_wood_density(self):
        if self._wood_density is None:
            cleaned_path = os.path.join(self.cleaned_dir, "wood_density_cleaned.csv")
            if os.path.exists(cleaned_path):
                self._wood_density = pd.read_csv(cleaned_path)
            else:
                path = os.path.join(self.ecologie_dir, "wood_density.csv")
                df = pd.read_csv(path, sep=';', encoding='latin-1')
                if 'Latitude' in df.columns:
                    df['Latitude'] = df['Latitude'].astype(str).str.replace(',', '.')
                    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
                if 'Longitude' in df.columns:
                    df['Longitude'] = df['Longitude'].astype(str).str.replace(',', '.')
                    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
                self._wood_density = df
        return self._wood_density

    def get_minerals_df(self):
        if self._minerals_df is None:
            mrds_path = os.path.join(self.cleaned_dir, "mrds_cleaned.csv")
            iron_path = os.path.join(self.cleaned_dir, "iron_mines_cleaned.csv")
            
            dfs = []
            if os.path.exists(mrds_path):
                df_mrds = pd.read_csv(mrds_path)
                df_mrds = df_mrds.dropna(subset=['Latitude', 'Longitude'])
                dfs.append(df_mrds[['Nom', 'Latitude', 'Longitude', 'Commodite']])
            if os.path.exists(iron_path):
                df_iron = pd.read_csv(iron_path)
                df_iron = df_iron.dropna(subset=['Latitude', 'Longitude'])
                df_iron['Commodite'] = 'Iron Ore'
                dfs.append(df_iron[['Nom', 'Latitude', 'Longitude', 'Commodite']])
                
            if dfs:
                self._minerals_df = pd.concat(dfs, ignore_index=True)
                coords = self._minerals_df[['Latitude', 'Longitude']].values
                from scipy.spatial import cKDTree
                self._minerals_tree = cKDTree(coords)
            else:
                self._minerals_df = pd.DataFrame(columns=['Nom', 'Latitude', 'Longitude', 'Commodite'])
                self._minerals_tree = None
        return self._minerals_df

    def get_fossils_df(self):
        if self._fossils_df is None:
            coal_path = os.path.join(self.cleaned_dir, "coal_mines_cleaned.csv")
            oil_path = os.path.join(self.cleaned_dir, "oil_gas_cleaned.csv")
            
            dfs = []
            if os.path.exists(coal_path):
                df_coal = pd.read_csv(coal_path)
                df_coal = df_coal.dropna(subset=['Latitude', 'Longitude'])
                fuel_col = 'Type_Charbon' if 'Type_Charbon' in df_coal.columns else ('Coal Type' if 'Coal Type' in df_coal.columns else None)
                t_series = df_coal[fuel_col] if fuel_col else 'Coal'
                df_c = pd.DataFrame({
                    'Nom': df_coal['Nom'],
                    'Latitude': df_coal['Latitude'],
                    'Longitude': df_coal['Longitude'],
                    'Type': t_series
                })
                dfs.append(df_c)
            if os.path.exists(oil_path):
                df_oil = pd.read_csv(oil_path)
                df_oil = df_oil.dropna(subset=['Latitude', 'Longitude'])
                fuel_col = 'Type_Combustible' if 'Type_Combustible' in df_oil.columns else ('Fuel type' if 'Fuel type' in df_oil.columns else None)
                t_series = df_oil[fuel_col] if fuel_col else 'Oil/Gas'
                df_o = pd.DataFrame({
                    'Nom': df_oil['Nom'],
                    'Latitude': df_oil['Latitude'],
                    'Longitude': df_oil['Longitude'],
                    'Type': t_series
                })
                dfs.append(df_o)
                
            if dfs:
                self._fossils_df = pd.concat(dfs, ignore_index=True)
                coords = self._fossils_df[['Latitude', 'Longitude']].values
                from scipy.spatial import cKDTree
                self._fossils_tree = cKDTree(coords)
            else:
                self._fossils_df = pd.DataFrame(columns=['Nom', 'Latitude', 'Longitude', 'Type'])
                self._fossils_tree = None
        return self._fossils_df

    def get_mineral_features(self, lat, lon):
        self.get_minerals_df()
        if self._minerals_tree is None or self._minerals_df.empty:
            return {
                "dist_to_mineral_resource_km": 9999.0,
                "closest_mineral_type": "None",
                "closest_mineral_site": "None"
            }
        dist, idx = self._minerals_tree.query([lat, lon])
        row = self._minerals_df.iloc[idx]
        hav_dist = self.haversine(lat, lon, row['Latitude'], row['Longitude'])
        return {
            "dist_to_mineral_resource_km": round(hav_dist, 2),
            "closest_mineral_type": str(row['Commodite']),
            "closest_mineral_site": str(row['Nom'])
        }

    def get_fossil_features(self, lat, lon):
        self.get_fossils_df()
        if self._fossils_tree is None or self._fossils_df.empty:
            return {
                "dist_to_fossil_resource_km": 9999.0,
                "closest_fossil_type": "None",
                "closest_fossil_site": "None"
            }
        dist, idx = self._fossils_tree.query([lat, lon])
        row = self._fossils_df.iloc[idx]
        hav_dist = self.haversine(lat, lon, row['Latitude'], row['Longitude'])
        return {
            "dist_to_fossil_resource_km": round(hav_dist, 2),
            "closest_fossil_type": str(row['Type']),
            "closest_fossil_site": str(row['Nom'])
        }

    def get_tide_amplitude(self, lat, lon):
        dist_coast = self.get_hydrology_features(lat, lon)["dist_to_coast_km"]
        if dist_coast is None:
            return 0.0
        if dist_coast > 100.0:
            return 0.0
        
        # Predict equilibrium tide over 24h cycle
        t = np.linspace(0.0, 1.0, 24)
        ds = xr.Dataset(coords={'y': [lat], 'x': [lon]})
        try:
            import pyTMD.predict
            tide_series = pyTMD.predict.equilibrium_tide(t, ds)
            eq_range = float(tide_series.values.max() - tide_series.values.min())
            decay = math.exp(-dist_coast / 20.0) # decay scale 20km
            tide_amplitude = eq_range * 1000.0 * decay # Scale to meters
            return round(tide_amplitude, 3)
        except Exception:
            decay = math.exp(-dist_coast / 20.0)
            return round(1.5 * decay, 3)

    def get_vector_suitability(self, temp_mean, precip_mean):
        if temp_mean is None or precip_mean is None:
            return 0.0
        s_t = math.exp(-((temp_mean - 26.0) ** 2) / (2 * (5.0 ** 2)))
        s_p = 1.0 - math.exp(-precip_mean / 800.0)
        vsi = 10.0 * s_t * s_p
        return round(vsi, 2)

    def get_photoperiod(self, lat, doy):
        phi = math.radians(lat)
        delta = 0.409 * math.sin(2.0 * math.pi * (doy - 80.0) / 365.0)
        val = -math.tan(phi) * math.tan(delta)
        if val >= 1.0:
            return 0.0
        elif val <= -1.0:
            return 24.0
        else:
            h_s = math.acos(val)
            return (24.0 / math.pi) * h_s

    # --- CLIMATE (WORLDCLIM) READER ---
    def _read_wc_tif(self, folder, pattern, lat, lon, is_monthly=False, month=None):
        """Reads a WorldClim TIFF file dynamically at a given coordinate."""
        folder_path = os.path.join(self.wc_dir, folder)
        if not os.path.exists(folder_path):
            return None
            
        filename = None
        if is_monthly and month is not None:
            # Build filename like wc2.1_10m_prec_01.tif or wc2.1_5m_prec_01.tif
            # Search for any file in the folder matching the pattern and month
            month_str = f"{month:02d}"
            for f in os.listdir(folder_path):
                if f.endswith('.tif') and pattern in f and month_str in f:
                    filename = f
                    break
        else:
            for f in os.listdir(folder_path):
                if f.endswith('.tif') and pattern in f:
                    filename = f
                    break
                    
        if not filename:
            return None
            
        tif_path = os.path.join(folder_path, filename)
        
        try:
            with Image.open(tif_path) as img:
                width, height = img.size
                
                # Coordinate mapping (Standard Cylindrical Projection EPSG:4326)
                col = int((lon + 180.0) / 360.0 * width)
                row = int((90.0 - lat) / 180.0 * height)
                
                # Boundary clipping
                col = max(0, min(col, width - 1))
                row = max(0, min(row, height - 1))
                
                val = img.getpixel((col, row))
                # Check for standard NoData values in WorldClim float32 (-3.4e38)
                if val is None or val < -10000 or val > 1e10 or math.isnan(val):
                    return None
                return float(val)
        except Exception:
            return None

    # --- CATEGORY GETTERS ---

    def get_climate_features(self, lat, lon):
        """Reads climate variables from WorldClim."""
        # BIO1 = Annual Mean Temperature
        temp_mean = self._read_wc_tif("biovar", "bio_1.tif", lat, lon)
        if temp_mean is None:
            # Try without underscore just in case (e.g. bio1.tif)
            temp_mean = self._read_wc_tif("biovar", "bio1.tif", lat, lon)
            
        # BIO5 = Max Temp of Warmest Month
        temp_max = self._read_wc_tif("biovar", "bio_5.tif", lat, lon)
        # BIO6 = Min Temp of Coldest Month
        temp_min = self._read_wc_tif("biovar", "bio_6.tif", lat, lon)
        # BIO12 = Annual Precipitation
        precip_mean = self._read_wc_tif("biovar", "bio_12.tif", lat, lon)
        # BIO15 = Precipitation Seasonality (CV)
        precip_seasonality = self._read_wc_tif("biovar", "bio_15.tif", lat, lon)
        
        # Calculate averages for Wind, Solar Radiation, Vapor Pressure
        wind_speeds = [self._read_wc_tif("wind", "wind", lat, lon, is_monthly=True, month=m) for m in range(1, 13)]
        wind_speeds = [w for w in wind_speeds if w is not None]
        wind_mean = sum(wind_speeds) / len(wind_speeds) if wind_speeds else None
        
        sol_rads = [self._read_wc_tif("solrad", "srad", lat, lon, is_monthly=True, month=m) for m in range(1, 13)]
        sol_rads = [s for s in sol_rads if s is not None]
        solar_radiation_mean = sum(sol_rads) / len(sol_rads) if sol_rads else None
        
        vapr_pressures = [self._read_wc_tif("vapr", "vapr", lat, lon, is_monthly=True, month=m) for m in range(1, 13)]
        vapr_pressures = [v for v in vapr_pressures if v is not None]
        vapor_pressure_mean = sum(vapr_pressures) / len(vapr_pressures) if vapr_pressures else None

        return {
            "temp_mean": temp_mean,
            "temp_max": temp_max,
            "temp_min": temp_min,
            "precip_mean": precip_mean,
            "precip_seasonality": precip_seasonality,
            "wind_speed_mean": wind_mean,
            "solar_radiation_mean": solar_radiation_mean,
            "vapor_pressure_mean": vapor_pressure_mean
        }

    def get_relief_features(self, lat, lon):
        """Reads topography, and computes slope, aspect, and roughness."""
        ds = self.get_etopo_ds()
        
        # Find index coordinates
        lat_arr = ds['latitude'].values
        lon_arr = ds['longitude'].values
        
        lat_idx = np.abs(lat_arr - lat).argmin()
        lon_idx = np.abs(lon_arr - lon).argmin()
        
        # Read a 3x3 grid around the coordinate to compute slope, aspect, roughness
        lat_slice = slice(max(0, lat_idx - 1), min(len(lat_arr), lat_idx + 2))
        lon_slice = slice(max(0, lon_arr - lon_idx - 2 if lon_idx + 2 > len(lon_arr) else lon_idx - 1), 
                          min(len(lon_arr), lon_idx + 2))
        
        grid = ds['topography'].isel(latitude=lat_slice, longitude=lon_slice).values
        elevation = float(ds['topography'].isel(latitude=lat_idx, longitude=lon_idx).values.item())
        
        # Calculate slope, aspect, and roughness if we have a full 3x3 window
        if grid.shape == (3, 3):
            # E = [[e00, e01, e02], [e10, e11, e12], [e20, e21, e22]]
            # Latitude is rows (lat increases downwards/upwards depending on netcdf coord arrangement)
            # Standard arrangement: latitude -90 to 90 (increasing index means increasing lat)
            # So row 0 is lower lat, row 2 is higher lat
            
            # Grid cell sizes in meters
            dy = 18553.0 # 10 minutes of latitude in meters
            dx = 18553.0 * math.cos(math.radians(lat))
            if dx < 100.0:
                dx = 100.0 # prevent division by zero near poles
                
            dz_dx = ((grid[0, 2] + 2*grid[1, 2] + grid[2, 2]) - (grid[0, 0] + 2*grid[1, 0] + grid[2, 0])) / (8.0 * dx)
            dz_dy = ((grid[2, 0] + 2*grid[2, 1] + grid[2, 2]) - (grid[0, 0] + 2*grid[0, 1] + grid[0, 2])) / (8.0 * dy)
            
            slope = math.sqrt(dz_dx**2 + dz_dy**2) * 100.0
            roughness = float(grid.max() - grid.min())
            
            # Aspect (orientation in degrees)
            if dz_dx != 0 or dz_dy != 0:
                aspect = math.degrees(math.atan2(dz_dy, -dz_dx))
                aspect = (270.0 - aspect) % 360.0
            else:
                aspect = 0.0 # flat land
        else:
            slope = 0.0
            aspect = 0.0
            roughness = 0.0
            
        return {
            "elevation": elevation,
            "slope_pct": slope,
            "aspect_deg": aspect,
            "roughness_m": roughness
        }

    def get_hydrology_features(self, lat, lon):
        """Computes distances to rivers, lakes, coastlines."""
        dist_river = self._get_min_distance_to_geojson(lat, lon, self.get_rivers())
        dist_lake = self._get_min_distance_to_geojson(lat, lon, self.get_lakes())
        dist_coast = self._get_min_distance_to_geojson(lat, lon, self.get_coastlines())
        
        # Distance to freshwater is the minimum of river and lake distance
        dist_freshwater = min(filter(None, [dist_river, dist_lake]), default=9999.0)
        
        # Simple local estimate for groundwater depth based on distance to freshwater and altitude
        # Typically, the higher the elevation and the further from rivers, the deeper the water table
        elevation = self.get_relief_features(lat, lon)["elevation"]
        # Groundwater depth approximation: close to rivers/lakes, it's near surface (1-5m). Far away, it scales with elevation.
        if elevation < 0:
            groundwater_depth = 0.0 # sea or depression
        else:
            groundwater_depth = 2.0 + (dist_freshwater * 0.5) + (elevation * 0.02)
            # Cap at 250m for realistic groundwater tables
            groundwater_depth = min(250.0, groundwater_depth)
            
        return {
            "dist_to_river_km": dist_river,
            "dist_to_lake_km": dist_lake,
            "dist_to_coast_km": dist_coast,
            "dist_to_freshwater_km": dist_freshwater,
            "groundwater_depth_m": groundwater_depth
        }

    def get_soil_features(self, lat, lon):
        """Queries ISRIC SoilGrids REST API with local cache/fallback."""
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=clay&property=sand&property=silt&property=phh2o&property=soc&depth=0-5cm&value=mean"
        
        fallback = {
            "clay_pct": 25.0,
            "silt_pct": 35.0,
            "sand_pct": 40.0,
            "soil_pH": 6.5,
            "organic_carbon_pct": 1.5
        }
        
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                properties = data.get("properties", {})
                
                # Parse variables (SoilGrids returns values in scale factors, e.g. pH is pH*10, Clay is g/kg)
                def get_mean_val(prop_name):
                    layers = properties.get(prop_name, {}).get("depths", [])
                    if layers:
                        return layers[0].get("values", {}).get("mean", None)
                    return None
                
                clay = get_mean_val("clay")   # g/kg -> %
                silt = get_mean_val("silt")   # g/kg -> %
                sand = get_mean_val("sand")   # g/kg -> %
                ph = get_mean_val("phh2o")    # pH*10 -> pH
                soc = get_mean_val("soc")     # dg/kg -> % (1 dg/kg = 0.1 g/kg = 0.01%)
                
                res = {}
                res["clay_pct"] = clay / 10.0 if clay is not None else fallback["clay_pct"]
                res["silt_pct"] = silt / 10.0 if silt is not None else fallback["silt_pct"]
                res["sand_pct"] = sand / 10.0 if sand is not None else fallback["sand_pct"]
                res["soil_pH"] = ph / 10.0 if ph is not None else fallback["soil_pH"]
                res["organic_carbon_pct"] = soc / 100.0 if soc is not None else fallback["organic_carbon_pct"]
                return res
            else:
                return fallback
        except Exception:
            return fallback

    def get_geology_features(self, lat, lon):
        """Computes distances to tectonic faults, volcanoes, and power plants."""
        # Dist to fault
        dist_fault = self._get_min_distance_to_geojson(lat, lon, self.get_plates())
        
        # Volcanic risk based on closest active volcano
        volc_df = self.get_volcanoes()
        min_dist_volc = float('inf')
        closest_volc_type = "None"
        for _, row in volc_df.iterrows():
            dist = self.haversine(lat, lon, row["latitude"], row["longitude"])
            if dist < min_dist_volc:
                min_dist_volc = dist
                closest_volc_type = row["primary_volcano_type"]
                
        # Volcanic risk index: 0 (safe) to 10 (extremely dangerous)
        if min_dist_volc < 15.0:
            volcanic_risk = 10.0
        elif min_dist_volc < 50.0:
            volcanic_risk = 6.0
        elif min_dist_volc < 150.0:
            volcanic_risk = 2.0
        else:
            volcanic_risk = 0.0
            
        # Seismic risk based on M5.0+ earthquakes since 2015 within 300km
        eq_df = self.get_earthquakes()
        eq_count = 0
        for _, row in eq_df.iterrows():
            dist = self.haversine(lat, lon, row["latitude"], row["longitude"])
            if dist <= 300.0:
                eq_count += 1
                
        # Seismic risk index scaled 0 to 10
        seismic_risk = min(10.0, eq_count / 15.0 * 10.0)
        
        # Energy and metal mines (via nearest power plant coordinates)
        pp_df = self.get_power_plants()
        min_dist_pp = float('inf')
        closest_fuel = "None"
        for _, row in pp_df.iterrows():
            dist = self.haversine(lat, lon, row["latitude"], row["longitude"])
            if dist < min_dist_pp:
                min_dist_pp = dist
                closest_fuel = row["primary_fuel"]
                
        return {
            "dist_to_fault_km": dist_fault,
            "seismic_risk_index": round(seismic_risk, 2),
            "dist_to_volcano_km": round(min_dist_volc, 2),
            "volcanic_risk_index": volcanic_risk,
            "closest_volcano_type": closest_volc_type,
            "dist_to_energy_source_km": round(min_dist_pp, 2),
            "closest_energy_type": closest_fuel
        }

    def get_ecology_features(self, lat, lon, temp_mean=None, precip_mean=None):
        """Computes NPP (Miami Model), Wood Density, and Faunal Densities (Hatton scaling)."""
        # Read climate values if not provided
        if temp_mean is None or precip_mean is None:
            clim = self.get_climate_features(lat, lon)
            temp_mean = clim["temp_mean"] if temp_mean is None else temp_mean
            precip_mean = clim["precip_mean"] if precip_mean is None else precip_mean
            
        # If climate data is missing, use global averages (temp=15C, precip=800mm)
        T = temp_mean if temp_mean is not None else 15.0
        P = precip_mean if precip_mean is not None else 800.0
        
        # Miami Model NPP (g/m2/year)
        npp_t = 3000.0 / (1.0 + math.exp(1.315 - 0.119 * T))
        npp_p = 3000.0 * (1.0 - math.exp(-0.000664 * P))
        npp = min(npp_t, npp_p)
        
        # Wood Density (nearest measured wood density species in the region)
        wd_df = self.get_wood_density()
        # Find nearest point in wood density database
        min_dist = float('inf')
        local_wood_density = 0.55 # global average fallback
        
        # Simple lookup of closest coordinate with wood density data
        for _, row in wd_df.head(1000).iterrows(): # inspect first 1000 rows for performance
            if not math.isnan(row["Latitude"]) and not math.isnan(row["Longitude"]):
                dist = self.haversine(lat, lon, row["Latitude"], row["Longitude"])
                if dist < min_dist:
                    min_dist = dist
                    # Parse density value which might have ',' instead of '.'
                    val_str = str(row["Wood Density"]).replace(',', '.')
                    try:
                        local_wood_density = float(val_str)
                    except ValueError:
                        pass
                        
        # Faunal Biomass Densities (Hatton et al. 2015 scaling)
        # Herbivore Biomass: B_h = 0.05 * NPP (kg/km^2)
        # Predator Biomass: B_p = 0.02 * B_h (2% of herbivore biomass)
        herbivore_biomass = 0.05 * npp
        predator_biomass = 0.02 * herbivore_biomass
        
        return {
            "npp_g_m2_yr": round(npp, 2),
            "estimated_wood_density_g_cm3": round(local_wood_density, 3),
            "fauna_herbivore_biomass_kg_km2": round(herbivore_biomass, 2),
            "fauna_predator_biomass_kg_km2": round(predator_biomass, 2)
        }

    # --- MAIN ENGINE FUNCTION ---
    def get_physical_features(self, lat, lon):
        """Runs the entire planetary simulation Layer 1 for a single coordinate."""
        try:
            relief = self.get_relief_features(lat, lon)
        except Exception:
            relief = {"elevation": 0.0, "slope_pct": 0.0, "aspect_deg": 0.0, "roughness_m": 0.0}
            
        clim = self.get_climate_features(lat, lon)
        hydrology = self.get_hydrology_features(lat, lon)
        sols = self.get_soil_features(lat, lon)
        geology = self.get_geology_features(lat, lon)
        ecology = self.get_ecology_features(lat, lon, temp_mean=clim["temp_mean"], precip_mean=clim["precip_mean"])
        
        # New Layer 1 features
        minerals = self.get_mineral_features(lat, lon)
        fossils = self.get_fossil_features(lat, lon)
        tide_amplitude = self.get_tide_amplitude(lat, lon)
        
        vector_suitability = self.get_vector_suitability(clim["temp_mean"], clim["precip_mean"])
        
        photo_summer = self.get_photoperiod(lat, 172)
        photo_winter = self.get_photoperiod(lat, 355)
        photo_range = abs(photo_summer - photo_winter)
        astronomy = {
            "photoperiod_summer_solstice_hours": round(photo_summer, 2),
            "photoperiod_winter_solstice_hours": round(photo_winter, 2),
            "photoperiod_range_hours": round(photo_range, 2)
        }
        
        # Combine everything
        features = {
            "latitude": lat,
            "longitude": lon,
            **clim,
            **relief,
            **hydrology,
            **sols,
            **geology,
            **ecology,
            **minerals,
            **fossils,
            "tide_amplitude_m": tide_amplitude,
            "vector_suitability_index": vector_suitability,
            **astronomy
        }
        return features
