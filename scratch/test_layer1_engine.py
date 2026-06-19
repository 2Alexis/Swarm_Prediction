import sys
import os
import pprint

# Add project root to sys.path to load layer1_engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from layer1_engine import Layer1Engine

def test_engine():
    print("=" * 80)
    print("RUNNING LAYER 1 PHYSICAL SIMULATION ENGINE TEST")
    print("=" * 80)
    
    engine = Layer1Engine()
    
    test_locations = [
        {"name": "Paris (Temperate Europe)", "lat": 48.8566, "lon": 2.3522},
        {"name": "Manaus (Equatorial Amazon)", "lat": -3.1190, "lon": -60.0217},
        {"name": "Cairo (Arid Egypt)", "lat": 30.0444, "lon": 31.2357}
    ]
    
    for loc in test_locations:
        print("\n" + "#" * 60)
        print(f"Testing location: {loc['name']}")
        print(f"Coordinates: Lat={loc['lat']}, Lon={loc['lon']}")
        print("#" * 60)
        
        try:
            features = engine.get_physical_features(loc['lat'], loc['lon'])
            
            # Print categories cleanly
            print("\n[1. Atmosphère et Climat]")
            print(f"  Annual Mean Temp: {features['temp_mean']:.2f} °C" if features['temp_mean'] is not None else "  Annual Mean Temp: None")
            print(f"  Max Temp (Warmest Month): {features['temp_max']:.2f} °C" if features['temp_max'] is not None else "  Max Temp: None")
            print(f"  Min Temp (Coldest Month): {features['temp_min']:.2f} °C" if features['temp_min'] is not None else "  Min Temp: None")
            print(f"  Annual Precipitation: {features['precip_mean']:.1f} mm" if features['precip_mean'] is not None else "  Precipitation: None")
            print(f"  Precipitation Seasonality (CV): {features['precip_seasonality']:.1f}" if features['precip_seasonality'] is not None else "  Precip Seasonality: None")
            print(f"  Wind Speed (Mean): {features['wind_speed_mean']:.2f} m/s" if features['wind_speed_mean'] is not None else "  Wind Speed: None")
            print(f"  Solar Radiation (Mean): {features['solar_radiation_mean']:.2f} kJ/m2/day" if features['solar_radiation_mean'] is not None else "  Solar Radiation: None")
            print(f"  Vapor Pressure (Mean): {features['vapor_pressure_mean']:.2f} kPa" if features['vapor_pressure_mean'] is not None else "  Vapor Pressure: None")
            
            print("\n[2. Topographie et Relief]")
            print(f"  Elevation (Altitude): {features['elevation']:.1f} m")
            print(f"  Slope (Pente): {features['slope_pct']:.2f} %")
            print(f"  Aspect (Orientation): {features['aspect_deg']:.1f} ° (clockwise from North)")
            print(f"  Roughness (Rugosité): {features['roughness_m']:.1f} m")
            
            print("\n[3. Hydrologie]")
            print(f"  Distance to nearest River: {features['dist_to_river_km']:.2f} km" if features['dist_to_river_km'] is not None else "  Distance to River: None")
            print(f"  Distance to nearest Lake: {features['dist_to_lake_km']:.2f} km" if features['dist_to_lake_km'] is not None else "  Distance to Lake: None")
            print(f"  Distance to Coastline: {features['dist_to_coast_km']:.2f} km" if features['dist_to_coast_km'] is not None else "  Distance to Coastline: None")
            print(f"  Distance to Freshwater: {features['dist_to_freshwater_km']:.2f} km")
            print(f"  Estimated Groundwater Depth: {features['groundwater_depth_m']:.2f} m")
            
            print("\n[4. Pédologie (Sols)]")
            print(f"  Clay content: {features['clay_pct']:.1f} %")
            print(f"  Silt content: {features['silt_pct']:.1f} %")
            print(f"  Sand content: {features['sand_pct']:.1f} %")
            print(f"  Soil pH: {features['soil_pH']:.2f}")
            print(f"  Organic Carbon content: {features['organic_carbon_pct']:.2f} %")
            
            print("\n[5. Géologie et Risques]")
            print(f"  Distance to nearest tectonic fault: {features['dist_to_fault_km']:.2f} km" if features['dist_to_fault_km'] is not None else "  Distance to fault: None")
            print(f"  Seismic Risk Index (0-10): {features['seismic_risk_index']}")
            print(f"  Distance to nearest active volcano: {features['dist_to_volcano_km']} km")
            print(f"  Volcanic Risk Index (0-10): {features['volcanic_risk_index']} (Type: {features['closest_volcano_type']})")
            print(f"  Distance to nearest major energy source: {features['dist_to_energy_source_km']} km (Type: {features['closest_energy_type']})")
            
            print("\n[6. Écologie Native]")
            print(f"  Net Primary Productivity (NPP - Miami Model): {features['npp_g_m2_yr']:.2f} g/m2/year")
            print(f"  Estimated Wood Density: {features['estimated_wood_density_g_cm3']:.3f} g/cm3")
            print(f"  Fauna Large Herbivore Biomass: {features['fauna_herbivore_biomass_kg_km2']:.2f} kg/km2")
            print(f"  Fauna Large Predator Biomass: {features['fauna_predator_biomass_kg_km2']:.2f} kg/km2")
            
            print("\n[7. Geological Resources (New)]")
            print(f"  Distance to nearest mineral resource: {features['dist_to_mineral_resource_km']:.2f} km (Type: {features['closest_mineral_type']}, Site: {features['closest_mineral_site']})")
            print(f"  Distance to nearest fossil resource: {features['dist_to_fossil_resource_km']:.2f} km (Type: {features['closest_fossil_type']}, Site: {features['closest_fossil_site']})")
            
            print("\n[8. Tides and Vectors (New)]")
            print(f"  Ocean Tide Amplitude: {features['tide_amplitude_m']:.3f} m")
            print(f"  Disease Vector Suitability Index (0-10): {features['vector_suitability_index']:.2f}")
            
            print("\n[9. Astronomy & Daylength (New)]")
            print(f"  Summer Solstice Daylength: {features['photoperiod_summer_solstice_hours']:.2f} hours")
            print(f"  Winter Solstice Daylength: {features['photoperiod_winter_solstice_hours']:.2f} hours")
            print(f"  Daylength Seasonal Range: {features['photoperiod_range_hours']:.2f} hours")
            
        except Exception as e:
            print(f"  FAIL: An error occurred testing this location: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_engine()
