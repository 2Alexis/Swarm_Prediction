import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from layer1_engine import Layer1Engine

# Set plotting style for premium aesthetics
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Create figures directory
figures_dir = r"c:\Users\alexi\Desktop\Swarm_Prediction\reports\figures"
os.makedirs(figures_dir, exist_ok=True)

def generate_grid_data(num_points=1000):
    print("Initializing Layer 1 Engine...")
    engine = Layer1Engine()
    
    # 1. Mock SoilGrids API to run offline and lightning fast
    engine.get_soil_features = lambda lat, lon: {
        "clay_pct": 25.0,
        "silt_pct": 35.0,
        "sand_pct": 40.0,
        "soil_pH": 6.5,
        "organic_carbon_pct": 1.5
    }
    
    # 2. Overwrite get_climate_features to skip reading monthly files (srad, wind, vapr)
    # which cuts out 36 file openings per coordinate!
    def fast_climate(lat, lon):
        temp_mean = engine._read_wc_tif("biovar", "bio_1.tif", lat, lon)
        if temp_mean is None:
            temp_mean = engine._read_wc_tif("biovar", "bio1.tif", lat, lon)
        temp_max = engine._read_wc_tif("biovar", "bio_5.tif", lat, lon)
        temp_min = engine._read_wc_tif("biovar", "bio_6.tif", lat, lon)
        precip_mean = engine._read_wc_tif("biovar", "bio_12.tif", lat, lon)
        precip_seasonality = engine._read_wc_tif("biovar", "bio_15.tif", lat, lon)
        return {
            "temp_mean": temp_mean,
            "temp_max": temp_max,
            "temp_min": temp_min,
            "precip_mean": precip_mean,
            "precip_seasonality": precip_seasonality,
            "wind_speed_mean": 3.0,
            "solar_radiation_mean": 15000.0,
            "vapor_pressure_mean": 1.2
        }
    engine.get_climate_features = fast_climate
    
    print("Generating global grid of land coordinates...")
    # Generate points across major landmasses
    lats = np.random.uniform(-50, 70, num_points * 2)
    lons = np.random.uniform(-120, 140, num_points * 2)
    
    records = []
    points_collected = 0
    
    for lat, lon in zip(lats, lons):
        if points_collected >= num_points:
            break
        try:
            features = engine.get_physical_features(lat, lon)
            # Filter out ocean cells (elevation <= 0 or no climate data)
            if features.get("elevation", 0) > 5 and features.get("temp_mean") is not None:
                records.append(features)
                points_collected += 1
                if points_collected % 100 == 0:
                    print(f"  Collected {points_collected}/{num_points} land points...")
        except Exception:
            continue
            
    df = pd.DataFrame(records)
    print(f"Generated DataFrame with {df.shape[0]} points and {df.shape[1]} features.")
    return df

def create_plots(df):
    print("\nGenerating diagnostic plots...")
    
    # ----------------------------------------------------
    # Plot 1: Climat & Ecologie (Whittaker Biome Space)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        df["temp_mean"], 
        df["precip_mean"], 
        c=df["npp_g_m2_yr"], 
        cmap="viridis", 
        alpha=0.7, 
        s=df["fauna_herbivore_biomass_kg_km2"] * 1.5,
        edgecolors='none'
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Net Primary Productivity (NPP - g/m²/year)", fontsize=12)
    plt.xlabel("Annual Mean Temperature (°C)")
    plt.ylabel("Annual Precipitation (mm)")
    plt.title("Whittaker Climate Space & Natural Biomass Productivity")
    
    sizes = [10, 50, 100]
    for s in sizes:
        plt.scatter([], [], c='gray', alpha=0.5, s=s * 1.5, label=f"{s} kg/km²")
    plt.legend(title="Herbivore Biomass Density", loc="upper left")
    plt.tight_layout()
    plot1_path = os.path.join(figures_dir, "climate_ecology.png")
    plt.savefig(plot1_path, dpi=200)
    plt.close()
    print(f"  Saved: {plot1_path}")
    
    # ----------------------------------------------------
    # Plot 2: Topographie vs Hydrologie
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.regplot(
        data=df, 
        x="elevation", 
        y="groundwater_depth_m", 
        ax=axes[0],
        scatter_kws={"alpha": 0.4, "color": "teal"},
        line_kws={"color": "red"}
    )
    axes[0].set_xlabel("Elevation (m)")
    axes[0].set_ylabel("Estimated Groundwater Table Depth (m)")
    axes[0].set_title("Groundwater Table Depth vs. Elevation")
    
    sns.scatterplot(
        data=df, 
        x="slope_pct", 
        y="roughness_m", 
        ax=axes[1],
        alpha=0.5,
        color="purple"
    )
    axes[1].set_xlabel("Slope (%)")
    axes[1].set_ylabel("Roughness (m)")
    axes[1].set_title("Terrain Roughness vs. Slope")
    
    plt.suptitle("Topographical and Hydrological Relationships", y=0.98)
    plt.tight_layout()
    plot2_path = os.path.join(figures_dir, "topography_hydrology.png")
    plt.savefig(plot2_path, dpi=200)
    plt.close()
    print(f"  Saved: {plot2_path}")
    
    # ----------------------------------------------------
    # Plot 3: Tectonique vs Séismes (Risques)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="dist_to_fault_km",
        y="seismic_risk_index",
        hue="elevation",
        palette="magma",
        alpha=0.7,
        edgecolor=None
    )
    plt.xlabel("Distance to Tectonic Plate Boundary (km)")
    plt.ylabel("Seismic Risk Index (USGS M5+ Density)")
    plt.title("Seismic Hazard vs. Proximity to Tectonic Plates")
    plt.tight_layout()
    plot3_path = os.path.join(figures_dir, "seismic_geology.png")
    plt.savefig(plot3_path, dpi=200)
    plt.close()
    print(f"  Saved: {plot3_path}")
    
    # ----------------------------------------------------
    # Plot 4: Ressources Énergétiques (Global Resource Map)
    # ----------------------------------------------------
    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=df,
        x="longitude",
        y="latitude",
        hue="closest_energy_type",
        size="dist_to_energy_source_km",
        sizes=(150, 10),
        alpha=0.8,
        palette="Set2"
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Planetary Distribution of Closest Major Energy Infrastructure")
    plt.legend(title="Closest Energy Resource", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot4_path = os.path.join(figures_dir, "global_energy_resources.png")
    plt.savefig(plot4_path, dpi=200)
    plt.close()
    print(f"  Saved: {plot4_path}")
    
    print("\nVisualizations generation complete!")

def main():
    df = generate_grid_data(1000)
    create_plots(df)

if __name__ == "__main__":
    main()
