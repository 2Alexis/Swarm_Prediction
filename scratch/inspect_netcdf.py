import xarray as xr

nc_path = r"c:\Users\alexi\Desktop\Swarm_Prediction\data\raw\topographie_relief\earth-topography-10arcmin.nc"

try:
    ds = xr.open_dataset(nc_path)
    print("Dataset structure:")
    print(ds)
    print("\nDimensions:")
    print(ds.dims)
    print("\nVariables:")
    print(list(ds.variables))
except Exception as e:
    print("Error reading NetCDF:", str(e))
