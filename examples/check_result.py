"""Quick comparison of a GCM file, its bias-corrected version and the obs.

usage: uv run python examples/check_result.py <gcm.nc> <bc.nc> <var> [obs files ...]

Prints domain statistics.  Works lazily, so it is fine for large files.
"""

import sys

import numpy as np
import xarray as xr


def stats(name, da):
    p = da.quantile([0.01, 0.5, 0.99], dim=["time", "lat", "lon"], skipna=True).values
    line = f"{name:5s} mean {float(da.mean()):10.4g}  std {float(da.std()):10.4g}  " \
           f"p01/p50/p99 {p[0]:.4g} / {p[1]:.4g} / {p[2]:.4g}"
    if var not in ("tas", "tasmax", "tasmin"):
        line += f"  dry(<1e-6) {float((da < 1e-6).where(da.notnull()).mean()):.3f}"
    print(line)


gcm_path, bc_path, var, *obs_paths = sys.argv[1:]
chunks = {"time": -1, "lat": 32, "lon": 32}
bc = xr.open_dataset(bc_path, chunks=chunks)[var]
mask = bc.isel(time=0).notnull()  # compare only points that were corrected


def on_bc_grid(da):
    """Nearest grid point of the corrected file, masked like it."""
    da = da.sel(lat=bc.lat, lon=bc.lon, method="nearest").assign_coords(lat=bc.lat, lon=bc.lon)
    return da.where(mask)


gcm = on_bc_grid(xr.open_dataset(gcm_path, chunks=chunks)[var])

stats("GCM", gcm)
stats("BC", bc)
if obs_paths:
    obs = xr.open_mfdataset(obs_paths, combine="nested", concat_dim="time", chunks=chunks)[var]
    stats("OBS", on_bc_grid(obs))

n_missing = int(bc.isnull().all("time").sum())
print(f"points never corrected (obs mask): {n_missing} of {mask.size}")
if not np.isfinite(float(bc.max())):
    print("WARNING: non-finite values in the corrected data")
