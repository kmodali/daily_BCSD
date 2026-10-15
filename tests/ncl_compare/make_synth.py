"""Synthetic GCM + obs files for comparing BCDaily.ncl and the python port.

usage: python make_synth.py <outdir> [calendar]

Writes <outdir>/{tas,pr}/{hist,rcp,obs}: an 8x10 grid, GCM decades 1950-2029
in <calendar> (default noleap), yearly obs files 1970-1989 in the standard
calendar with two masked points.
"""
import sys
from pathlib import Path

import cftime
import numpy as np
import xarray as xr

root = Path(sys.argv[1])
cal = sys.argv[2] if len(sys.argv) > 2 else "noleap"
rng = np.random.default_rng(0)
lat = np.arange(-10.0, 30.0, 5.0)  # 8
lon = np.arange(100.0, 150.0, 5.0)  # 10
latc = xr.DataArray(lat, dims="lat", attrs={"units": "degrees_north", "standard_name": "latitude"})
lonc = xr.DataArray(lon, dims="lon", attrs={"units": "degrees_east", "standard_name": "longitude"})
ny, nx = lat.size, lon.size


def field(var, times, obs):
    doy = np.array([t.dayofyr for t in times])
    yr = np.array([t.year + (t.dayofyr - 1) / 365.0 for t in times])
    shape = (len(times), ny, nx)
    if var == "tas":
        base = 290 + 10 * np.cos(2 * np.pi * (doy - 200) / 365.0)[:, None, None] + lat[None, :, None] * -0.2
        warm = 0.0 if obs else 0.06 * np.clip(yr - 1980, 0, None)[:, None, None]
        sd = 2.5 if obs else 3.5
        x = base + warm + (1.5 if obs else -1.0) + sd * rng.standard_normal(shape)
    else:
        wet = rng.random(shape) < (0.45 if obs else 0.6)
        scale = (6.0 if obs else 4.0) * (1 + 0.5 * np.sin(2 * np.pi * doy / 365.0))[:, None, None]
        if not obs:
            scale = scale * (1 + 0.02 * np.clip(yr - 1990, 0, None))[:, None, None]
        x = np.where(wet, rng.gamma(0.7, 1.0, shape) * scale, 0.0)
    return x.astype(np.float32)


def write(path, var, times, data, units, calendar):
    da = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"lat": latc, "lon": lonc},
                      attrs={"units": "K" if var == "tas" else "kg m-2 s-1"}, name=var)
    num = cftime.date2num(times, units, calendar)
    ds = da.to_dataset().assign_coords(time=("time", num, {"units": units, "calendar": calendar,
                                                           "standard_name": "time"}))
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding={var: {"_FillValue": np.float32(1e20)}, "time": {"_FillValue": None},
                                 "lat": {"_FillValue": None}, "lon": {"_FillValue": None}},
                 unlimited_dims=["time"], format="NETCDF4_CLASSIC")


def days(y0, y1, calendar):
    t, out = cftime.datetime(y0, 1, 1, calendar=calendar), []
    step = cftime.datetime(1950, 1, 2, calendar=calendar) - cftime.datetime(1950, 1, 1, calendar=calendar)
    while t.year <= y1:
        out.append(t)
        t = t + step
    return out


for var in ("tas", "pr"):
    for d0 in range(1950, 2030, 10):
        sub = "hist" if d0 < 2000 else "rcp"
        times = days(d0, d0 + 9, cal)
        write(root / var / sub / f"{var}_day_GCM_{sub}_{d0}-{d0 + 9}.nc", var, times,
              field(var, times, obs=False), "days since 1950-01-01", cal)
    for y in range(1970, 1990):
        times = days(y, y, "standard")
        x = field(var, times, obs=True)
        x[:, 0, 0] = np.nan  # an "ocean" point
        x[:, 3, 7] = np.nan
        write(root / var / "obs" / f"{var}_{y}.nc", var, times, x, "days since 1970-01-01", "standard")
print("done", root)
