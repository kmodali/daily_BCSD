"""Compare an NCL and a python output file.

usage: python compare.py <ncl.nc> <py.nc> <var>
"""
import sys

import numpy as np
import xarray as xr

ncl, py, var = sys.argv[1:4]
a = xr.open_dataset(ncl, decode_times=False)
b = xr.open_dataset(py, decode_times=False)
x, y = a[var].values.astype(np.float64), b[var].values.astype(np.float64)
print(f"shape ncl {x.shape} py {y.shape}")
print("time equal:", np.array_equal(a.time.values, b.time.values), "| units:", a.time.attrs.get("units"), "/", b.time.attrs.get("units"),
      "| cal:", a.time.attrs.get("calendar"), "/", b.time.attrs.get("calendar"))
print("lat/lon equal:", np.array_equal(a.lat.values, b.lat.values), np.array_equal(a.lon.values, b.lon.values))
nx, ny = np.isnan(x), np.isnan(y)
print("missing mask identical:", np.array_equal(nx, ny), f"(ncl {nx.sum()}, py {ny.sum()})")
ok = ~nx & ~ny
d = np.abs(x[ok] - y[ok])
rel = d / np.maximum(np.abs(x[ok]), 1e-6)
print(f"abs diff: max {d.max():.3e}  mean {d.mean():.3e}  | rel diff max {rel.max():.3e}")
for tol in (1e-5, 1e-4, 1e-3, 1e-2):
    print(f"  frac |diff| > {tol:g}: {(d > tol).mean():.2e}")
if var == "pr":
    print("zeros: ncl", (x[ok] == 0).mean().round(4), "py", (y[ok] == 0).mean().round(4))
i = np.argmax(np.where(ok, np.abs(x - y), -1))
t, j, k = np.unravel_index(i, x.shape)
print(f"worst at t={t} lat={j} lon={k}: ncl {x[t, j, k]:.6g} py {y[t, j, k]:.6g}")
