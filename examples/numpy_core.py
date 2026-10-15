"""Use the bias-correction core directly on numpy arrays (no files, no dask).

usage: uv run python examples/numpy_core.py

Arrays are shaped (points, time); doy_* are 0-based days of year.  This does
NOT remove the tas trend -- only bc_daily() does that.
"""

import numpy as np

from daily_bcsd.bc_daily import bias_correct_block

rng = np.random.default_rng(0)
doy = lambda n: np.arange(n) % 365  # noleap calendar

# Temperature-like: GCM too warm and too variable compared to obs
hist = rng.normal(285, 4, (4, 20 * 365)).astype(np.float32)  # GCM, reference period
obs = rng.normal(288, 3, (4, 20 * 365)).astype(np.float32)   # obs, reference period
fut = rng.normal(287, 4, (4, 10 * 365)).astype(np.float32)   # GCM decade to correct

bc = bias_correct_block(fut, hist, obs, doy_gcm=doy(fut.shape[-1]), doy_hist=doy(hist.shape[-1]),
                        doy_obs=doy(obs.shape[-1]), ndays=365, buff=15, prcp=False)
print(f"tas  std  GCM {fut.std():.2f} -> BC {bc.std():.2f}   (obs {obs.std():.2f})")
print(f"tas  mean GCM {fut.mean():.2f} -> BC {bc.mean():.2f}  (obs {obs.mean():.2f}, GCM warms by 2 K)")

# Precipitation-like: GCM drizzles too often
def precip(wet, scale, shape):
    return np.where(rng.random(shape) < wet, rng.gamma(0.7, scale, shape), 0.0).astype(np.float32)

hist = precip(0.7, 3.0, (4, 20 * 365))
obs = precip(0.4, 6.0, (4, 20 * 365))
fut = precip(0.7, 3.5, (4, 10 * 365))
bc = bias_correct_block(fut, hist, obs, doy_gcm=doy(fut.shape[-1]), doy_hist=doy(hist.shape[-1]),
                        doy_obs=doy(obs.shape[-1]), ndays=365, buff=15, prcp=True)
print(f"pr   dry fraction GCM {(fut == 0).mean():.2f} -> BC {(bc == 0).mean():.2f}   (obs {(obs == 0).mean():.2f})")
print(f"pr   mean         GCM {fut.mean():.2f} -> BC {bc.mean():.2f}   (obs {obs.mean():.2f})")

# A masked point: missing in the first obs time step -> stays missing
obs[0, 0] = np.nan
bc = bias_correct_block(fut, hist, obs, doy_gcm=doy(fut.shape[-1]), doy_hist=doy(hist.shape[-1]),
                        doy_obs=doy(obs.shape[-1]), ndays=365, buff=15, prcp=True)
print("masked point all NaN:", bool(np.isnan(bc[0]).all()))
