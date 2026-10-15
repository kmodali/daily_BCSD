"""Daily bias correction (the BC step of BCSD): xarray/dask port of BCDaily.ncl.

One GCM file -- index ``dec`` in the sorted list of historical + future files --
is corrected by quantile mapping every day against the GCM and observed
distributions of a +/- ``buff`` day window around its day of year over the
reference period.  For temperature (``tas*``) the GCM's monthly trend relative
to the reference period is removed first and added back afterwards.

Everything stays lazy.  The domain is cut into lat/lon tiles that each carry
the full time series, and every tile is read, corrected and written on its
own, so peak memory is roughly ``workers x tile footprint`` regardless of the
domain size.  The tile footprint is logged before computing.
"""

import argparse
import glob
import logging
import os
from pathlib import Path

import dask
import netCDF4
import numpy as np
import xarray as xr

from .bcfuncs import DAT, PRCP, QUAN, TEMP, lookup, stat4

log = logging.getLogger(__name__)

FILL = 1.0e20
# HDF5 chunk cache per open variable.  netCDF's 64 MB default, kept by every
# open file, dominates memory with many input files; tiles read each HDF5
# chunk only once per hyperslab call, so a big cache buys nothing here.
NC_CHUNK_CACHE = 4 * 2**20
TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)


# --------------------------------------------------------------------------
# Per-tile computation (pure numpy, runs inside dask tasks)
# --------------------------------------------------------------------------

def _wrap(doys, ndays, upper):
    doys = np.where(doys < 0, doys + ndays, doys)
    return np.where(doys >= upper, doys - ndays, doys)


def _quantiles(n):
    return np.arange(1, n + 1, dtype=np.float32) / np.float32(n + 1)


def bias_correct_block(gcm, hist, obs, *, doy_gcm, doy_hist, doy_obs, ndays, buff, prcp):
    """Quantile-map one spatial tile.

    gcm (..., t), hist (..., t_hist), obs (..., t_obs): the (detrended) GCM data
    to correct, the GCM reference period and the observed reference period;
    time is the last axis.  ``doy_*`` are the matching 0-based days of year.
    Points where the first observed time step is missing are left missing.
    """
    spatial = gcm.shape[:-1]
    # float32 throughout, as in NCL (see bcfuncs)
    gcm, hist, obs = (a.reshape(-1, a.shape[-1]).astype(np.float32, copy=False) for a in (gcm, hist, obs))
    out = np.full(gcm.shape, np.nan, dtype=np.float32)

    pts = ~np.isnan(obs[:, 0])  # obsmask
    if not pts.any():
        return out.reshape(spatial + out.shape[-1:])
    gcm, hist, obs = gcm[pts], hist[pts], obs[pts]
    par = PRCP if prcp else TEMP
    res = np.empty(gcm.shape, dtype=np.float32)

    # The CDFs depend only on the day of year, so build them once per doy and
    # correct all days sharing it together (NCL redoes this for every day).
    doy_gcm = np.minimum(doy_gcm, ndays)
    for doy in np.unique(doy_gcm):
        days = doy_gcm == doy
        win = np.arange(doy - buff, doy + buff + 1)
        # NCL wraps the obs window at 365 regardless of the GCM calendar
        sort_hist = np.sort(hist[:, np.isin(doy_hist, _wrap(win, ndays, ndays))], axis=-1)
        sort_obs = np.sort(obs[:, np.isin(doy_obs, _wrap(win, ndays, 365))], axis=-1)
        quant_hist = _quantiles(sort_hist.shape[-1])
        quant_obs = _quantiles(sort_obs.shape[-1])

        mean_hist, var_hist, skew_hist = stat4(sort_hist)
        skew_hist = np.where(var_hist == 0, 0.0, skew_hist)
        mean_obs, var_obs, skew_obs = stat4(sort_obs)
        skew_obs = np.where(var_obs == 0, 0.0, skew_obs)

        q = lookup(gcm[:, days], sort_hist, quant_hist, par, QUAN,
                   mean_hist, np.sqrt(var_hist), skew_hist)
        bc = lookup(q, quant_obs, sort_obs, par, DAT, mean_obs, np.sqrt(var_obs), skew_obs)
        if prcp:
            bc = np.where(bc <= 0.0, 0.0, bc)
        res[:, days] = bc

    out[pts] = res
    return out.reshape(spatial + out.shape[-1:])


# --------------------------------------------------------------------------
# Monthly trend for temperature
# --------------------------------------------------------------------------

def monthly_trend(gcm_all, ref_st, ref_end, pad):
    """Running ``2*pad+1``-year mean of monthly anomalies vs. the reference climatology.

    Returns a lazy (yyyymm, lat, lon) array spanning whole years from the first
    to the last year of ``gcm_all``; zero up to the end of the reference period.
    """
    years = gcm_all.time.dt.year.values
    start, end = int(years[0]), int(years[-1])
    ny = end - start + 1
    yyyymm = np.array([y * 100 + m for y in range(start, end + 1) for m in range(1, 13)])

    key = (gcm_all.time.dt.year * 100 + gcm_all.time.dt.month).rename("yyyymm")
    # Averages are summed in double and stored as float, as NCL does
    f32, f64 = np.float32, np.float64
    mon = gcm_all.astype(f64).groupby(key).mean().astype(f32).reindex(yyyymm=yyyymm)
    month = xr.DataArray(yyyymm % 100, dims="yyyymm", coords={"yyyymm": yyyymm}, name="month")
    ref = mon.sel(yyyymm=slice(ref_st * 100 + 1, ref_end * 100 + 12))
    clim = ref.astype(f64).groupby(month.sel(yyyymm=ref.yyyymm)).mean().astype(f32)
    anom = (mon.groupby(month) - clim).drop_vars("month").transpose("yyyymm", "lat", "lon")

    # Per calendar month, running mean over years (shifts by whole years keep
    # the spatial tiling, unlike a dask reshape).  As in NCL (runave_n, opt=0)
    # windows touching the ends or a missing value are missing, and the last
    # `pad` years repeat the last full window.
    anom = anom.chunk({"yyyymm": -1})
    run = (sum(anom.astype(f64).shift(yyyymm=12 * k) for k in range(-pad, pad + 1)) / (2 * pad + 1)).astype(f32)
    idx = np.arange(ny * 12)
    idx[(ny - pad) * 12:] = (ny - pad - 1) * 12 + idx[(ny - pad) * 12:] % 12
    trend = run.isel(yyyymm=idx).assign_coords(yyyymm=yyyymm)
    return trend.where(trend.yyyymm > ref_end * 100 + 12, 0.0)


def _trend_at(trend, time):
    """Monthly trend repeated onto the daily ``time`` axis."""
    idx = np.searchsorted(trend.yyyymm.values, (time.dt.year * 100 + time.dt.month).values)
    return xr.DataArray(trend.data[idx], dims=("time", "lat", "lon"),
                        coords={"time": time, "lat": trend.lat, "lon": trend.lon})


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def _slice(coord, lo, hi):
    return slice(lo, hi) if coord[0] <= coord[-1] else slice(hi, lo)


def _open(path, var, box, chunks):
    """Lazily open ``var`` from one file, cut to ``box`` and tiled into ``chunks``.

    Opening without dask and chunking after the cut keeps tiles aligned to the
    box, and every tile task reads just its own hyperslab from disk.
    """
    min_lat, max_lat, min_lon, max_lon = box
    da = xr.open_dataset(path, decode_times=TIME_CODER, cache=False)[var]
    da = da.sel(lat=_slice(da.lat.values, min_lat, max_lat), lon=_slice(da.lon.values, min_lon, max_lon))
    return da.reset_coords(drop=True).chunk(chunks)


def _concat(das):
    return xr.concat(das, dim="time", coords="minimal", compat="override", join="override")


def _strip_bounds(da):
    da.attrs.pop("bounds", None)
    da.encoding.pop("bounds", None)
    return da


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def bc_daily(dhist, d21, dhistout, d21out, dobs, var, dec, ref_st, ref_end,
             min_lat, max_lat, min_lon, max_lon, pad=4, buff=15, chunk_lat=32, chunk_lon=32,
             workers=None, overwrite=False, nc_chunk_cache=NC_CHUNK_CACHE):
    """Bias-correct GCM file number ``dec``; returns the output path.

    ``nc_chunk_cache`` is the HDF5 chunk cache in bytes per variable and open
    file; it is set process-wide before any file is opened.
    """
    netCDF4.set_chunk_cache(nc_chunk_cache)
    box = (min_lat, max_lat, min_lon, max_lon)
    chunks = {"time": -1, "lat": chunk_lat, "lon": chunk_lon}
    prcp = "tas" not in var

    files_hist = sorted(glob.glob(os.path.join(dhist, "*.nc")))
    log.info("Found %d 20C GCM files", len(files_hist))
    files21c = sorted(glob.glob(os.path.join(d21, "*.nc")))
    log.info("Found %d 21C GCM files", len(files21c))
    filesgcm = files_hist + files21c

    if dec < len(files_hist):
        fname = filesgcm[dec].replace(dhist, dhistout)
    else:
        fname = filesgcm[dec].replace(d21, d21out)
    if os.path.exists(fname) and not overwrite:
        log.info("%s already exists. Exiting.", fname)
        return fname

    gcm = [_open(f, var, box, chunks) for f in filesgcm]
    gcmcal = gcm[0].time.dt.calendar.lower()
    log.info("GCM calendar = %s", gcmcal)
    ndays = 360 if "360" in gcmcal else 365
    startyrs = [int(g.time.dt.year[0]) for g in gcm]
    endyrs = [int(g.time.dt.year[-1]) for g in gcm]
    fref_st = next(i for i, (s, e) in enumerate(zip(startyrs, endyrs)) if s <= ref_st <= e)
    fref_end = next(i for i, (s, e) in enumerate(zip(startyrs, endyrs)) if s <= ref_end <= e)

    # Aggregated obs files of the reference period
    ls_obsfiles = sorted(glob.glob(os.path.join(dobs, f"{var}_*.nc")))
    ii_stf, ii_endf = 0, len(ls_obsfiles) - 1
    for i, f in enumerate(ls_obsfiles):
        if str(ref_st) in f:
            ii_stf = i
        if str(ref_end) in f:
            ii_endf = i
    obs = _concat([_open(f, var, box, chunks) for f in ls_obsfiles[ii_stf:ii_endf + 1]])
    lats, lons = _strip_bounds(obs.lat.copy()), _strip_bounds(obs.lon.copy())

    # NCL combines GCM and obs by index; give the GCM the obs coordinates so
    # xarray does not try to align slightly different floating point values.
    for i, g in enumerate(gcm):
        if (g.sizes["lat"], g.sizes["lon"]) != (lats.size, lons.size):
            raise ValueError(f"{filesgcm[i]}: lat/lon box {g.sizes['lat']}x{g.sizes['lon']} "
                             f"does not match obs {lats.size}x{lons.size}")
        gcm[i] = g.assign_coords(lat=lats, lon=lons)

    hist = _concat(gcm[fref_st:fref_end + 1])
    hist = hist.isel(time=((hist.time.dt.year >= ref_st) & (hist.time.dt.year <= ref_end)).values)

    decdata = gcm[dec].astype(np.float32)
    dectimes = decdata.time
    if not prcp:
        log.info("Calculating monthly trend")
        trend = monthly_trend(_concat(gcm), ref_st, ref_end, pad)
        trend_dec = _trend_at(trend, dectimes).chunk({"time": -1})
        decdata = decdata - trend_dec

    corrected = xr.apply_ufunc(
        bias_correct_block,
        decdata,
        hist.rename(time="time_hist").chunk({"time_hist": -1}),
        obs.rename(time="time_obs").chunk({"time_obs": -1}),
        input_core_dims=[["time"], ["time_hist"], ["time_obs"]],
        output_core_dims=[["time"]],
        kwargs=dict(doy_gcm=dectimes.dt.dayofyear.values - 1,
                    doy_hist=hist.time.dt.dayofyear.values - 1,
                    doy_obs=obs.time.dt.dayofyear.values - 1,
                    ndays=ndays, buff=buff, prcp=prcp),
        dask="parallelized",
        output_dtypes=[np.float32],
    ).transpose("time", "lat", "lon")
    if not prcp:
        corrected = corrected + trend_dec

    src = gcm[dec]
    out = xr.Dataset({var: corrected.astype(np.float32)})
    out[var].attrs = {k: src.attrs[k] for k in ("standard_name", "long_name", "units") if k in src.attrs}
    out["time"].attrs = {k: v for k, v in dectimes.attrs.items() if k != "bounds"}
    out["lat"].attrs, out["lon"].attrs = lats.attrs, lons.attrs

    nt, ny, nx = out[var].shape
    tile_mb = (nt + hist.sizes["time"] + obs.sizes["time"]) * min(chunk_lat, ny) * min(chunk_lon, nx) * 4 * 3 / 2**20
    nworkers = workers or os.cpu_count()
    log.info("Correcting bias in dec %d: %d tiles, ~%.0f MB per tile, %d workers",
             dec, out[var].data.npartitions, tile_mb, nworkers)

    tenc = dectimes.encoding
    encoding = {
        var: {"zlib": True, "complevel": 5, "_FillValue": FILL, "dtype": "float32",
              "chunksizes": (min(nt, 365), min(chunk_lat, ny), min(chunk_lon, nx))},
        "time": {"units": tenc["units"], "calendar": tenc.get("calendar", gcmcal),
                 "dtype": tenc.get("dtype", "float64"), "_FillValue": None},
        "lat": {"_FillValue": None},
        "lon": {"_FillValue": None},
    }
    log.info("Creating file %s", fname)
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    tmp = fname + ".part"
    with dask.config.set(scheduler="threads", num_workers=nworkers):
        out.to_netcdf(tmp, format="NETCDF4_CLASSIC", unlimited_dims=["time"], encoding=encoding)
    os.replace(tmp, fname)
    return fname


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dhist", required=True, help="directory with historical GCM files")
    p.add_argument("--d21", required=True, help="directory with 21st century GCM files")
    p.add_argument("--dhistout", required=True, help="output directory for historical files")
    p.add_argument("--d21out", required=True, help="output directory for 21st century files")
    p.add_argument("--dobs", required=True, help="directory with obs aggregated to the GCM grid")
    p.add_argument("--var", required=True, help="variable name, e.g. tas or pr")
    p.add_argument("--dec", required=True, type=int, help="index of the GCM file to correct")
    p.add_argument("--ref-st", required=True, type=int, help="first year of reference period")
    p.add_argument("--ref-end", required=True, type=int, help="last year of reference period")
    p.add_argument("--min-lat", type=float, default=-90.0)
    p.add_argument("--max-lat", type=float, default=90.0)
    p.add_argument("--min-lon", type=float, default=0.0)
    p.add_argument("--max-lon", type=float, default=360.0)
    p.add_argument("--pad", type=int, default=4, help="half width (years) of the detrending window")
    p.add_argument("--buff", type=int, default=15, help="half width (days) of the CDF window")
    p.add_argument("--chunk-lat", type=int, default=32, help="tile size in lat (sets memory use)")
    p.add_argument("--chunk-lon", type=int, default=32, help="tile size in lon (sets memory use)")
    p.add_argument("--workers", type=int, default=None, help="dask threads (default: all cores)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--nc-cache-mb", type=float, default=NC_CHUNK_CACHE / 2**20,
                   help="netCDF/HDF5 chunk cache per variable and open file, in MB")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    bc_daily(a.dhist, a.d21, a.dhistout, a.d21out, a.dobs, a.var, a.dec, a.ref_st, a.ref_end,
             a.min_lat, a.max_lat, a.min_lon, a.max_lon, pad=a.pad, buff=a.buff,
             chunk_lat=a.chunk_lat, chunk_lon=a.chunk_lon, workers=a.workers, overwrite=a.overwrite,
             nc_chunk_cache=int(a.nc_cache_mb * 2**20))


if __name__ == "__main__":
    main()
